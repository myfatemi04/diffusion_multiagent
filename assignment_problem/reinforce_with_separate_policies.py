"""
Main gist:
    - Encode the environment as a 2D image overlaid with a graph representing agents and tasks
    - Agents make plans over tasks, taking into account the environment. They also maintain a model
    of what they expect OTHER AGENTS to do, and use this to make better plans
    - What this means is that we will have agent tokens and task tokens. Then, we will use a transformer
    model with a communication mask.
    - Ideally, at some point, we deepen how far ahead the models can look for collaboration. For example,
    if they know that in the long term they want to end up at location X, they might bias themselves
    towards completing intermediate tasks x1...n instead of x'1...n.
How do we define collaborative tasks?
    - Could just be that when two agents reach a task at the same time, they get a bonus. I guess we could
    imagine one of the agents providing supplies to the other.
"""

import json
import os
import random
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import HeteroData
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import tqdm

# HeteroData -> graph with both agent nodes and task nodes (heterogenous)
# to do this, we have a different edge_index for each relationship
def generate_scenario(agents: int, tasks: int, width: int, height: int, show=False):
    n_locations = width * height
    x, y = np.unravel_index(np.random.permutation(n_locations)[:agents + tasks], (width, height))
    agent_locations = np.stack((x[:agents], y[:agents]), axis=1)
    task_locations = np.stack((x[agents:], y[agents:]), axis=1)
    agent_task_distance = np.linalg.norm(
        agent_locations[:, None, :].repeat(tasks, axis=1) -
        task_locations[None, :, :].repeat(agents, axis=0),
        axis=2
    )
    # chooses assignment to minimize cost
    _agent_idx, task_assignment = opt.linear_sum_assignment(agent_task_distance)

    if show:
        print(agent_locations)
        print(task_locations)

        plt.scatter(agent_locations[:, 0], agent_locations[:, 1], color='blue', label='agent locations')
        plt.scatter(task_locations[:, 0], task_locations[:, 1], color='red', label='task locations')
        for agent_i, task_i in zip(range(agents), task_assignment):
            plt.plot([agent_locations[agent_i, 0], task_locations[task_i, 0]], [agent_locations[agent_i, 1], task_locations[task_i, 1]], color='green')

        plt.legend()
        plt.show()

    return agent_locations, task_locations, task_assignment

# heterogenous graph learning: from https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html
# we create a "heterognn"
class GNN(nn.Module):
    def __init__(self, channel_counts):
        super().__init__()
        # these are `lazy`, input_channels=-1 are rederived at first forward() pass
        # and are automatically converted to use the correct message passing functions
        # with heterodata
        convs = []
        lins = []
        for i in range(len(channel_counts)):
            # convs.append(gnn.SAGEConv((-1, -1) if i == 0 else channel_counts[i - 1], channel_counts[i]))
            # lins.append(gnn.Linear(-1, channel_counts[i]))
            convs.append(gnn.GATConv((-1, -1) if i == 0 else channel_counts[i - 1], channel_counts[i], heads=1, dropout=0.1, add_self_loops=False))
            lins.append(gnn.Linear(-1, channel_counts[i]))
        self.convs = nn.ModuleList(convs)
        self.lins = nn.ModuleList(lins)
        self.layernorm = gnn.LayerNorm(channel_counts[-1])
        self.projection = gnn.Linear(channel_counts[-1], 1)

    def forward(self, x, edge_index):
        for i, (conv, lin) in enumerate(zip(self.convs, self.lins)):
            x = conv(x, edge_index) + lin(x)
            if i != len(self.convs) - 1:
                x = x.relu()
        x = self.layernorm(x)
        # Project to output space.
        # Both are dim=1 so this is fine to share.
        # However at some point heterogenous GNNs might suffer...
        return self.projection(x)

def encode_2d_position_as_1d_one_hot_vector(positions: torch.Tensor, width: int, height: int):
    x_indicator = nn.functional.one_hot(positions[..., 0], num_classes=width)
    y_indicator = nn.functional.one_hot(positions[..., 1], num_classes=height)
    return torch.concatenate([x_indicator, y_indicator], dim=-1)

def create_heterodata(agent_locations, task_locations):
    data = HeteroData()
    data['agent'].x = torch.concatenate((
        encode_2d_position_as_1d_one_hot_vector(
            torch.tensor(agent_locations, dtype=torch.long),
            width, height
        ).float(),
        # Additional flags:
        # 1. Whether to treat this node as "self" or another agent
        # 2... (Other things at some point)
        torch.zeros((agent_locations.shape[0], 1), dtype=torch.float),
    ), dim=-1)
    data['task'].x = torch.concatenate((
        encode_2d_position_as_1d_one_hot_vector(
            torch.tensor(task_locations, dtype=torch.long),
            width, height
        ).float(),
        # No additional flags.
    ), dim=-1)
    # tasks <-> agents
    data['agent', 'sees', 'task'].edge_index = torch.tensor([
        [i for i in range(agent_locations.shape[0]) for j in range(task_locations.shape[0])],
        [j for i in range(agent_locations.shape[0]) for j in range(task_locations.shape[0])]
    ], dtype=torch.long)
    data['task', 'sees', 'agent'].edge_index = torch.tensor([
        [i for i in range(task_locations.shape[0]) for j in range(agent_locations.shape[0])],
        [j for i in range(task_locations.shape[0]) for j in range(agent_locations.shape[0])]
    ], dtype=torch.long)
    # agent <-> agent, fully-connected
    data['agent', 'sees', 'agent'].edge_index = torch.tensor([
        [i for i in range(agent_locations.shape[0]) for j in range(agent_locations.shape[0])],
        [j for i in range(agent_locations.shape[0]) for j in range(agent_locations.shape[0])]
    ], dtype=torch.long)
    return data

def calculate_assignment_reward_per_agent(choices, agent_locations, task_locations):
    """
    Returns total reward of the provided assignment.
    """
    agent_task_distance = np.linalg.norm(
        agent_locations[:, None, :].repeat(len(task_locations), axis=1) -
        task_locations[None, :, :].repeat(len(agent_locations), axis=0),
        axis=2
    )
    # calculate a value for each agent
    agent_values = [0.] * len(agent_locations)
    for task_id in range(n_tasks):
        least_cost = None
        least_cost_agent = None
        for agent_id, choice in enumerate(choices):
            if choice == task_id:
                if least_cost is None or agent_task_distance[choice, task_id] < least_cost:
                    least_cost = agent_task_distance[choice, task_id]
                    least_cost_agent = agent_id
        if least_cost is not None:
            assert least_cost_agent is not None
            # 10 can be reconfigured to mean a decay rate
            # agent_values[least_cost_agent] = 1 * np.exp(-least_cost / 40)
            # give a reward of 1
            agent_values[least_cost_agent] = 1 # 1 * np.exp(-least_cost / 40)
    # calculate cost incurred by moving far
    for agent_id, choice in enumerate(choices):
        movement_cost = (1/100)
        agent_values[agent_id] -= float(np.linalg.norm(agent_locations[agent_id] - task_locations[choice])) * movement_cost

    return agent_values

def multiagent_forward(net, sample):
    """ Returns logprobs and values for each agent. """
    # Run separate policies for each agent to calculate corresponding logits and values
    # Run REINFORCE with baseline
    logits_per_agent = []
    value_estimates_per_agent = []

    for agent_id in range(n_agents):
        # update x_dict to have current agent flag set appropriately
        new_x_dict = sample.x_dict
        new_x_dict['agent'] = new_x_dict['agent'].clone()
        new_x_dict['agent'][agent_id, -1] = 1

        out = net(new_x_dict, sample.edge_index_dict)

        value_estimates_per_agent.append(
            out['agent'][agent_id, 0]
        )
        logits_per_agent.append(
            out['task'][:, 0]
        )
    
    logprobs_per_agent = torch.stack(logits_per_agent).log_softmax(dim=-1)

    return logprobs_per_agent, value_estimates_per_agent

# we will first use a contextualized bandit and make decisions with a gnn
# will just use one-hot encoding for x and y positions
width = 100
height = 100
n_agents = 10
n_tasks = 10
n_scenarios = 10000

def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    data: List[Tuple[HeteroData, np.ndarray, np.ndarray, np.ndarray]] = []

    for idx in tqdm.tqdm(range(n_scenarios), desc='Generating scenarios'):
        agent_locations, task_locations, task_assignment = generate_scenario(n_agents, n_tasks, width, height)
        data.append((create_heterodata(agent_locations, task_locations), agent_locations, task_locations, task_assignment))

    dummy = data[0][0]

    net = GNN([64, 64])
    net = gnn.to_hetero(net, dummy.metadata(), aggr='sum')

    # populate the channel sizes by passing in a dummy dataset of the same shape
    with torch.no_grad():
        net(dummy.x_dict, dummy.edge_index_dict)

    skip = False
    # os.chdir('runs/run_27_reinforce_with_separate_policies')
    # net.load_state_dict(torch.load("model.pth"))
    # skip = True

    split = 0.9
    train = data[:int(split * n_scenarios)]
    test = data[int(split * n_scenarios):]

    loss_hist = []
    reward_hist = []

    if not skip:
        temperature = 0.1
        epochs = 2
        initial_lr = 1e-4
        end_lr = 1e-6

        alg = 'reinforce_with_separate_policies'

        import wandb
        wandb.init(
            # set the wandb project where this run will be logged
            project="arl-collab-planning",
            # track hyperparameters and run metadata
            config={
                "lr_schedule": "cosine_annealing",
                "initial_lr": initial_lr,
                "end_lr": end_lr,
                "architecture": alg,
                "epochs": epochs,
                "n_scenarios": n_scenarios,
                "n_agents": n_agents,
                "n_tasks": n_tasks,
            }
        )

        optim = torch.optim.Adam([*net.parameters()], lr=initial_lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_scenarios * epochs, eta_min=end_lr)
        for ep in range(epochs):
            for (sample, agent_locations, task_locations, task_assignment) in (pbar := tqdm.tqdm(train, desc=f'Epoch {ep}')):
                logprobs_per_agent, value_estimates_per_agent = multiagent_forward(net, sample)

                # Make choices for each agent
                # choices_per_agent = [
                #     torch.multinomial(F.softmax(logits / temperature, dim=0), 1).item()
                #     for logits in logprobs_per_agent
                # ]
                choices_per_agent = logprobs_per_agent.argmax(dim=1).numpy()

                reward_per_agent = calculate_assignment_reward_per_agent(choices_per_agent, agent_locations, task_locations)
                value_loss = F.mse_loss(torch.stack(value_estimates_per_agent), torch.tensor(reward_per_agent))
                chosen_logprobs_tensor = logprobs_per_agent.gather(1, torch.tensor(choices_per_agent).view(-1, 1))
                delta = (
                    torch.tensor(reward_per_agent)
                ).detach()
                policy_loss = (-chosen_logprobs_tensor * delta).mean()
                
                loss = value_loss + policy_loss

                optim.zero_grad()
                loss.backward()
                optim.step()
                lr_scheduler.step()
                pbar.set_postfix(loss=f"{loss.item():.3e}")

                loss_hist.append(loss.item())
                reward_hist.append(sum(reward_per_agent))

                wandb.log({
                    "loss": loss.item(),
                    "value_sum": sum(reward_per_agent),
                    "value_per_agent": sum(reward_per_agent)/n_agents,
                    "temperature": temperature,
                    "lr": optim.param_groups[0]['lr'],
                })

            # reduce gamma each epoch
            # temperature *= torch.pow(torch.tensor(0.1), 1/epochs)

        run_id = 0
        os.listdir()
        while any([f'run_{run_id}_' in f for f in os.listdir('runs')]):
            run_id += 1
        os.makedirs(f'runs/run_{run_id}_{alg}')
        os.chdir(f'runs/run_{run_id}_{alg}')
        np.save('loss_hist.npy', loss_hist)
        np.save('value_hist.npy', reward_hist)
        with open("info.json", "w") as f:
            json.dump({
                "alg": alg,
                "gamma": float(temperature),
                "n_agents": n_agents,
                "n_tasks": n_tasks,
                "n_scenarios": n_scenarios,
                "width": width,
                "height": height,
                "epochs": epochs,
            }, f)

        torch.save(net.state_dict(), "model.pth")

        # plot loss_hist
        loss_hist = np.array(loss_hist)
        loss_hist = np.convolve(loss_hist, np.ones(100) / 100, mode='valid')
        plt.subplot(2, 1, 1)
        plt.plot(loss_hist)
        plt.title("Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss (log)")
        plt.yscale('log')
        # plot value_hist
        reward_hist = np.array(reward_hist)
        reward_hist = np.convolve(reward_hist, np.ones(100) / 100, mode='valid')
        plt.subplot(2, 1, 2)
        plt.plot(reward_hist)
        plt.title("Value")
        plt.xlabel("Step")
        plt.ylabel("Value")
        # save
        plt.tight_layout()
        plt.savefig("loss_value.png")
        plt.show()

    # eval
    softs = []
    greedys = []
    trues = []
    randoms = []
    with torch.no_grad():
        temperature = 0.5
        for i, (sample, agent_locations, task_locations, task_assignment) in (pbar := tqdm.tqdm(enumerate(test), desc='Evaluating...')):
            display = (i == 0) and True

            logprobs_per_agent, value_estimates_per_agent = multiagent_forward(net, sample)
            
            choices_per_agent = [
                torch.multinomial(F.softmax(logits / temperature, dim=0), 1).item()
                for logits in logprobs_per_agent
            ]
            choices_per_agent_greedy = logprobs_per_agent.argmax(dim=1).numpy()

            random_choices = [random.randint(0, n_tasks - 1) for _ in range(n_agents)]

            soft_eval_value = sum(calculate_assignment_reward_per_agent(choices_per_agent, agent_locations, task_locations))
            greedy_eval_value = sum(calculate_assignment_reward_per_agent(choices_per_agent_greedy, agent_locations, task_locations))
            true_value = sum(calculate_assignment_reward_per_agent(task_assignment, agent_locations, task_locations))
            random_value = sum(calculate_assignment_reward_per_agent(random_choices, agent_locations, task_locations))
            softs.append(soft_eval_value)
            greedys.append(greedy_eval_value)
            trues.append(true_value)
            randoms.append(random_value)

            if display:
                print("soft eval value:", soft_eval_value)
                print("greedy eval value:", greedy_eval_value)
                print("true value:", true_value)
                print("random value:", random_value)
                print("pred assignment:", choices_per_agent)
                print("greedy assignment:", choices_per_agent_greedy)
                print("true assignment:", task_assignment)
                print()
                
                # plot scores matrix
                plt.title("Scores matrix")
                plt.xlabel("Task")
                plt.ylabel("Agent")
                plt.imshow(logprobs_per_agent.detach().numpy())
                plt.colorbar()
                plt.show()

                # plot soft value
                plt.subplot(2, 1, 1)
                plt.scatter(agent_locations[:, 0], agent_locations[:, 1], color='blue', label='agent locations')
                plt.scatter(task_locations[:, 0], task_locations[:, 1], color='red', label='task locations')
                for agent_i, task_i in zip(range(len(agent_locations)), task_assignment):
                    plt.plot(
                        [agent_locations[agent_i, 0], task_locations[task_i, 0]],
                        [agent_locations[agent_i, 1], task_locations[task_i, 1]],
                        color='green', label='true' if agent_i == 0 else None, linewidth=3
                    )
                for agent_i, task_i in zip(range(len(agent_locations)), choices_per_agent):
                    task_i = int(task_i)
                    plt.plot(
                        [agent_locations[agent_i, 0] + 0.5, task_locations[task_i, 0] + 0.5],
                        [agent_locations[agent_i, 1] + 0.5, task_locations[task_i, 1] + 0.5],
                        color='purple', label='pred' if agent_i == 0 else None, linewidth=3
                    )

                plt.legend()

                # plot greedy value
                plt.subplot(2, 1, 2)
                plt.scatter(agent_locations[:, 0], agent_locations[:, 1], color='blue', label='agent locations')
                plt.scatter(task_locations[:, 0], task_locations[:, 1], color='red', label='task locations')
                for agent_i, task_i in zip(range(len(agent_locations)), task_assignment):
                    plt.plot(
                        [agent_locations[agent_i, 0], task_locations[task_i, 0]],
                        [agent_locations[agent_i, 1], task_locations[task_i, 1]],
                        color='green', label='true' if agent_i == 0 else None, linewidth=3
                    )
                for agent_i, task_i in zip(range(len(agent_locations)), choices_per_agent_greedy):
                    task_i = int(task_i)
                    plt.plot(
                        [agent_locations[agent_i, 0] + 0.5, task_locations[task_i, 0] + 0.5],
                        [agent_locations[agent_i, 1] + 0.5, task_locations[task_i, 1] + 0.5],
                        color='purple', label='pred' if agent_i == 0 else None, linewidth=3
                    )

                plt.legend()
                plt.show()

    trues = np.array(trues)
    print("mean true:", trues.mean())
    print("std true:", trues.std())

    softs = np.array(softs)
    print("mean soft:", softs.mean())
    print("std soft:", softs.std())

    greedys = np.array(greedys)
    print("mean greedy:", greedys.mean())
    print("std greedy:", greedys.std())

    randoms = np.array(randoms)
    print("mean random:", randoms.mean())
    print("std random:", randoms.std())

if __name__ == '__main__':
    main()

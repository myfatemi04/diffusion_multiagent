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

    def forward(self, x, edge_index):
        for i, (conv, lin) in enumerate(zip(self.convs, self.lins)):
            x = conv(x, edge_index) + lin(x)
            if i != len(self.convs) - 1:
                x = x.relu()
        # x = self.conv1(x, edge_index) + self.lin1(x)
        # x = x.relu()
        # x = self.conv2(x, edge_index) + self.lin2(x)
        # so the task and policy embeddings are reasonable
        x = self.layernorm(x)
        return x

def one_hot_2d(positions: torch.Tensor, width: int, height: int):
    x_emb = nn.functional.one_hot(positions[..., 0], num_classes=width)
    y_emb = nn.functional.one_hot(positions[..., 1], num_classes=height)
    return torch.concatenate([x_emb, y_emb], dim=-1)

def create_heterodata(agent_locations, task_locations):
    data = HeteroData()
    data['agent'].x = one_hot_2d(torch.tensor(agent_locations, dtype=torch.long), width, height).float()
    data['task'].x = one_hot_2d(torch.tensor(task_locations, dtype=torch.long), width, height).float()
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

def evaluate_assignment(choices, agent_locations, task_locations):
    agent_task_distance = np.linalg.norm(
        agent_locations[:, None, :].repeat(len(task_locations), axis=1) -
        task_locations[None, :, :].repeat(len(agent_locations), axis=0),
        axis=2
    )
    # calculate a value for each agent
    # by default, agents get penalized for not finding a task
    agent_values = [-1.0] * len(agent_locations)
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
    # for agent_id, choice in enumerate(choices):
    #     movement_cost = (1/100 * 1/100)
    #     agent_values[agent_id] -= float(np.linalg.norm(agent_locations[agent_id] - task_locations[choice])) * movement_cost
    return agent_values

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
        out = net(dummy.x_dict, dummy.edge_index_dict)
        # now we can see that the output is a dictionary with keys 'agent' and 'task'
        # print(out.keys())
    # used in CLIP, going to use it here
    scale = torch.nn.Parameter(torch.tensor(1.0))

    skip = False
    # os.chdir('runs/run_7')
    # net.load_state_dict(torch.load("model.pth"))
    # skip = True

    split = 0.9
    train = data[:int(split * n_scenarios)]
    test = data[int(split * n_scenarios):]

    loss_hist = []
    value_hist = []

    if not skip:
        temperature = 0.1
        epochs = 25
        initial_lr = 1e-4
        end_lr = 1e-6

        import wandb
        wandb.init(
            # set the wandb project where this run will be logged
            project="arl-collab-planning",
            # track hyperparameters and run metadata
            config={
                "lr_schedule": "cosine_annealing",
                "initial_lr": initial_lr,
                "end_lr": end_lr,
                "architecture": "squarecb-0.2-factorized",
                "n_scenarios": n_scenarios,
                "epochs": epochs,
            }
        )

        optim = torch.optim.Adam([*net.parameters()], lr=initial_lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_scenarios * epochs, eta_min=end_lr)
        for ep in range(epochs):
            for (sample, agent_locations, task_locations, task_assignment) in (pbar := tqdm.tqdm(train, desc=f'Epoch {ep}')):
                out = net(sample.x_dict, sample.edge_index_dict)
                # create score matrix
                scores: torch.Tensor = (out['agent'] @ out['task'].T) * scale
                # choose which nodes to assign selves to through some exploration method
                # (for example, SquareCB). shape is [n_agents, n_tasks]
                gap = scores.max(dim=1, keepdim=True).values - scores
                # inverse gap weighting
                # Will calculate the actual lambda later, for now will just to softmax
                p = F.softmax(-gap/temperature, dim=1)
                soft_choices = [torch.multinomial(p[i], 1, replacement=False)[0].item() for i in range(n_agents)]
                # p = 1/(lda + gamma * gap)

                # calculate value of assignment
                # choices = list(scores.argmax(dim=1).detach().numpy())
                values = evaluate_assignment(soft_choices, agent_locations, task_locations)
                # value2 = evaluate_assignment(task_assignment, agent_locations, task_locations)
                # output has shape [n_agents, n_tasks], and task_assignment has shape [n_agents]
                # train outputs to approximate log-scaled value
                # logvalue = torch.log1p(torch.tensor(value, dtype=torch.float))
                loss = ((scores[torch.arange(0, n_agents), soft_choices] - torch.tensor(values).float()) ** 2).mean()

                optim.zero_grad()
                loss.backward()
                optim.step()
                lr_scheduler.step()
                pbar.set_postfix(loss=f"{loss.item():.3e}")

                loss_hist.append(loss.item())
                value_hist.append(sum(values))
                wandb.log({
                    "loss": loss.item(),
                    "value_sum": sum(values),
                    "value_per_agent": sum(values)/n_agents,
                    "temperature": temperature,
                    "lr": optim.param_groups[0]['lr'],
                })
            # reduce gamma each epoch
            # temperature *= torch.pow(torch.tensor(0.1), 1/epochs)

        alg = "squarecb-0.2-factorized"
        run_id = 0
        os.listdir()
        while any([f'run_{run_id}_' in f for f in os.listdir('runs')]):
            run_id += 1
        os.makedirs(f'runs/run_{run_id}_{alg}')
        os.chdir(f'runs/run_{run_id}_{alg}')
        np.save('loss_hist.npy', loss_hist)
        np.save('value_hist.npy', value_hist)
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
        value_hist = np.array(value_hist)
        value_hist = np.convolve(value_hist, np.ones(100) / 100, mode='valid')
        plt.subplot(2, 1, 2)
        plt.plot(value_hist)
        plt.title("Value")
        plt.xlabel("Step")
        plt.ylabel("Value")
        # save
        plt.tight_layout()
        plt.savefig("loss_value.png")
        plt.show()

    # eval
    with torch.no_grad():
        temperature = 0.5
        for (sample, agent_locations, task_locations, task_assignment) in test:
            out = net(sample.x_dict, sample.edge_index_dict)

            # create score matrix
            scores: torch.Tensor = (out['agent'] @ out['task'].T) * scale
            # choose which nodes to assign selves to through some exploration method
            # (for example, SquareCB). shape is [n_agents, n_tasks]
            gap = scores.max(dim=1, keepdim=True).values - scores
            # inverse gap weighting
            # Will calculate the actual lambda later, for now will just to softmax
            p = F.softmax(-gap/temperature, dim=1)
            soft_choices = [torch.multinomial(p[i], 1, replacement=False)[0].item() for i in range(n_agents)]
            greedy_choices = list(scores.argmax(dim=1).detach().numpy())
            crossentropy = F.cross_entropy(scores, torch.tensor(task_assignment).long())
            print("eval crossentropy:", crossentropy.item())
            print("soft eval value:", sum(evaluate_assignment(soft_choices, agent_locations, task_locations)))
            print("greedy eval value:", sum(evaluate_assignment(greedy_choices, agent_locations, task_locations)))
            print("true value:", sum(evaluate_assignment(task_assignment, agent_locations, task_locations)))
            print("pred assignment:", soft_choices)
            print("greedy assignment:", greedy_choices)
            print("true assignment:", task_assignment)
            print()
            
            # plot scores matrix
            plt.title("Scores matrix")
            plt.xlabel("Task")
            plt.ylabel("Agent")
            plt.imshow(scores.numpy())
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
            for agent_i, task_i in zip(range(len(agent_locations)), soft_choices):
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
            for agent_i, task_i in zip(range(len(agent_locations)), greedy_choices):
                plt.plot(
                    [agent_locations[agent_i, 0] + 0.5, task_locations[task_i, 0] + 0.5],
                    [agent_locations[agent_i, 1] + 0.5, task_locations[task_i, 1] + 0.5],
                    color='purple', label='pred' if agent_i == 0 else None, linewidth=3
                )

            plt.legend()
            plt.show()

            exit()

if __name__ == '__main__':
    main()

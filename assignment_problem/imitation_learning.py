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
import visualize
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

width = 100
height = 100
n_agents = 10
n_tasks = 10
n_scenarios = 10000

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

def create_heterodata(agent_locations, task_locations, width, height):
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

def evaluate_assignment_0(choices, agent_locations, task_locations):
    agent_task_distance = np.linalg.norm(
        agent_locations[:, None, :].repeat(len(task_locations), axis=1) -
        task_locations[None, :, :].repeat(len(agent_locations), axis=0),
        axis=2
    )
    completion_value = (width ** 2 + height ** 2) ** 0.5
    # credit assignment will become an interesting subproblem
    value = 0
    for i in range(n_tasks):
        least_cost = None
        for choice in choices:
            if choice == i:
                if least_cost is None or agent_task_distance[choice, i] < least_cost:
                    least_cost = agent_task_distance[choice, i]
        if least_cost is not None:
            value += completion_value - least_cost
    return value

def evaluate_assignment(choices, agent_locations, task_locations):
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

    return sum(agent_values)

def main():
    import random
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # we will first use a contextualized bandit and make decisions with a gnn
    # will just use one-hot encoding for x and y positions
    data: List[Tuple[HeteroData, np.ndarray, np.ndarray, np.ndarray]] = []

    for idx in tqdm.tqdm(range(n_scenarios), desc='Generating scenarios'):
        agent_locations, task_locations, task_assignment = generate_scenario(n_agents, n_tasks, width, height)
        data.append((create_heterodata(agent_locations, task_locations, width, height), agent_locations, task_locations, task_assignment))

    dummy = data[0][0]

    sizes = [128, 128, 64]
    net = GNN(sizes)
    net = gnn.to_hetero(net, dummy.metadata(), aggr='sum')

    # populate the channel sizes by passing in a dummy dataset of the same shape
    with torch.no_grad():
        out = net(dummy.x_dict, dummy.edge_index_dict)
        # now we can see that the output is a dictionary with keys 'agent' and 'task'
        # print(out.keys())
    # used in CLIP, going to use it here
    scale = torch.nn.Parameter(torch.tensor(1.0))

    path = 'runs/run_28_imitationlearning'
    split = 0.9
    train = data[:int(split * n_scenarios)]
    test = data[int(split * n_scenarios):]

    if os.path.exists(path):
        net.load_state_dict(torch.load(f'{path}/model.pth'))
        net.eval()
    else:
        optim = torch.optim.Adam([*net.parameters(), scale], lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_scenarios, eta_min=1e-6)

        loss_hist = []
        value_hist = []

        epochs = 1
        for ep in range(epochs):
            for (sample, agent_locations, task_locations, task_assignment) in (pbar := tqdm.tqdm(train, desc=f'Epoch {ep}')):
                out = net(sample.x_dict, sample.edge_index_dict)
                # create assignment matrix
                logprobs_table: torch.Tensor = (out['agent'] @ out['task'].T) * scale
                loss = F.cross_entropy(logprobs_table, torch.tensor(task_assignment).long())
                # calculate value of assignment
                choices = list(logprobs_table.argmax(dim=1).detach().numpy())
                value = evaluate_assignment(choices, agent_locations, task_locations)

                optim.zero_grad()
                loss.backward()
                optim.step()
                lr_scheduler.step()
                pbar.set_postfix(loss=f"{loss.item():.3e}")
                loss_hist.append(loss.item())
                value_hist.append(value)

        alg = "imitationlearning"
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
                "n_agents": n_agents,
                "n_tasks": n_tasks,
                "n_scenarios": n_scenarios,
                "width": width,
                "height": height,
                "epochs": epochs,
                "sizes": sizes,
            }, f)

        visualize.plot_loss_and_reward(loss_hist, value_hist)

        torch.save(net.state_dict(), "model.pth")

    evaluation = []
    randoms = []
    trues = []

    # eval
    with torch.no_grad():
        for i, (sample, agent_locations, task_locations, task_assignment) in (pbar := tqdm.tqdm(enumerate(test), desc='Evaluating')):
            display = (i == 0) and True

            out = net(sample.x_dict, sample.edge_index_dict)
            logprobs_table = (out['agent'] @ out['task'].T) * scale

            eval_value, true_value, random_value = visualize.evaluate(
                logprobs_table,
                agent_locations,
                task_locations,
                task_assignment,
                display=display
            )
            
            evaluation.append(eval_value)
            trues.append(true_value)
            randoms.append(random_value)
    
    evaluation = np.array(evaluation)
    print("mean evaluation:", evaluation.mean())
    print("std evaluation:", evaluation.std())
    trues = np.array(trues)
    print("mean true:", trues.mean())
    print("std true:", trues.std())
    randoms = np.array(randoms)
    print("mean random:", randoms.mean())
    print("std random:", randoms.std())

if __name__ == '__main__':
    main()

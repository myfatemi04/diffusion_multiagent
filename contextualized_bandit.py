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

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.data import HeteroData
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

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
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # these are `lazy`
        self.conv1 = gnn.SAGEConv((-1, -1), hidden_channels, dropout=0.1)
        self.lin1 = gnn.Linear(-1, hidden_channels)
        self.conv2 = gnn.SAGEConv(hidden_channels, out_channels, dropout=0.1)
        self.lin2 = gnn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x

# we will first use a contextualized bandit and make decisions with a gnn
# will just use one-hot encoding for x and y positions
agent_locations, task_locations, task_assignment = generate_scenario(10, 10, 100, 100)
data = HeteroData()
data['agent'].x = torch.tensor(agent_locations, dtype=torch.float)
data['task'].x = torch.tensor(task_locations, dtype=torch.float)
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
print(data)

net = GNN(64, 64)
net = gnn.to_hetero(net, data.metadata(), aggr='sum')

out = net(data.x_dict, data.edge_index_dict)
# now we can see that the output is a dictionary with keys 'agent' and 'task'
print(out.keys())

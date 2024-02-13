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
import matplotlib.pyplot as plt
import tqdm
import visualize
from data import generate_scenario, create_heterodata
import reward_functions

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

def sample_actions(logprobs_table: torch.Tensor, temperature=0.1):
    """
    temperature=0: greedy
    otherwise, divide all logprobs by temperature and do softmax
    """
    if temperature == 0:
        return logprobs_table.argmax(dim=1).numpy()
    else:
        return np.array([torch.multinomial(F.softmax(logprobs / temperature), 1).item() for logprobs in logprobs_table])

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
        data.append((create_heterodata(width, height, agent_locations, task_locations), agent_locations, task_locations, task_assignment))

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
                choices_per_agent = sample_actions(logprobs_per_agent, temperature)

                reward_per_agent: list[float] = reward_functions.evaluate_assignment(choices_per_agent, agent_locations, task_locations) # type: ignore
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

        visualize.plot_loss_and_reward(loss_hist, reward_hist)

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

            greedy_eval_value, true_value, random_value = visualize.evaluate(logprobs_per_agent, agent_locations, task_locations, task_assignment, display=display)
            greedys.append(greedy_eval_value)
            trues.append(true_value)
            randoms.append(random_value)

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

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from reward_functions import evaluate_assignment

def plot_loss_and_reward(loss_hist, value_hist):
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

def evaluate(logprobs_table, agent_locations, task_locations, task_assignment, reward_fn=evaluate_assignment, display=False):
    loss = F.cross_entropy(logprobs_table, torch.tensor(task_assignment).long())
    neural_assn = logprobs_table.argmax(dim=1).numpy()
    eval_value = reward_fn(neural_assn, agent_locations, task_locations)
    true_value = reward_fn(task_assignment, agent_locations, task_locations)
    random_choices = [torch.randint(0, len(task_locations) - 1, ()).item() for _ in range(len(agent_locations))]
    random_value = reward_fn(random_choices, agent_locations, task_locations)
    
    if display:
        print("eval crossentropy:", loss.item())
        print("eval value:", eval_value)
        print("true value:", true_value)
        print("pred assignment:", neural_assn)
        print("true assignment:", task_assignment)
        print()

        # plot scores matrix
        plt.title("Scores matrix")
        plt.xlabel("Task")
        plt.ylabel("Agent")
        plt.imshow(logprobs_table.numpy())
        plt.colorbar()
        plt.savefig("scores_matrix.png")
        plt.show()

        plt.scatter(agent_locations[:, 0], agent_locations[:, 1], color='blue', label='agent locations')
        plt.scatter(task_locations[:, 0], task_locations[:, 1], color='red', label='task locations')
        for agent_i, task_i in zip(range(len(agent_locations)), task_assignment):
            plt.plot(
                [agent_locations[agent_i, 0], task_locations[task_i, 0]],
                [agent_locations[agent_i, 1], task_locations[task_i, 1]],
                color='green', label='true' if agent_i == 0 else None, linewidth=3
            )
        for agent_i, task_i in zip(range(len(agent_locations)), neural_assn):
            plt.plot(
                [agent_locations[agent_i, 0] + 0.5, task_locations[task_i, 0] + 0.5],
                [agent_locations[agent_i, 1] + 0.5, task_locations[task_i, 1] + 0.5],
                color='purple', label='pred' if agent_i == 0 else None, linewidth=3
            )

        plt.legend()
        plt.show()

    return (eval_value, true_value, random_value)

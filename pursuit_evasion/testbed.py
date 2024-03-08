"""
Approach:
 - Render the scene as high-dimensional feature map
   centered at the agent's current position
 - Patchify the feature map and use the patches as
   tokens in a transformer
 - Use the transformer to predict the best action to
   take with a specialized "action readout" head
"""

import copy
import random
import sys

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import traceback

from marl import MultiAgentEpisode
from episode_collection import collect_episode
import grid_world_environment as E
from transformer import TransformerNetwork
import visualizer

def log_reward_statistics(episodes: list[MultiAgentEpisode]):
  # Log reward statistics
  mean_episode_reward = sum(
    sum(tup.reward.sum() for tup in episode.steps)
    for episode in episodes
  ) / len(episodes)
  mean_episode_length = sum(len(episode.steps) for episode in episodes) / len(episodes)
  mean_reached_goal = sum(len(episode.steps[-1].global_state.successful_evaders) for episode in episodes) / len(episodes)
  mean_caught_evaders = sum(len(episode.steps[-1].global_state.caught_evaders) for episode in episodes) / len(episodes)

  wandb.log({
    'total_reward': mean_episode_reward,
    'episode_length': mean_episode_length,
    'successful_evaders': mean_reached_goal,
    'caught_evaders': mean_caught_evaders,
  })

def visualize_episode(episode: MultiAgentEpisode):
  for step in episode.steps:
    print("reward:", step.reward)
    print("done:", step.done)
    print("probs:", step.action_probs)
    print("selection:", step.action_selection)
    plt.title("Episode Visualizer")
    visualizer.render_scene(
      {
        agent_id: {
          "xy": (step.global_state.agent_positions[agent_id].x, step.global_state.agent_positions[agent_id].y),
          "color": "red",
          "action_probs": step.action_probs[agent_id].tolist(), # type: ignore
          # "action_values": action_values_per_agent[agent].tolist(),
        }
        for agent_id in episode.agents
        if step.action_space[agent_id] is not None
      },
      {
        f"target": {
          "xy": step.global_state.evader_target_location,
          "color": "blue"
        }
      },
    )

def calculate_value_function_loss(valuefunction: TransformerNetwork, episode: MultiAgentEpisode):
  # Input features are restricted to agents in the `active_mask`.
  valuefunction_per_step: list[torch.Tensor] = [
    valuefunction(*episode_step.global_input_features)[0, :, 0]
    for episode_step in episode.steps
  ]
  discounted_rewards_per_step = [
    episode_step.discounted_reward[episode_step.active_mask].to(valuefunction_per_step[0].device)
    for episode_step in episode.steps
  ]
  loss = F.smooth_l1_loss(
    torch.cat(valuefunction_per_step, dim=0),
    torch.cat(discounted_rewards_per_step, dim=0),
  )
  return loss, [t.detach() for t in valuefunction_per_step]

def calculate_policy_losses_for_agent(episode: MultiAgentEpisode, agent_i: int, policy: TransformerNetwork, policy_ref: TransformerNetwork, valuefunction_per_step: list[torch.Tensor]):
  agent_id = episode.agents[agent_i]
  selected_action_logprobs = []
  selected_action_logprobs_ref = []
  values = []
  entropies = []

  # Aggregate relevant information from episode
  for step_i, episode_step in enumerate(episode.steps):
    # Check if the agent was active at this time
    if not episode_step.active_mask[agent_i]:
      continue

    action_space = episode_step.action_space[agent_id]
    local_input_features = episode_step.local_input_features_per_agent[agent_id]
    action_selection = int(episode_step.action_selection[agent_i])
    
    assert local_input_features is not None and action_selection is not None, "Agent should have input features and action selection if it is active"
    assert action_selection in action_space, f"Action selection was not in action space. Action selection: {action_selection}. Action space: {action_space}"

    # Forward pass: Get action logits and value.
    # We store a mapping between nodes in the local subgraph (which are numbered 0...n)
    # and the agent IDs that correspond to them.
    out = policy(*local_input_features)
    # first 0 is for batch, second 0 is for agent token (token index 0 is "ego")
    # mask out tokens that are not in the action space at this timestep
    logits = out[0, 0][action_space]

    with torch.no_grad():
      out_ref = policy_ref(*local_input_features)
      logits_ref = out_ref[0, 0][action_space]

    # Store logprobs for executed policy
    logprobs = torch.log_softmax(logits, dim=-1)
    logprobs_ref = torch.log_softmax(logits_ref, dim=-1)
    action_selection_index = action_space.index(action_selection)
    selected_action_logprobs.append(logprobs[action_selection_index])
    selected_action_logprobs_ref.append(logprobs_ref[action_selection_index])

    # Store entropy
    entropy = -torch.sum(logprobs * torch.exp(logprobs), dim=-1)
    entropies.append(entropy)

    # Store predicted value for this step (pre-computed, but added to an array through this loop)
    index_in_valuefunction = episode_step.active_mask[:agent_i].sum()
    values.append(valuefunction_per_step[step_i][index_in_valuefunction])

  selected_action_logprobs = torch.stack(selected_action_logprobs)
  selected_action_logprobs_ref = torch.stack(selected_action_logprobs_ref).detach()
  entropies = torch.stack(entropies)

  discounted_rewards = torch.stack([
    step.discounted_reward[agent_i]
    for step in episode.steps
    if step.active_mask[agent_i]
  ]).to(values[0].device)

  # PPO loss
  ratios = selected_action_logprobs - selected_action_logprobs_ref
  clipped_ratios = torch.clamp(ratios, -0.2, 0.2)
  advantage = discounted_rewards - torch.stack(values).detach()

  # print advantage per action
  # if ppo_iter == 0 and plot_graph:
  #   print("### Agent:", agent_id)

  #   advantage_per_action = {}
  #   for step_i, step in enumerate(episode.steps):
  #     action_selection = step.action_selection[agent_i]
  #     advantage_per_action[action_selection] = advantage[step_i].item()
    
  #   print("advantage_per_action:", advantage_per_action)
  #   print("discounted rewards:", discounted_rewards)
  #   print("values:", values)

  # loss could be really really positive if, for example, ratios * advantage were really negative.
  actor_loss = -torch.min(ratios * advantage, clipped_ratios * advantage).mean()
  mean_entropy = entropies.mean()

  return (actor_loss, mean_entropy)

def main():
  torch.random.manual_seed(0)
  np.random.seed(0)
  random.seed(0)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  print("Using device:", device)

  #region Initialization

  # initial_lr = 1e-3
  # end_lr = 1e-5
  n_scenarios = 1
  n_agents = 1
  n_tasks = 2
  alg = 'v001-pursuit-evasion-transformers'
  n_batches = 1000000
  n_ppo_iterations = 1
  n_batch_episodes = 16
  lr = 1e-3

  # Make an epsilon value that decays exponentially from 0.5 to 0.005 over the first 10000 episodes, then goes to 0.
  start_epsilon = 0.5
  end_epsilon = 0.005
  epsilon_decay = 500

  entropy_weight = 5e-3

  use_wandb = '--debug' not in sys.argv
  if not use_wandb:
    wandb.init(mode="disabled")
  else:
    wandb.init(
      # set the wandb project where this run will be logged
      project="arl-collab-planning",
      # track hyperparameters and run metadata
      config={
        "lr_schedule": "constant",
        "environment": "randomized",
        # "initial_lr": initial_lr,
        # "end_lr": end_lr,
        "lr": lr,
        "architecture": alg,
        "n_episodes": n_batches,
        "n_scenarios": n_scenarios,
        "n_agents": n_agents,
        "n_tasks": n_tasks,
        "n_ppo_iterations": n_ppo_iterations,
        "n_batch_episodes": n_batch_episodes,
        "start_epsilon": start_epsilon,
        "end_epsilon": end_epsilon,
        "epsilon_decay": epsilon_decay,
        "entropy_weight": entropy_weight,
        "conv_layer": "SAGEConv",
      }
    )

  environment = E.PursuitEvasionEnvironment(
    grid=torch.zeros((32, 32), device=device),
    evader_target_location=(30, 16),
    agents=[
      E.Agent('pursuer0', 'pursuer'),
      E.Agent('pursuer1', 'pursuer'),
      E.Agent('evader0', 'evader'),
      E.Agent('evader1', 'evader'),
    ],
    agent_extrinsics={
      'pursuer0': E.AgentExtrinsics(5, 7),
      'pursuer1': E.AgentExtrinsics(5, 25),
      'evader0': E.AgentExtrinsics(1, 7),
      'evader1': E.AgentExtrinsics(1, 25),
    },
  )
  
  d_model = 64
  policy = TransformerNetwork(d_model, patch_size=16, num_heads=8, num_layers=6, num_outputs=5)
  policy_ref = TransformerNetwork(d_model, patch_size=16, num_heads=8, num_layers=6, num_outputs=5)
  valuefunction = TransformerNetwork(d_model, patch_size=16, num_heads=8, num_layers=6, num_outputs=1)

  policy.to(device)
  policy_ref.to(device)
  valuefunction.to(device)

  optimizer = torch.optim.Adam([*policy.parameters(), *valuefunction.parameters()], lr=lr)

  # exponential decay from end epsilon to start epsilon over epsilon_decay episodes.
  # can think of this as a linear interpolation in logarithmic space.
  epsilon_ = lambda episode: np.exp(
    np.log(start_epsilon) * (1 - episode / epsilon_decay) + 
    np.log(end_epsilon) * (episode / epsilon_decay)
  )

  # vectorize environment
  environments = [copy.deepcopy(environment) for _ in range(n_batch_episodes)]

  #endregion

  try:
    for train_step in range(n_batches):
      epsilon = epsilon_(train_step) if train_step < epsilon_decay else 0

      episodes = [collect_episode(env, policy, device) for env in environments]

      log_reward_statistics(episodes)

      # Debugging
      # if (train_step + 1) % 100 == 0:
      #   visualize_episode(episodes[0])
      
      # Backpropagation
      vf_losses = []
      ppo_losses = []
      entropies = []

      policy_ref.load_state_dict(policy.state_dict())
      for ppo_iter in range(n_ppo_iterations):
        optimizer.zero_grad()

        discount_factor = 0.99

        for episode in episodes:
          episode.populate_discounted_rewards(discount_factor)

          vf_loss, vf_output = calculate_value_function_loss(valuefunction, episode)
          vf_loss.backward()
          vf_losses.append(vf_loss.item())

          # calculate policy loss for each agent one at a time
          for agent_i in range(len(episode.agents)):
            ppo_loss, entropy = calculate_policy_losses_for_agent(episode, agent_i, policy, policy_ref, vf_output)
            ppo_losses.append(ppo_loss.item())
            entropies.append(entropy.item())
            loss = ppo_loss - entropy * entropy_weight
            loss.backward()
            
            # end range(len(environment.agents))

          # end iteration over episodes

        optimizer.step()

        # end range(n_ppo_iterations)

      wandb.log({
        'policy_loss': sum(ppo_losses)/len(ppo_losses),
        'vf_loss': sum(vf_losses)/len(vf_losses),
        'entropy': sum(entropies)/len(entropies),
      })
  except Exception as e:
    traceback.print_exc()

    torch.save(policy.state_dict(), "policy.pt")
    torch.save(valuefunction.state_dict(), "qfunction.pt")

if __name__ == "__main__":
  main()

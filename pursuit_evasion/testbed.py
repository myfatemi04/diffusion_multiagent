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

import grid_world_environment as E
import numpy as np
import torch
import torch.nn.functional as F
import visualizer
import wandb
from marl import MultiAgentEpisode, MultiAgentSARSTuple
from matplotlib import pyplot as plt
from transformer import TransformerNetwork


def create_local_feature_graph(
    global_state: E.GlobalState,
    agent_fov: float,
    ego_agent_tag: str,
    visibility_radius: float):
  pass

def collect_episode(
  environment: E.PursuitEvasionEnvironment,
  policy: TransformerNetwork,
  epsilon: float,
  policy_agent_task_connectivity_radius: float,
  policy_agent_agent_connectivity_radius: float,
  qfunction_agent_task_connectivity_radius: float,
  qfunction_agent_agent_connectivity_radius: float,
  device: torch.device,
):
  print("episode")
  obs = environment.reset()

  steps: list[MultiAgentSARSTuple] = []

  is_artificial_episode = False # torch.rand(()).item() < 0.01

  # run for a max of 20 steps per episode
  for episode_step in range(40):
    # Simultaneously generate an action for all agents
    action_selection_per_agent: dict[str, int | None] = {}
    action_probs_per_agent: dict[str, torch.Tensor | None] = {}
    local_input_features_per_agent: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None] = {}

    # Policy rollout
    observability_matrix_agent_ids, observability_matrix = obs.observability_matrix
    for (agent_id, agent) in environment.agent_map.items():
      # Check to see if this is a caught or successful evader
      if agent_id in environment.caught_evaders or agent_id in environment.successful_evaders:
        action_selection_per_agent[agent_id] = None
        action_probs_per_agent[agent_id] = None
        local_input_features_per_agent[agent_id] = None
        assert obs.action_space[agent_id] is None, "Caught or successful evaders should not have action spaces"
        continue

      action_space = obs.action_space[agent_id]
      assert action_space is not None, "Active agents should have action spaces"

      my_index = observability_matrix_agent_ids.index(agent_id)
      visible_other_agents = [
        obs.state.agent_map[visible_agent_id]
        for i, visible_agent_id in enumerate(observability_matrix_agent_ids)
        if observability_matrix[my_index, i] and visible_agent_id != agent_id
      ]

      agent_locations = torch.tensor([obs.state.agent_positions[agent.id].tuple, *[obs.state.agent_positions[agent.id].tuple for agent in visible_other_agents]], device=device)
      agent_teams = torch.tensor([0, *[1 if agent.team == 'evader' else 0 for agent in visible_other_agents]], device=device)

      # apply policy forward method with batch size of 1 (surely parallelizable at some point in the future)
      with torch.no_grad():
        local_input_features = (
          agent_locations.unsqueeze(0),
          agent_teams.unsqueeze(0),
          torch.tensor(obs.state.evader_target_location, device=device).unsqueeze(0),
          obs.state.grid.unsqueeze(0),
        )
        # first 0 is for batch, second 0 is for agent token (token index 0 is "ego")
        logits = policy.forward(*local_input_features)[0, 0]
        # mask out invalid actions
        action_probs_per_agent[agent.id] = F.softmax(logits[action_space], dim=-1)
        action_selection_per_agent[agent.id] = action_space[int(torch.multinomial(action_probs_per_agent[agent.id], 1))] # type: ignore # for action_probs_per_agent[agent.id]
        local_input_features_per_agent[agent.id] = local_input_features
    # end agent loop
    # select active agent ids and put them in the global state
    agent_ids = [agent_id for agent_id in obs.state.agent_map.keys() if obs.action_space[agent_id] is not None]
    global_input_features = (
      agent_ids,
      torch.tensor([obs.state.agent_positions[agent_id].tuple for agent_id in agent_ids], device=device).unsqueeze(0),
      torch.tensor([0 if obs.state.agent_map[agent_id].team == 'pursuer' else 1 for agent_id in agent_ids], device=device).unsqueeze(0),
      torch.tensor(obs.state.evader_target_location, device=device).unsqueeze(0),
      obs.state.grid.unsqueeze(0),
    )
    next_obs = environment.step(action_selection_per_agent)
    steps.append(MultiAgentSARSTuple(
      obs.state,
      local_input_features_per_agent,
      global_input_features,
      action_selection_per_agent,
      obs.action_space,
      action_probs_per_agent,
      next_obs.reward,
      next_obs.done,
    ))
    obs = next_obs

    if next_obs.done:
      break

  return MultiAgentEpisode(list(environment.agent_map), steps)


def main():
  torch.random.manual_seed(0)
  np.random.seed(0)
  random.seed(0)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  print("Using device:", device)

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
      'pursuer0': E.AgentExtrinsics(5, 1),
      'pursuer1': E.AgentExtrinsics(5, 31),
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

  # Graph construction parameters
  qfunction_agent_task_connectivity_radius = 20
  qfunction_agent_agent_connectivity_radius = 20
  policy_agent_agent_connectivity_radius = 10
  policy_agent_task_connectivity_radius = 10

  # exponential decay from end epsilon to start epsilon over epsilon_decay episodes.
  # can think of this as a linear interpolation in logarithmic space.
  epsilon_ = lambda episode: np.exp(
    np.log(start_epsilon) * (1 - episode / epsilon_decay) + 
    np.log(end_epsilon) * (episode / epsilon_decay)
  )

  # vectorize environment
  environments = [copy.deepcopy(environment) for _ in range(n_batch_episodes)]

  try:
    for train_step in range(n_batches):
      epsilon = epsilon_(train_step) if train_step < epsilon_decay else 0

      print("train_step")

      episodes = [
        collect_episode(
          env, policy, epsilon,
          policy_agent_task_connectivity_radius,
          policy_agent_agent_connectivity_radius,
          qfunction_agent_task_connectivity_radius,
          qfunction_agent_agent_connectivity_radius,
          device,
        )
        for env in environments
      ]

      # Log reward statistics
      mean_episode_reward = sum(
        sum(sum(v for v in tup.reward.values() if v is not None) for tup in episode.steps)
        for episode in episodes
      ) / len(episodes)
      mean_episode_length = sum(len(episode.steps) for episode in episodes) / len(episodes)
      mean_reached_goal = sum(len(episode.steps[-1].global_state.successful_evaders) for episode in episodes) / len(episodes)
      mean_caught_evaders = sum(len(episode.steps[-1].global_state.caught_evaders) for episode in episodes) / len(episodes)

      # Backpropagation
      total_policy_loss = 0
      total_qfunction_loss = 0

      policy_ref.load_state_dict(policy.state_dict())

      plot_graph = False # (train_step % 100 == 0) and train_step > 0
      if plot_graph:
        for step in episodes[0].steps:
          print("reward:", step.reward)
          print("done:", step.done)
          print("probs:", step.action_probs)
          print("selection:", step.action_selection)
          plt.title("Step: " + str(train_step))
          visualizer.render_scene(
            {
              agent_id: {
                "xy": (step.global_state.agent_positions[agent_id].x, step.global_state.agent_positions[agent_id].y),
                "color": "red",
                "action_probs": step.action_probs[agent_id].tolist(), # type: ignore
                # "action_values": action_values_per_agent[agent].tolist(),
              }
              for agent_id in episodes[0].agents
              if step.action_space[agent_id] is not None
            },
            {
              f"target": {
                "xy": step.global_state.evader_target_location,
                "color": "blue"
              }
            },
          )

      num_loss_accumulations = 0

      for ppo_iter in range(n_ppo_iterations):
        optimizer.zero_grad()

        discount_factor = 0.99

        for episode in episodes:
          print("train_step: episode backprop")

          episode.populate_discounted_rewards(discount_factor)

          # calculate policy loss for each agent one at a time
          for agent_i in range(len(episode.agents)):

            agent_id = episode.agents[agent_i]
            selected_action_logprobs = []
            selected_action_logprobs_ref = []
            values = []
            entropies = []

            # Aggregate relevant information from episode
            for step_i, episode_step in enumerate(episode.steps):
              action_space = episode_step.action_space[agent_id]
              local_input_features = episode_step.local_input_features_per_agent[agent_id]
              action_selection = episode_step.action_selection[agent_id]

              # Check if the agent was activate at this time
              if action_space is None:
                assert (
                  local_input_features is None and \
                    action_selection is None and \
                    episode.steps[step_i].discounted_reward[agent_id] is None # type: ignore
                ), "Agent should have no input features, action selection, or discounted reward if it is not active. agent_id: " + agent_id + ", step_i: " + str(step_i) + ", action_selection: " + str(action_selection) + ", discounted_reward: " + str(episode.steps[step_i].discounted_reward[agent_id]) # type: ignore
                continue
              
              assert local_input_features is not None and action_selection is not None, "Agent should have input features and action selection if it is active"
              assert action_selection in action_space

              (active_agents, *global_input_features) = episode_step.global_input_features

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

              # enumerate(episode.steps)

            selected_action_logprobs = torch.stack(selected_action_logprobs)
            selected_action_logprobs_ref = torch.stack(selected_action_logprobs_ref).detach()
            values = torch.stack(values)
            entropies = torch.stack(entropies)
  
            discounted_rewards = torch.tensor([
              step.discounted_reward[agent_id] # type: ignore
              for step in episode.steps
              if step.discounted_reward[agent_id] is not None # type: ignore
            ], device=device)

            # PPO loss
            ratios = selected_action_logprobs - selected_action_logprobs_ref
            clipped_ratios = torch.clamp(ratios, -0.2, 0.2)
            advantage = discounted_rewards - values.detach()

            # print advantage per action
            if ppo_iter == 0 and plot_graph:
              print("### Agent:", agent_id)

              advantage_per_action = {}
              for step_i, step in enumerate(episode.steps):
                action_selection = step.action_selection[agent_id]
                advantage_per_action[action_selection] = advantage[step_i].item()
              
              print("advantage_per_action:", advantage_per_action)
              print("discounted rewards:", discounted_rewards)
              print("values:", values)

            # loss could be really really positive if, for example, ratios * advantage were really negative.
            actor_loss = -torch.min(ratios * advantage, clipped_ratios * advantage).mean()
            # critic_loss = F.smooth_l1_loss(values, discounted_rewards)
            # agent_loss = critic_loss + actor_loss - entropy_weight * entropies.mean()
            agent_policy_loss = actor_loss - entropy_weight * entropies.mean()
            agent_policy_loss.backward()

            total_policy_loss += actor_loss.item()
            num_loss_accumulations += 1

            # end range(len(environment.agents))

          valuefunction_per_step = [
            valuefunction(*episode_step.global_input_features[1:])[0, :, 0]
            for episode_step in episode.steps
          ]
          discounted_rewards_per_step = [
            torch.tensor([
              episode_step.discounted_reward[agent_id] # type: ignore
              for agent_id in episode_step.global_input_features[0]
            ], device=device)
            for episode_step in episode.steps
          ]
          vf_loss = sum(
            (F.smooth_l1_loss(vf, discounted_rewards)
            for vf, discounted_rewards in zip(valuefunction_per_step, discounted_rewards_per_step)),
            start=torch.tensor(0.0, device=device)
          ) / len(valuefunction_per_step)
          vf_loss.backward()

          # end iteration over episodes

        optimizer.step()

        # end range(n_ppo_iterations)

      # if num_loss_accumulations == 0:
      #   print(episodes)

      wandb.log({
        'epsilon': epsilon,
        'total_reward': mean_episode_reward,
        'episode_length': mean_episode_length,
        'loss': (total_policy_loss + total_qfunction_loss) / num_loss_accumulations,
        'policy_loss': total_policy_loss / num_loss_accumulations,
        'qfunction_loss': total_qfunction_loss / num_loss_accumulations,
        'reached_goal': mean_reached_goal,
        'caught_evaders': mean_caught_evaders,
      })
  except Exception as e:
    print(e)

    import traceback
    traceback.print_exc()

    torch.save(policy.state_dict(), "policy.pt")
    torch.save(valuefunction.state_dict(), "qfunction.pt")

if __name__ == "__main__":
  main()

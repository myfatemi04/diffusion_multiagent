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
import torch_geometric.data
import torch_geometric.nn as gnn
import visualizer
import wandb
from marl import MultiAgentEpisode, MultiAgentSARSTuple
from matplotlib import pyplot as plt
from sparse_graph_network import (SparseGraphNetwork,
                                  SparseGraphNetworkWithPositionalEmbedding)


def create_global_feature_graph(global_state: E.GlobalState, agent_action_selections: dict[str, int], agent_task_connectiity_radius: float, agent_agent_connectivity_radius: float):
  """
  generates a heterodata object. used for the q function. therefore, each agent's feature vector
  includes the action it took.
  """
  agents = global_state.agent_tags

  data = torch_geometric.data.HeteroData()
  data['agent'].x = torch.tensor([
    [
      global_state.agent_positions[agent].x,
      global_state.agent_positions[agent].y,
    ]
    for agent in agents
  ]).float()
  # create one-hot action embedding
  data['agent'].action = torch.tensor([
    [
      1 if agent_action_selections[agent] == i else 0
      for i in range(5)
    ]
    for agent in agents
  ]).float()
  data['task'].x = torch.tensor([
    [task.x, task.y]
    for task in global_state.tasks
  ]).float()

  # Construct edge lists that determine connectivity of graph
  # We add a dummy value so that when the tensor is created, it has the correct
  # number of dims, even if there is actually nothing in the edge_index.
  agent_agent_edge_index = [[-1, -1]]
  task_agent_edge_index = [[-1, -1]]
  for agent_i in range(len(agents)):
    for task_i in range(len(global_state.tasks)):
      agent = agents[agent_i]
      task = global_state.tasks[task_i]
      distance = np.linalg.norm([
        global_state.agent_positions[agent].x - task.x,
        global_state.agent_positions[agent].y - task.y
      ])
      if distance <= agent_task_connectiity_radius:
        task_agent_edge_index.append([task_i, agent_i])

    # All agents can see all others
    for agent_j in range(len(agents)):
      agent_a = agents[agent_i]
      agent_b = agents[agent_j]
      distance = np.linalg.norm([
        global_state.agent_positions[agent_a].x - global_state.agent_positions[agent_b].x,
        global_state.agent_positions[agent_a].y - global_state.agent_positions[agent_b].y
      ])
      if distance <= agent_agent_connectivity_radius:
        agent_agent_edge_index.append([agent_i, agent_j])
  
  data['task', 'visible_by', 'agent'].edge_index = torch.tensor(task_agent_edge_index).T[:, 1:]
  data['agent', 'visible_by', 'agent'].edge_index = torch.tensor(agent_agent_edge_index).T[:, 1:]
  data['task', 'visible_by', 'task'].edge_index = torch.tensor([[i, i] for i in range(len(global_state.tasks))]).T

  return data

def create_local_feature_graph(
    global_state: E.GlobalState,
    agent_fov: float,
    ego_agent_tag: str,
    # graph_construction_nearest_neighbors_k: int,
    graph_construction_radius: float):
  """
  generates a heterodata object
  """
  agents = list(global_state.agent_positions.keys())
  my_pos = global_state.agent_positions[ego_agent_tag]

  data = torch_geometric.data.HeteroData()

  visible_agent_tags = []
  agent_feature_vectors = []
  for agent_a in range(len(agents)):
    their_pos = global_state.agent_positions[agents[agent_a]]
    distance = np.linalg.norm([
      my_pos.x - their_pos.x,
      my_pos.y - their_pos.y
    ])
    if distance <= agent_fov:
      agent_feature_vectors.append([their_pos.x, their_pos.y])
      visible_agent_tags.append(agents[agent_a])

  visible_task_ids = []
  task_feature_vectors = []
  for visible_task_id in range(len(global_state.tasks)):
    task = global_state.tasks[visible_task_id]
    distance = np.linalg.norm([
      my_pos.x - task.x,
      my_pos.y - task.y
    ])
    # if distance <= agent_fov:
    # just assume that all tasks are visible
    task_feature_vectors.append([task.x, task.y])
    visible_task_ids.append(visible_task_id)
  
  data['agent'].x = torch.tensor(agent_feature_vectors).float()
  data['task'].x = torch.tensor(task_feature_vectors).float()

  agent_agent_edge_index = []
  task_agent_edge_index = []
  for agent_a in visible_agent_tags:
    # tasks should be globally visible static entities
    for visible_task_id in visible_task_ids:
      task_agent_edge_index.append([visible_task_ids.index(visible_task_id), visible_agent_tags.index(agent_a)])

    # agents should only be able to view each other if they are within a certain radius
    for agent_b in visible_agent_tags:
      distance = np.linalg.norm([
        global_state.agent_positions[agent_a].x - global_state.agent_positions[agent_b].x,
        global_state.agent_positions[agent_a].y - global_state.agent_positions[agent_b].y
      ])
      if distance <= graph_construction_radius:
        agent_agent_edge_index.append([visible_agent_tags.index(agent_a), visible_agent_tags.index(agent_b)])
  
  data['task', 'visible_by', 'agent'].edge_index = torch.tensor(task_agent_edge_index).T.long()
  data['agent', 'visible_by', 'agent'].edge_index = torch.tensor(agent_agent_edge_index).T.long()
  data['task', 'visible_by', 'task'].edge_index = torch.tensor([[i, i] for i in range(len(visible_task_ids))]).T.long()

  return data, visible_agent_tags

def collect_episode(
  environment: E.TaskSimulator,
  policy: SparseGraphNetworkWithPositionalEmbedding,
  epsilon: float,
  policy_agent_task_connectivity_radius: float,
  policy_agent_agent_connectivity_radius: float,
  qfunction_agent_task_connectivity_radius: float,
  qfunction_agent_agent_connectivity_radius: float
):
  obs = environment.reset()

  steps: list[MultiAgentSARSTuple] = []

  is_artificial_episode = False # torch.rand(()).item() < 0.01

  # run for a max of 20 steps per episode
  for episode_step in range(40):
    # Simultaneously generate an action for all agents
    action_selection_per_agent = {}
    action_probs_per_agent = {}
    # action_values_per_agent = {}
    local_graphs_per_agent = {}
    local_feature_visible_agents = {}

    # Policy rollout. No grad.
    with torch.no_grad():
      for agent in environment.agents:
        # This feature map represents this specific agent's field of view
        local_features, agent_order = create_local_feature_graph(obs.state, policy_agent_task_connectivity_radius, agent, policy_agent_agent_connectivity_radius)
        out = policy(local_features.x_dict, local_features.edge_index_dict)
        action_logit_vector = out['agent']

        local_feature_visible_agents[agent] = agent_order

        my_agent_index = agent_order.index(agent)
        # this is generated a few lines above
        action_space = obs.action_space[agent]

        # if torch.rand(()).item() < epsilon:
        #   selection_index = torch.randint(len(action_space), (1,))
        #   action_probability_vector = torch.ones(len(action_space)) / len(action_space)
        if is_artificial_episode:
          # Use gold demonstrations 50% of the time
          if agent == 'agent:1':
            action_logit_vector_dummy = torch.zeros_like(action_logit_vector[0])
            action_logit_vector_dummy[[0, 1, 3, 4]] = -100
            action_logit_vector_dummy[[2]] = 0
            action_probability_vector = F.softmax(action_logit_vector_dummy[action_space], dim=-1)
            selection_index = torch.multinomial(action_probability_vector, 1, False)
          elif agent == 'agent:0':
            action_logit_vector_dummy = torch.zeros_like(action_logit_vector[0])
            action_logit_vector_dummy[[0, 1, 2, 3]] = -100
            action_logit_vector_dummy[[4]] = 0
            action_probability_vector = F.softmax(action_logit_vector_dummy[action_space], dim=-1)
            selection_index = torch.multinomial(action_probability_vector, 1, False)
          # print(agent, action_space, action_space[selection_index])
        else:
          action_probability_vector = F.softmax(action_logit_vector[my_agent_index, action_space], dim=-1)
          # # give each action at least 1/(2 * num_actions) probability of being selected
          # num_actions = len(action_availability)
          # action_probability_vector = (action_probability_vector * (1 - 1/(2 * num_actions))) + (1/(2 * num_actions))
          selection_index = torch.multinomial(action_probability_vector, 1, False)

          # if episode > epsilon_decay:
          #   print(action_logit_vector, action_probability_vector)

        action_selection_per_agent[agent] = action_space[selection_index]
        local_graphs_per_agent[agent] = local_features
        action_probs_per_agent[agent] = torch.zeros(5)
        action_probs_per_agent[agent][action_space] = action_probability_vector

      # Simultaneously take action step
      global_graph = create_global_feature_graph(obs.state, action_selection_per_agent, qfunction_agent_task_connectivity_radius, qfunction_agent_agent_connectivity_radius)
      # next_global_graph = create_global_feature_graph(next_state, next_action_availability_per_agent, qfunction_agent_task_connectivity_radius, qfunction_agent_agent_connectivity_radius)
      next_obs = environment.step(action_selection_per_agent)
      steps.append(MultiAgentSARSTuple(
        next_obs.state,
        # next_state,
        local_graphs_per_agent,
        local_feature_visible_agents,
        global_graph,
        # next_global_graph,
        action_selection_per_agent,
        obs.action_space,
        action_probs_per_agent,
        obs.reward,
        next_obs.done,
        next_obs.total_completed_tasks,
      ))

      obs = next_obs

      if next_obs.done:
        break

  # if is_artificial_episode:
  #   print("Artificial episode reward:", sum(sum(tup.reward.values()) for tup in steps))
  #   print("Artificial episode num. reached goals:", steps[-1].num_completed_tasks)
  #   print(environment.tasks, environment.agents, environment.agent_extrinsics)

  return MultiAgentEpisode(environment.agents, steps)


def main():
  torch.random.manual_seed(0)
  np.random.seed(0)
  random.seed(0)

  # initial_lr = 1e-3
  # end_lr = 1e-5
  n_scenarios = 1
  n_agents = 1
  n_tasks = 2
  alg = 'v000-sparse-graph-network'
  n_batches = 1000000
  n_ppo_iterations = 1
  n_batch_episodes = 16
  lr = 1e-3

  # Make an epsilon value that decays exponentially from 0.5 to 0.005 over the first 10000 episodes, then goes to 0.
  start_epsilon = 0.5
  end_epsilon = 0.005
  epsilon_decay = 500

  entropy_weight = 5e-3

  layer_sizes = [64, 64, 64]

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
        "layer_sizes": layer_sizes,
        "conv_layer": "SAGEConv",
      }
    )

  environment = E.TaskSimulator(
    grid=np.zeros((20, 20)),
    tasks=[
      E.Task(x=5, y=5, reward=1),
      E.Task(x=15, y=15, reward=1),
    ],
    agents=['agent:0', 'agent:1'],
    agent_extrinsics={
      'agent:0': E.AgentExtrinsics(x=5, y=12),
      'agent:1': E.AgentExtrinsics(x=15, y=8)
    },
    randomize=False
  )
  
  dummy_global_state = E.GlobalState(
    ['dummy-agent'],
    [E.Task(1, 1, 1, False)],
    {'dummy-agent': E.AgentExtrinsics(1, 1)},
    2,
    2
  )
  dummy_global_features = create_global_feature_graph(
    dummy_global_state,
    {'dummy-agent': 0},
    10, 10,
  )
  dummy_local_features = create_local_feature_graph(
    dummy_global_state,
    agent_fov=10,
    ego_agent_tag='dummy-agent',
    graph_construction_radius=5,
  )[0]
  
  # GATConv = partial(gnn.GATConv, add_self_loops=False)

  # policy and q function will be separate for now
  policy = SparseGraphNetworkWithPositionalEmbedding(layer_sizes, head_dim=5, conv_layer=gnn.SAGEConv).make_heterogeneous(dummy_local_features)
  # policy_ref is for PPO.
  policy_ref = SparseGraphNetworkWithPositionalEmbedding(layer_sizes, head_dim=5, conv_layer=gnn.SAGEConv).make_heterogeneous(dummy_local_features)
  # SparseGraphQNetwork takes in agent actions as well.
  valuefunction = SparseGraphNetworkWithPositionalEmbedding(layer_sizes, head_dim=1, conv_layer=gnn.SAGEConv).make_heterogeneous(dummy_global_features)

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

      # Create state-action-reward streams for each agent.
      # use_thread_pool = True
      # tic = time.time()
      # if use_thread_pool:
      #   with ThreadPoolExecutor() as executor:
      #     episodes = list(executor.map(
      #       lambda env: collect_episode(env, policy, epsilon, policy_agent_task_connectivity_radius, policy_agent_agent_connectivity_radius, qfunction_agent_task_connectivity_radius, qfunction_agent_agent_connectivity_radius),
      #       environments
      #     ))
      # else:
      episodes = [
        collect_episode(env, policy, epsilon, policy_agent_task_connectivity_radius, policy_agent_agent_connectivity_radius, qfunction_agent_task_connectivity_radius, qfunction_agent_agent_connectivity_radius)
        for env in environments
      ]
      # toc = time.time()
      # print("Episode collection time:", (toc - tic) / n_batch_episodes, "seconds")

      # Log reward statistics
      mean_episode_reward = sum(
        sum(sum(tup.reward.values()) for tup in episode.steps)
        for episode in episodes
      ) / len(episodes)
      mean_episode_length = sum(len(episode.steps) for episode in episodes) / len(episodes)
      mean_completed_tasks = sum(episode.steps[-1].num_completed_tasks for episode in episodes) / len(episodes)

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
              agent: {
                "xy": (step.global_state.agent_positions[agent].x, step.global_state.agent_positions[agent].y),
                "color": "red",
                "action_probs": step.action_probs[agent].tolist(),
                # "action_values": action_values_per_agent[agent].tolist(),
              }
              for agent in episodes[0].agents
            },
            {
              f"task:{i}": {
                "xy": (task.x, task.y),
                "color": "blue"
              }
              for i, task in enumerate(step.global_state.tasks)
            },
          )

      for ppo_iter in range(n_ppo_iterations):
        optimizer.zero_grad()

        discount_factor = 0.99

        for episode in episodes:
          episode.populate_discounted_rewards(discount_factor)

          # calculate loss for each agent one at a time
          for agent_i in range(len(environment.agents)):
            agent = environment.agents[agent_i]
            selected_action_logprobs = []
            selected_action_logprobs_ref = []
            values = []
            entropies = []

            # Aggregate relevant information from episode
            for step_i, episode_step in enumerate(episode.steps):
              local_graph = episode_step.local_graph[agent]
              action_selection = episode_step.action_selection[agent]
              action_space = episode_step.action_availability[agent]
              global_graph = episode_step.global_graph

              assert action_selection in action_space

              # Forward pass: Get action logits and value.
              # We store a mapping between nodes in the local subgraph (which are numbered 0...n)
              # and the agent IDs that correspond to them.
              out = policy(local_graph.x_dict, local_graph.edge_index_dict)
              my_index_in_local_subgraph = episode_step.local_feature_visible_agents[agent].index(agent)
              logits = out['agent'][my_index_in_local_subgraph][action_space]

              with torch.no_grad():
                out_ref = policy_ref(local_graph.x_dict, local_graph.edge_index_dict)
                logits_ref = out_ref['agent'][my_index_in_local_subgraph][action_space]

              out = valuefunction(global_graph.x_dict, global_graph.edge_index_dict)
              value = out['agent'].squeeze(-1)[agent_i]

              # Store logprobs for executed policy
              logprobs = torch.log_softmax(logits, dim=-1)
              logprobs_ref = torch.log_softmax(logits_ref, dim=-1)
              action_selection_index = action_space.index(action_selection)
              selected_action_logprobs.append(logprobs[action_selection_index])
              selected_action_logprobs_ref.append(logprobs_ref[action_selection_index])

              # Store state values
              values.append(value)

              # Store entropy
              entropy = -torch.sum(logprobs * torch.exp(logprobs), dim=-1)
              entropies.append(entropy)

              # enumerate(episode.steps)

            selected_action_logprobs = torch.stack(selected_action_logprobs)
            selected_action_logprobs_ref = torch.stack(selected_action_logprobs_ref).detach()
            values = torch.stack(values)
            entropies = torch.stack(entropies)
  
            discounted_rewards = torch.tensor([
              step.discounted_reward[agent] # type: ignore
              for step in episode.steps
            ])

            # PPO loss
            ratios = selected_action_logprobs - selected_action_logprobs_ref
            clipped_ratios = torch.clamp(ratios, -0.2, 0.2)
            advantage = discounted_rewards - values.detach()

            # print advantage per action
            if ppo_iter == 0 and plot_graph:
              print("### Agent:", environment.agents[agent_i])

              advantage_per_action = {}
              for step_i, step in enumerate(episode.steps):
                action_selection = step.action_selection[agent]
                advantage_per_action[action_selection] = advantage[step_i].item()
              
              print("advantage_per_action:", advantage_per_action)
              print("discounted rewards:", discounted_rewards)
              print("values:", values)

            # loss could be really really positive if, for example, ratios * advantage were really negative.
            actor_loss = -torch.min(ratios * advantage, clipped_ratios * advantage).mean()
            critic_loss = F.smooth_l1_loss(values, discounted_rewards)
            agent_loss = critic_loss + actor_loss - entropy_weight * entropies.mean()
            agent_loss.backward()

            total_policy_loss += actor_loss.item()
            total_qfunction_loss += critic_loss.item()

            # end range(len(environment.agents))
          # end iteration over episodes

        optimizer.step()

        # end range(n_ppo_iterations)

      wandb.log({
        'epsilon': epsilon,
        'total_reward': mean_episode_reward,
        'episode_length': mean_episode_length,
        'loss': (total_policy_loss + total_qfunction_loss) / (len(environment.agents) * n_ppo_iterations * n_batch_episodes),
        'policy_loss': total_policy_loss / (len(environment.agents) * n_ppo_iterations * n_batch_episodes),
        'qfunction_loss': total_qfunction_loss / (len(environment.agents) * n_ppo_iterations * n_batch_episodes),
        'completed_tasks': mean_completed_tasks,
      })
  except Exception as e:
    print(e)

    import traceback
    traceback.print_exc()

    torch.save(policy.state_dict(), "policy.pt")
    torch.save(valuefunction.state_dict(), "qfunction.pt")

if __name__ == "__main__":
  main()

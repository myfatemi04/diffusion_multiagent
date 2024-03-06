"""
Approach:
 - Render the scene as high-dimensional feature map
   centered at the agent's current position
 - Patchify the feature map and use the patches as
   tokens in a transformer
 - Use the transformer to predict the best action to
   take with a specialized "action readout" head
"""

import grid_world_environment as E
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data
import torch_geometric.nn as gnn
from matplotlib import pyplot as plt
from sparse_graph_network import SparseGraphNetwork, SparseGraphNetworkWithPositionalEmbedding


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
      # *[
      #   1 if agent_action_selections[agent] == i else 0
      #   for i in range(4)
      # ]
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

def main():
  import wandb

  # initial_lr = 1e-3
  # end_lr = 1e-5
  n_scenarios = 1
  n_agents = 1
  n_tasks = 2
  alg = 'v000-sparse-graph-network'
  n_episodes = 1000000
  lr = 1e-3
  wandb.init(
    # set the wandb project where this run will be logged
    project="arl-collab-planning",
    # track hyperparameters and run metadata
    config={
      "lr_schedule": "constant",
      # "initial_lr": initial_lr,
      # "end_lr": end_lr,
      "lr": lr,
      "architecture": alg,
      "n_episodes": n_episodes,
      "n_scenarios": n_scenarios,
      "n_agents": n_agents,
      "n_tasks": n_tasks,
    }
  )

  environment = E.TaskSimulator(
    grid=np.zeros((20, 20)),
    tasks=[
      E.Task(x=5, y=5, reward=1),
      E.Task(x=15, y=15, reward=1),
    ]
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
  
  # policy and q function will be separate for now
  policy = SparseGraphNetworkWithPositionalEmbedding([64, 64, 64], head_dim=5).make_heterogeneous(dummy_local_features)
  # policy_ref is for PPO.
  policy_ref = SparseGraphNetworkWithPositionalEmbedding([64, 64, 64], head_dim=5).make_heterogeneous(dummy_local_features)
  valuefunction = SparseGraphNetworkWithPositionalEmbedding([64, 64, 64], head_dim=1).make_heterogeneous(dummy_global_features)

  optimizer = torch.optim.Adam([*policy.parameters(), *valuefunction.parameters()], lr=lr)

  # Graph construction parameters
  qfunction_agent_task_connectivity_radius = 20
  qfunction_agent_agent_connectivity_radius = 20
  policy_agent_agent_connectivity_radius = 10
  policy_agent_task_connectivity_radius = 10

  # run_optimization_step_every = 10

  # Make an epsilon value that decays exponentially from 0.5 to 0.005 over the first 10000 episodes, then goes to 0.
  start_epsilon = 0.5
  end_epsilon = 0.005
  epsilon_decay = 10000
  epsilon_ = lambda episode: end_epsilon + (start_epsilon - end_epsilon) * np.exp(-episode / epsilon_decay)

  try:
    for episode in range(n_episodes):
      state, action_availability_per_agent, reward_per_agent, done = environment.reset()

      epsilon = epsilon_(episode) if episode < epsilon_decay else 0

      # Create state-action-reward streams for each agent.
      SARS_tuples_per_agent = {agent: [] for agent in environment.agents}

      # run for a max of 20 steps per episode
      for step in range(20):
        # Simultaneously generate an action for all agents
        action_selection_per_agent = {}
        local_graphs_per_agent = {}

        # Policy rollout. No grad.
        with torch.no_grad():
          for agent in environment.agents:
            # This feature map represents this specific agent's field of view
            local_features, agent_order = create_local_feature_graph(state, policy_agent_task_connectivity_radius, agent, policy_agent_agent_connectivity_radius)
            out = policy(local_features.x_dict, local_features.edge_index_dict)
            action_logit_vector = out['agent']

            my_agent_index = agent_order.index(agent)
            # this is generated a few lines above
            action_availability = action_availability_per_agent[agent]

            if torch.rand(()).item() < epsilon:
              selection_index = torch.randint(len(action_availability), (1,))
            else:
              action_probability_vector = F.softmax(action_logit_vector[my_agent_index, action_availability], dim=-1)
              # # give each action at least 1/(2 * num_actions) probability of being selected
              # num_actions = len(action_availability)
              # action_probability_vector = (action_probability_vector * (1 - 1/(2 * num_actions))) + (1/(2 * num_actions))
              selection_index = torch.multinomial(action_probability_vector, 1, False)

            action_selection_per_agent[agent] = action_availability[selection_index]
            local_graphs_per_agent[agent] = local_features

          # Simultaneously take action step
          global_graph = create_global_feature_graph(state, action_selection_per_agent, qfunction_agent_task_connectivity_radius, qfunction_agent_agent_connectivity_radius)
          state, action_availability_per_agent, reward_per_agent, done = environment.step(action_selection_per_agent)

          for agent in environment.agents:
            SARS_tuples_per_agent[agent].append((
              local_graphs_per_agent[agent],
              global_graph,
              action_selection_per_agent[agent],
              reward_per_agent[agent],
            ))

        # plot agent locations every 100 episodes
        # if (episode + 1) % 100 == 0:
        #   plt.clf()
        #   agent_x = [state.agent_positions[agent].x for agent in state.agent_positions]
        #   agent_y = [state.agent_positions[agent].y for agent in state.agent_positions]
        #   plt.scatter(agent_x, agent_y, c='r', label='agents')
        #   task_x = [task.x for task in state.tasks]
        #   task_y = [task.y for task in state.tasks]
        #   plt.scatter(task_x, task_y, c='b', label='tasks')
        #   plt.legend()
        #   plt.pause(0.00001)

        if done:
          break

      
      # Log reward statistics
      total_reward = sum([
        sum([
          reward for (local_graph, global_graph, action_selection, reward) in SARS_tuples
        ])
        for SARS_tuples in SARS_tuples_per_agent.values()
      ])

      # Backpropagation
      total_policy_loss = 0
      total_qfunction_loss = 0

      policy_ref.load_state_dict(policy.state_dict())

      num_ppo_iterations = 5
      for batch_optimization_epoch in range(num_ppo_iterations):
        optimizer.zero_grad()

        # Accumulate gradients for each agent
        for agent_i, agent in enumerate(environment.agents):
          SARS_tuples = SARS_tuples_per_agent[agent]
          discount_factor = 0.99

          # Calculates discounted rewards for THIS AGENT at EACH TIME STEP
          discounted_rewards = [SARS_tuples[-1][3]]
          for step_i in range(len(SARS_tuples) - 2, -1, -1):
            reward_at_step_i = SARS_tuples[step_i][3]
            discounted_reward_at_step_i = reward_at_step_i + discount_factor * discounted_rewards[-1]
            discounted_rewards.append(discounted_reward_at_step_i)

          discounted_rewards = torch.tensor(discounted_rewards[::-1], dtype=torch.float32)

          # Calculate global q-values for each time step
          action_logprobs = []
          action_logprobs_ref = []
          action_values = []
          for step_i, (local_graph, global_graph, action_selection, _reward) in enumerate(SARS_tuples):
            out = policy(local_graph.x_dict, local_graph.edge_index_dict)
            action_logit_vector = out['agent'][agent_i]

            with torch.no_grad():
              out_ref = policy_ref(local_graph.x_dict, local_graph.edge_index_dict)
              action_logit_vector_ref = out_ref['agent'][agent_i]

            action_logprob_vector = torch.log_softmax(action_logit_vector, dim=-1)
            action_logprob_vector_ref = torch.log_softmax(action_logit_vector_ref, dim=-1)

            action_values_for_all_agents = valuefunction(global_graph.x_dict, global_graph.edge_index_dict)['agent'].squeeze(-1)
            action_values.append(action_values_for_all_agents[agent_i])
            action_logprobs.append(action_logprob_vector[action_selection])
            action_logprobs_ref.append(action_logprob_vector_ref[action_selection])

          action_logprobs = torch.stack(action_logprobs)
          action_logprobs_ref = torch.stack(action_logprobs_ref).detach()
          action_values = torch.stack(action_values)

          # PPO loss
          ratios = action_logprobs - action_logprobs_ref
          clipped_ratios = torch.clamp(ratios, -0.2, 0.2)
          advantage = discounted_rewards - action_values
          # loss could be really really positive if, for example, ratios * advantage were really high.
          actor_loss = -torch.min(ratios * advantage.detach(), clipped_ratios * advantage.detach()).mean()
          critic_loss = F.mse_loss(action_values, discounted_rewards)
          agent_loss = critic_loss + actor_loss
          agent_loss.backward()

          total_policy_loss += actor_loss.item()
          total_qfunction_loss += critic_loss.item()

        # Take optimization step after all agents have had gradient updates allowed
        optimizer.step()

      wandb.log({
        'epsilon': epsilon,
        'total_reward': total_reward,
        'total_loss': (total_policy_loss + total_qfunction_loss) / (len(environment.agents) * num_ppo_iterations),
        'policy_loss': total_policy_loss / (len(environment.agents) * num_ppo_iterations),
        'qfunction_loss': total_qfunction_loss / (len(environment.agents) * num_ppo_iterations),
      })
  except Exception as e:
    print(e)

    import traceback
    traceback.print_exc()

    torch.save(policy.state_dict(), "policy.pt")
    torch.save(valuefunction.state_dict(), "qfunction.pt")

if __name__ == "__main__":
  main()

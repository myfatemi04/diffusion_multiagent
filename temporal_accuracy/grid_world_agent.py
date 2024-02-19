"""
Approach:
 - Render the scene as high-dimensional feature map
   centered at the agent's current position
 - Patchify the feature map and use the patches as
   tokens in a transformer
 - Use the transformer to predict the best action to
   take with a specialized "action readout" head
"""

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import grid_world_environment as E

def generate_local_feature_map(agent_tag: str, global_state, field_of_view):
  """
  Renders a local feature map.
  Feature map dimensions:
  (1) Task here
  (2) Agent here
  (3) Obstacle here [or restricted area here]

  :params:
  - agent_location: `AgentExtrinsics`
  - global_state: `GlobalState`
  - field_of_view: int - The size of the field of view
  - size: int - Size of the feature map to generate
  """

  agent_location = global_state.agent_positions[agent_tag]
  feature_map = torch.zeros((
    2 * field_of_view + 1,
    2 * field_of_view + 1,
    3
  ))
  for task in global_state.tasks:
    if abs(task.x - agent_location.x) <= field_of_view and abs(task.y - agent_location.y) <= field_of_view:
      feature_map[
        task.y - agent_location.y + field_of_view,
        task.x - agent_location.x + field_of_view,
        0,
      ] = task.reward

  for agent in global_state.agent_positions:
    if abs(global_state.agent_positions[agent].x - agent_location.x) <= field_of_view and abs(global_state.agent_positions[agent].y - agent_location.y) <= field_of_view:
      feature_map[
        global_state.agent_positions[agent].y - agent_location.y + field_of_view,
        global_state.agent_positions[agent].x - agent_location.x + field_of_view,
        1,
      ] = 1

  # Calculate restricted area
  for y in range(agent_location.y - field_of_view, agent_location.y + field_of_view + 1):
    for x in range(agent_location.x - field_of_view, agent_location.x + field_of_view + 1):
      if not (0 <= y < global_state.height and 0 <= x < global_state.width):
        feature_map[
          y - agent_location.y + field_of_view,
          x - agent_location.x + field_of_view,
          2,
        ] = 1
      # elif global_state.grid[y, x] == 4:
      #   feature_map[
      #     y - agent_location.y + field_of_view,
      #     x - agent_location.x + field_of_view,
      #     2,
      #   ] = 1
  
  return feature_map

class Policy(nn.Module):
  def __init__(self, d_model: int, feature_map_dimensions: int, max_input_size: int, patch_size: int, num_actions: int):
    super().__init__()

    self.d_model = d_model
    self.feature_map_dimensions = feature_map_dimensions
    self.max_input_size = max_input_size
    self.patch_size = patch_size
    self.num_actions = num_actions
    # Converts raw feature map into model features
    self.patchify = nn.Conv2d(
      feature_map_dimensions,
      d_model,
      kernel_size=patch_size,
      stride=patch_size,
      padding=0
    )
    self.embeddings = nn.Embedding(int((max_input_size // patch_size) ** 2), d_model)
    self.action_embedding = nn.Parameter(torch.randn(d_model))
    self.transformer_encoder = nn.TransformerEncoder(
      nn.TransformerEncoderLayer(
        d_model,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        batch_first=True,
      ),
      num_layers=6,
    )
    self.action_head = nn.Linear(d_model, num_actions * 2)

  def forward(self, feature_map: torch.Tensor):
    """
    :params:
    - feature_map: torch.Tensor - The feature map to process

    :returns:
    - action_logits: torch.Tensor - The logits for each action
    - action_values: torch.Tensor - The value estimates for each action
    """
    # Patchify the feature map
    batch_size = feature_map.shape[0]
    feature_map = self.patchify(feature_map.permute((0, 3, 1, 2)))
    feature_map = feature_map.reshape(batch_size, -1, self.d_model)
    feature_map = feature_map + self.embeddings(
      torch.arange(feature_map.shape[1], device=feature_map.device)
    )
    action_embedding = self.action_embedding.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
    tokens = torch.cat([action_embedding, feature_map], dim=1)
    tokens = self.transformer_encoder(tokens)
    action_output = self.action_head(tokens[:, 0, :])
    action_logits = action_output[:, :self.num_actions]
    action_values = action_output[:, self.num_actions:]
    return (action_logits, action_values)

def main():
  environment = E.TaskSimulator(
    grid=np.zeros((20, 20)),
    tasks=[
      E.Task(x=5, y=5, reward=1),
      E.Task(x=15, y=15, reward=1),
    ]
  )
  policy = Policy(64, 3, 11, 1, num_actions=5)
  optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)


  for episode in range(100):
    state, action_availability_per_agent, reward_per_agent, done = environment.reset()

    # Create state-action-reward streams for each agent.
    SAR_tuples_per_agent = {agent: [] for agent in environment.agents}
    for step in range(20):
      # Simultaneously generate an action for all agents
      action_selection_per_agent = {}
      for agent in environment.agents:
        # This feature map represents this specific agent's field of view
        feature_map = generate_local_feature_map(agent, state, 5)

        # action_value_vector is not used during action selection, only to stabilize training
        action_logit_vector, _action_value_vector = policy(feature_map.unsqueeze(0))
        action_availability = action_availability_per_agent[agent]
        action_probability_vector = torch.softmax(action_logit_vector[0, action_availability], dim=-1)

        selection_index = torch.multinomial(action_probability_vector, 1, False)
        action_selection_per_agent[agent] = action_availability[selection_index]

      # Simultaneously take action step
      state, action_availability_per_agent, reward_per_agent, done = environment.step(action_selection_per_agent)

      for agent in environment.agents:
        SAR_tuples_per_agent[agent].append((
          feature_map,
          action_selection_per_agent[agent],
          reward_per_agent[agent]
        ))

      if done:
        break

      plt.clf()
      plt.title("Feature Map")
      plt.imshow(feature_map)
      plt.pause(0.01)

    # Backpropagation
    optimizer.zero_grad()

    total_loss = 0

    # Accumulate gradients for each agent
    for agent in environment.agents:
      SAR_tuples = SAR_tuples_per_agent[agent]
      discount_factor = 0.9
      discounted_rewards = [SAR_tuples[-1][2]]
      for i in range(len(SAR_tuples) - 2, -1, -1):
        reward_at_step_i = SAR_tuples[i][2]
        discounted_reward_at_step_i = reward_at_step_i + discount_factor * discounted_rewards[-1]
        discounted_rewards.append(discounted_reward_at_step_i)

      discounted_rewards = torch.tensor(discounted_rewards[::-1], dtype=torch.float32)

      # Calculate model output for each feature map
      feature_map_batch = torch.stack([t[0] for t in SAR_tuples])
      action_logit_vector, action_value_vector = policy(feature_map_batch)
      action_logprob_vector = torch.log_softmax(action_logit_vector, dim=-1)
      action_logprobs = action_logprob_vector[
        # Each timestep
        range(len(SAR_tuples)),
        # Selected action
        [t[1] for t in SAR_tuples]
      ]
      action_values = action_value_vector[
        # Each timestep
        range(len(SAR_tuples)),
        # Selected action
        [t[1] for t in SAR_tuples]
      ]
      # Actor-Critic loss
      advantage = discounted_rewards - action_values
      critic_loss = F.mse_loss(action_value_vector, discounted_rewards)
      actor_loss = torch.mean(-action_logprobs * advantage.detach())
      agent_loss = critic_loss + actor_loss
      agent_loss.backward()

      total_loss += agent_loss.item()

    # Take optimization step after all agents have had gradient updates allowed
    optimizer.step()
    print(total_loss / len(environment.agents), reward_per_agent)

  torch.save(policy.state_dict(), "policy.pt")

if __name__ == "__main__":
  main()

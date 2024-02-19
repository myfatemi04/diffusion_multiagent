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
    self.action_head = nn.Linear(d_model, num_actions)

  def forward(self, feature_map: torch.Tensor):
    """
    :params:
    - feature_map: torch.Tensor - The feature map to process
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
    action_logits = self.action_head(tokens[:, 0, :])
    return action_logits

environment = E.TaskSimulator(
  grid=np.zeros((10, 10)),
  tasks=[
    E.Task(x=5, y=5, reward=1),
  ]
)

state, available_actions, rewards, done = environment.reset()
state_action_reward_tuples = []

for step in range(100):
  feature_map = generate_local_feature_map('agent:0', state, 5)

  policy = Policy(64, 3, 11, 1, num_actions=5)
  action_logits = policy(feature_map.unsqueeze(0))

  available_actions = available_actions['agent:0']
  selection_index = torch.multinomial(torch.softmax(action_logits[0, available_actions], dim=-1), 1, False)
  action = available_actions[selection_index]

  state, available_actions, rewards, done = environment.step({"agent:0": action})

  state_action_reward_tuples.append((feature_map, action, rewards['agent:0']))

  if done:
    break

  plt.clf()
  plt.title("Feature Map")
  plt.imshow(feature_map)
  plt.pause(0.01)


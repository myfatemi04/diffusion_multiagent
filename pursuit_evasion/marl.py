from dataclasses import dataclass

import torch
import grid_world_environment as E

@dataclass
class MultiAgentSARSTuple:
  global_state: E.GlobalState
  # next_global_state: E.GlobalState
  # caches the input features for each agent that were used to calculate the policy at this timestep
  local_input_features_per_agent: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None]
  global_input_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
  # next_global_graph: torch_geometric.data.HeteroData
  action_selection: torch.Tensor
  action_space: dict[str, list[int]]
  action_probs: dict[str, torch.Tensor]
  active_mask: torch.Tensor
  reward: torch.Tensor
  done: bool

  discounted_reward: torch.Tensor
  has_discounted_reward: bool

@dataclass
class MultiAgentEpisode:
  agents: list[str]
  steps: list[MultiAgentSARSTuple]

  def populate_discounted_rewards(self, gamma: float):
    self.has_discounted_reward = True
    for step_i in reversed(range(len(self.steps))):
      self.steps[step_i].discounted_reward = self.steps[step_i].reward + ((gamma * self.steps[step_i + 1].discounted_reward) if step_i + 1 < len(self.steps) else 0)

from dataclasses import dataclass

import torch
import torch_geometric.data
import grid_world_environment as E

@dataclass
class MultiAgentSARSTuple:
  global_state: E.GlobalState
  # next_global_state: E.GlobalState
  # caches the input features for each agent that were used to calculate the policy at this timestep
  local_input_features_per_agent: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None]
  global_input_features: tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
  # next_global_graph: torch_geometric.data.HeteroData
  action_selection: dict[str, int | None]
  action_space: dict[str, list[int] | None]
  action_probs: dict[str, torch.Tensor | None]
  reward: dict[str, float]
  done: bool

  discounted_reward: dict[str, float] | None = None

@dataclass
class MultiAgentEpisode:
  agents: list[str]
  steps: list[MultiAgentSARSTuple]

  def populate_discounted_rewards(self, gamma: float):
    for agent in self.agents:
      discounted_reward = 0
      for step in reversed(self.steps):
        # reward is None after the agent has been inactivated
        reward = step.reward[agent]
        if reward is None:
          continue
        # add to discounted reward
        discounted_reward = reward + gamma * discounted_reward
        if step.discounted_reward is None:
          step.discounted_reward = {}
        step.discounted_reward[agent] = discounted_reward

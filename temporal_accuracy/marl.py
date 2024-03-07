from dataclasses import dataclass

import torch
import torch_geometric.data
import grid_world_environment as E

@dataclass
class MultiAgentSARSTuple:
  global_state: E.GlobalState
  # next_global_state: E.GlobalState
  local_graph: dict[str, torch_geometric.data.HeteroData]
  global_graph: torch_geometric.data.HeteroData
  # next_global_graph: torch_geometric.data.HeteroData
  action_selection: dict[str, int]
  action_availability: dict[str, list[int]]
  action_probs: dict[str, torch.Tensor]
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
        reward = step.reward.get(agent, 0)
        discounted_reward = reward + gamma * discounted_reward
        if step.discounted_reward is None:
          step.discounted_reward = {}
        step.discounted_reward[agent] = discounted_reward

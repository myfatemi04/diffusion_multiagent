from dataclasses import dataclass
import torch_geometric.data

@dataclass
class MultiAgentSARSTuple:
  local_graph: dict[str, torch_geometric.data.HeteroData]
  global_graph: torch_geometric.data.HeteroData
  action_selection: dict[str, int]
  action_availability: dict[str, list[int]]
  reward: dict[str, float]
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

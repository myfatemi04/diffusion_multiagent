import copy
from dataclasses import dataclass

import numpy as np

"""
::: Multi-Agent Simulator Paradigm :::

There is a clock which will `tick` the environment at a set rate.
During each `tick`, each agent selects an action.

The environment will share full state information at each step. It is
the responsibility of the caller to ensure that decentralized execution
is valid. I provide methods that mask out tiles that are out of view
for each agent.

The caller takes actions on behalf of all the agents, and sends the joint
action vector into the `step` function.

Each agent has `intrinsic` and `extrinsic` information. Extrinsic information,
like where the agent is in the environment, is managed by the simulator.
Intrinsic information, like which team the agent is on, is encapsulated in an
`Agent` class. This `Agent` class will be the class that manages partial observability.

Each agent has a globally-unique identifier called a `tag`. This is used for taking
actions.
"""

@dataclass
class Agent:
    tag: str

@dataclass
class AgentExtrinsics:
    x: int
    y: int

@dataclass
class Task:
    x: int
    y: int
    reward: float
    completed: bool = False

@dataclass
class GlobalState:
    agent_tags: list[str]
    tasks: list[Task]
    agent_positions: dict[str, AgentExtrinsics]
    width: int
    height: int

@dataclass
class Observation:
    state: GlobalState
    action_space: dict[str, list[int]]
    reward: dict[str, float]
    total_completed_tasks: int
    done: bool = False

class TaskSimulator:
    def __init__(self, grid: np.ndarray, tasks: list[Task], agents: list[str], agent_extrinsics: dict[str, AgentExtrinsics]):
        self._original_tasks = copy.deepcopy(tasks)
        self._original_agent_extrinsics = copy.deepcopy(agent_extrinsics)
        self.tasks = tasks
        self.agents = agents
        self.agent_extrinsics = agent_extrinsics
        self.grid = grid

    def reset(self) -> Observation:
        self.tasks = copy.deepcopy(self._original_tasks)
        # self.agent_extrinsics['agent:0'].x = np.random.randint(0, self.width)
        # self.agent_extrinsics['agent:0'].y = np.random.randint(0, self.height)
        self.agent_extrinsics = copy.deepcopy(self._original_agent_extrinsics)

        # Return the state information, the set of valid actions, and the reward vector.
        num_incomplete_tasks = sum(1 for task in self.tasks if not task.completed)
        
        state = self.state
        action_space = self.valid_actions()
        reward = {agent: 0.0 for agent in self.agents}
        done = num_incomplete_tasks == 0

        return Observation(
            state=state,
            action_space=action_space,
            reward=reward,
            total_completed_tasks=len(self.tasks) - num_incomplete_tasks,
            done=done
        )

    @property
    def width(self):
        return self.grid.shape[1]
    
    @property
    def height(self):
        return self.grid.shape[0]

    """ Determine the action space for each agent. """
    def valid_actions(self) -> dict[str, list[int]]:
        actions = {}
        for agent in self.agents:
            pos = self.agent_extrinsics[agent]
            actions[agent] = []
            if pos.x < self.width - 1:
                actions[agent].append(1)
            if pos.y < self.height - 1:
                actions[agent].append(2)
            if pos.x > 0:
                actions[agent].append(3)
            if pos.y > 0:
                actions[agent].append(4)
        return actions
    
    @property
    def state(self):
        return GlobalState(
            agent_tags=self.agents[:],
            tasks=copy.deepcopy(self.tasks),
            agent_positions=copy.deepcopy(self.agent_extrinsics),
            width=self.width,
            height=self.height
        )
    
    def step(self, action) -> Observation:
        assert len(action) == len(self.agents), "Action vector must have the same length as the number of agents."

        rewards = {}

        for i, agent_tag in enumerate(self.agents):
            rewards[agent_tag] = 0

            pos = self.agent_extrinsics[agent_tag]
            if action[agent_tag] == 0:
                continue
            
            if action[agent_tag] == 1:
                pos.x += 1
            if action[agent_tag] == 2:
                pos.y += 1
            if action[agent_tag] == 3:
                pos.x -= 1
            if action[agent_tag] == 4:
                pos.y -= 1

            # Slightly penalize movements
            rewards[agent_tag] = -0.05
            
            assert (0 <= pos.x < self.width and 0 <= pos.y < self.height), f"Agent {agent_tag} tried to move out of bounds."

            for task in self.tasks:
                if task.completed:
                    continue

                if task.x == pos.x and task.y == pos.y:
                    task.completed = True
                    rewards[agent_tag] += task.reward

        # Return the state information, the set of valid actions, and the reward vector.
        num_incomplete_tasks = sum(1 for task in self.tasks if not task.completed)
        
        state = self.state
        action_space = self.valid_actions()
        done = num_incomplete_tasks == 0

        return Observation(
            state=state,
            action_space=action_space,
            reward=rewards,
            total_completed_tasks=len(self.tasks) - num_incomplete_tasks,
            done=done
        )

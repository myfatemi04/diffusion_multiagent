import copy
from dataclasses import dataclass

import numpy as np
import torch

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
    id: str
    # 'pursuer' or 'evader'
    team: str

@dataclass
class AgentExtrinsics:
    x: int
    y: int

    @property
    def tuple(self):
        return (self.x, self.y)

# @dataclass
# class Task:
#     x: int
#     y: int
#     reward: float
#     completed: bool = False

@dataclass
class GlobalState:
    grid: torch.Tensor
    agent_map: dict[str, Agent]
    pursuers: list[Agent]
    evaders: list[Agent]
    agent_positions: dict[str, AgentExtrinsics]
    width: int
    height: int
    successful_evaders: set[str]
    caught_evaders: set[str]
    evader_target_location: tuple[int, int]

@dataclass
class Observation:
    state: GlobalState
    action_space: dict[str, list[int] | None] # None represents that the agent is inactive
    observability_matrix: tuple[list[str], torch.Tensor]
    reward: dict[str, float]
    done: bool = False

"""
An environment with a set of agents, two teams, and a grid.

Assume we want as many agents as possible to reach location X.

"""
class PursuitEvasionEnvironment:
    def __init__(self,
                 grid: torch.Tensor,
                 agents: list[Agent],
                 agent_extrinsics: dict[str, AgentExtrinsics],
                 evader_target_location: tuple[int, int]):

        self._original_agent_extrinsics = copy.deepcopy(agent_extrinsics)
        self.agent_map = {agent.id: agent for agent in agents}
        self.pursuers = [agent for agent in agents if agent.team == 'pursuer']
        self.evaders = [agent for agent in agents if agent.team == 'evader']
        self.agent_extrinsics = agent_extrinsics
        self.successful_evaders = set()
        self.caught_evaders = set()
        self.remaining_evaders = {evader.id for evader in self.evaders}
        self.evader_target_location = evader_target_location
        self.grid = grid # 1 if forest, 0 if clear
        self.step_counter = 0

        # Store historical observations per agent
        # (i.e. pursuer 1 has observed agent X at location Y at time t - 5)
        self.observations_by_agent = {}

        # the chance of viewing decays by 1/2 every 10 steps
        # if you are only 5 steps away, you are guaranteed to see the other agent
        # make the probability of observation decay linearly
        self.observation_always_visible_distance = 5
        self.observation_never_visible_distance = 45

    def get_observation_likelihood(self, observer_location: AgentExtrinsics, observed_location: AgentExtrinsics) -> float:
        distance = np.sqrt((observer_location.x - observed_location.x) ** 2 + (observer_location.y - observed_location.y) ** 2)

        observer_in_forest = self.grid[observer_location.y, observer_location.x] == 1
        observed_in_forest = self.grid[observed_location.y, observed_location.x] == 1

        coefficient = 1.0
        if observer_in_forest:
            if observed_in_forest:
                coefficient = 0.8
            else:
                coefficient = 1.0
        else:
            if observed_in_forest:
                coefficient = 0.5
            else:
                coefficient = 1.0

        if distance <= self.observation_always_visible_distance:
            base_value = 1.0
        elif distance >= self.observation_never_visible_distance:
            base_value = 0.0
        else:
            # distance = near(prob) + far(1 - prob) = far - (far - near) * prob
            # prob = (far - distance) / (far - near)
            base_value = (self.observation_always_visible_distance - distance) / (self.observation_always_visible_distance - self.observation_never_visible_distance)
        
        return base_value * coefficient

    def sample_observability_matrix(self) -> tuple[list[str], torch.Tensor]:
        agent_order = [agent.id for agent in [*self.pursuers, *self.evaders]]
        observability_matrix = torch.zeros((len(agent_order), len(agent_order)), dtype=torch.bool)

        # go through each pair of pursuers and evaders
        # make a stochastic decision about whether the pursuer can see the evader
        for i, observer in enumerate(agent_order):
            for j, observed in enumerate(agent_order):
                if observer == observed:
                    observability_matrix[i, j] = True
                    continue

                observer_pos = self.agent_extrinsics[observer]
                observed_pos = self.agent_extrinsics[observed]
                likelihood = self.get_observation_likelihood(observer_pos, observed_pos)
                observability_matrix[i, j] = bool(np.random.rand() < likelihood)
        
        return agent_order, observability_matrix

    def reset(self) -> Observation:
        self.agent_extrinsics = copy.deepcopy(self._original_agent_extrinsics)

        self.step_counter = 0
        self.remaining_evaders = {evader.id for evader in self.evaders}
        self.successful_evaders = set()
        self.caught_evaders = set()
        state = self.get_state_copy()
        action_space = self.get_action_space()
        reward = {agent.id: 0.0 for agent in [*self.pursuers, *self.evaders]}
        done = len(self.caught_evaders) + len(self.successful_evaders) == len(self.evaders)

        return Observation(
            state=state,
            action_space=action_space,
            observability_matrix=self.sample_observability_matrix(),
            reward=reward,
            done=done
        )

    @property
    def width(self):
        return self.grid.shape[1]
    
    @property
    def height(self):
        return self.grid.shape[0]

    """ Determine the action space for each agent. """
    def get_action_space(self) -> dict[str, list[int] | None]:
        actions = {}

        for agent in self.pursuers:
            agent_id = agent.id
            pos = self.agent_extrinsics[agent_id]
            actions[agent_id] = [0]
            if pos.x < self.width - 1:
                actions[agent_id].append(1)
            if pos.y < self.height - 1:
                actions[agent_id].append(2)
            if pos.x > 0:
                actions[agent_id].append(3)
            if pos.y > 0:
                actions[agent_id].append(4)

        for agent in self.evaders:
            agent_id = agent.id
            if agent_id in self.caught_evaders or agent_id in self.successful_evaders:
                actions[agent_id] = None
                continue

            pos = self.agent_extrinsics[agent_id]
            actions[agent_id] = [0]
            if pos.x < self.width - 1:
                actions[agent_id].append(1)
            if pos.y < self.height - 1:
                actions[agent_id].append(2)
            if pos.x > 0:
                actions[agent_id].append(3)
            if pos.y > 0:
                actions[agent_id].append(4)
        
        return actions
    
    def get_state_copy(self):
        return GlobalState(
            agent_map=self.agent_map,
            grid=self.grid,
            pursuers=self.pursuers,
            evaders=self.evaders,
            agent_positions=copy.deepcopy(self.agent_extrinsics),
            width=self.width,
            height=self.height,
            caught_evaders={*self.caught_evaders},
            successful_evaders={*self.successful_evaders},
            evader_target_location=self.evader_target_location
        )
    
    def step(self, action: dict[str, int | None]) -> Observation:
        assert len(action) == len(self.pursuers) + len(self.evaders), "Action vector must have the same length as the number of agents. If an agent has been inactivated, set `None` as their action."

        rewards: dict[str, float] = {aid: 0.0 for aid in action.keys()}

        for agent in self.evaders:
            if agent.id in self.caught_evaders or agent.id in self.successful_evaders:
                rewards[agent.id] = None # type: ignore

        original_pursuer_positions = {agent.id: self.agent_extrinsics[agent.id].tuple for agent in self.pursuers}
        original_evader_positions = {agent.id: self.agent_extrinsics[agent.id].tuple for agent in self.evaders}
        target_pursuer_positions = {}
        target_evader_positions = {}
        target_pos_to_pursuer: dict[tuple[int, int], list] = {}
        target_pos_to_evader: dict[tuple[int, int], list] = {}

        # Iterate through pursuers. Pursuers move first.
        for i, agent in enumerate(self.pursuers):
            aid = agent.id
            pos = self.agent_extrinsics[aid]
            if action[aid] == 1:
                pos.x += 1
            if action[aid] == 2:
                pos.y += 1
            if action[aid] == 3:
                pos.x -= 1
            if action[aid] == 4:
                pos.y -= 1

            assert (0 <= pos.x < self.width and 0 <= pos.y < self.height), f"Agent {agent} tried to move out of bounds."

            target_pursuer_positions[aid] = pos.tuple
            if pos.tuple not in target_pos_to_pursuer:
                target_pos_to_pursuer[pos.tuple] = [aid]
            else:
                target_pos_to_pursuer[pos.tuple].append(aid)

        # Iterate through evaders.
        for i, agent in enumerate(self.evaders):
            aid = agent.id

            if aid in self.caught_evaders or aid in self.successful_evaders:
                assert action[aid] is None, "Evader action must be None if they are caught or successful."

            pos = self.agent_extrinsics[aid]
            if action[aid] == 1:
                pos.x += 1
            if action[aid] == 2:
                pos.y += 1
            if action[aid] == 3:
                pos.x -= 1
            if action[aid] == 4:
                pos.y -= 1

            assert (0 <= pos.x < self.width and 0 <= pos.y < self.height), f"Agent {agent} tried to move out of bounds."

            target_evader_positions[aid] = pos.tuple
            if pos.tuple not in target_pos_to_evader:
                target_pos_to_evader[pos.tuple] = [aid]
            else:
                target_pos_to_evader[pos.tuple].append(aid)
        
        # To deal with collisions:
        # Apply a -1 penalty for agents that collide into each other, and do not move either of them.
        # Check for collisions within the same team.
        for pos, agents in target_pos_to_pursuer.items():
            if len(agents) > 1:
                for agent in agents:
                    rewards[agent] -= 0.1
                    target_pursuer_positions[agent] = original_pursuer_positions[agent]
        
        for pos, agents in target_pos_to_evader.items():
            if len(agents) > 1:
                for agent in agents:
                    rewards[agent] -= 0.1
                    target_evader_positions[agent] = original_evader_positions[agent]

        for evader in self.evaders:
            if evader.id in self.caught_evaders or evader.id in self.successful_evaders:
                continue
            
            # Check to see if they reached the goal location.
            if target_evader_positions[evader.id] == self.evader_target_location:
                self.successful_evaders.add(evader.id)
                self.remaining_evaders.remove(evader.id)
                rewards[evader.id] += 1.0

                # Give a penalty to all pursuers. We don't actually know who was responsible for catching them, though.
                # Maybe I can check out QMIX to try to resolve this problem.
                for pursuer in self.pursuers:
                    rewards[pursuer.id] -= 1.0

                # If they reach the goal in the same turn as being "caught" by the pursuer,
                # consider it as if they were never caught.
                continue
            
            # Check to see if they were caught by a pursuer.
            for pursuer in self.pursuers:
                if target_pursuer_positions[pursuer.id] == target_evader_positions[evader.id]:
                    self.caught_evaders.add(evader.id)
                    self.remaining_evaders.remove(evader.id)
                    rewards[pursuer.id] += 1.0
                    rewards[evader.id] -= 1.0
                    break

        done = len(self.remaining_evaders) == 0

        return Observation(
            state=self.get_state_copy(),
            action_space=self.get_action_space(),
            reward=rewards,
            observability_matrix=self.sample_observability_matrix(),
            done=done
        )

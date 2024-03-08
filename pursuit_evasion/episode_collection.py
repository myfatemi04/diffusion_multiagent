import torch
import torch.nn.functional as F

import grid_world_environment as E
from marl import MultiAgentEpisode, MultiAgentSARSTuple
from transformer import TransformerNetwork

def collect_episode(
  environment: E.PursuitEvasionEnvironment,
  policy: TransformerNetwork,
  device: torch.device,
):
  obs = environment.reset()

  steps: list[MultiAgentSARSTuple] = []

  # run for a max of 20 steps per episode
  for episode_step in range(40):
    # Simultaneously generate an action for all agents
    action_selection_per_agent: dict[str, int | None] = {}
    action_probs_per_agent: dict[str, torch.Tensor | None] = {}
    local_input_features_per_agent: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None] = {}

    # Policy rollout
    observability_matrix_agent_ids, observability_matrix = obs.observability_matrix
    for (agent_id, agent) in environment.agent_map.items():
      # Check to see if this is a caught or successful evader
      if agent_id in environment.caught_evaders or agent_id in environment.successful_evaders:
        action_selection_per_agent[agent_id] = None
        action_probs_per_agent[agent_id] = None
        local_input_features_per_agent[agent_id] = None
        assert obs.action_space[agent_id] is None, "Caught or successful evaders should not have action spaces"
        continue

      action_space = obs.action_space[agent_id]
      assert action_space is not None, "Active agents should have action spaces"

      my_index = observability_matrix_agent_ids.index(agent_id)
      visible_other_agents = [
        obs.state.agent_map[visible_agent_id]
        for i, visible_agent_id in enumerate(observability_matrix_agent_ids)
        if observability_matrix[my_index, i] and visible_agent_id != agent_id
      ]

      agent_locations = torch.tensor([obs.state.agent_positions[agent.id].tuple, *[obs.state.agent_positions[agent.id].tuple for agent in visible_other_agents]], device=device)
      agent_teams = torch.tensor([0, *[1 if agent.team == 'evader' else 0 for agent in visible_other_agents]], device=device)

      # apply policy forward method with batch size of 1 (surely parallelizable at some point in the future)
      with torch.no_grad():
        local_input_features = (
          agent_locations.unsqueeze(0),
          agent_teams.unsqueeze(0),
          torch.tensor(obs.state.evader_target_location, device=device).unsqueeze(0),
          obs.state.grid.unsqueeze(0),
        )
        # first 0 is for batch, second 0 is for agent token (token index 0 is "ego")
        logits = policy.forward(*local_input_features)[0, 0]
        # mask out invalid actions
        action_probs_per_agent[agent.id] = F.softmax(logits[action_space], dim=-1)
        action_selection_per_agent[agent.id] = action_space[int(torch.multinomial(action_probs_per_agent[agent.id], 1))] # type: ignore # for action_probs_per_agent[agent.id]
        local_input_features_per_agent[agent.id] = local_input_features
    # end agent loop
    # select active agent ids and put them in the global state
    agent_ids = [agent_id for agent_id in obs.state.agent_map.keys() if obs.action_space[agent_id] is not None]
    global_input_features = (
      agent_ids,
      torch.tensor([obs.state.agent_positions[agent_id].tuple for agent_id in agent_ids], device=device).unsqueeze(0),
      torch.tensor([0 if obs.state.agent_map[agent_id].team == 'pursuer' else 1 for agent_id in agent_ids], device=device).unsqueeze(0),
      torch.tensor(obs.state.evader_target_location, device=device).unsqueeze(0),
      obs.state.grid.unsqueeze(0),
    )
    next_obs = environment.step(action_selection_per_agent)
    steps.append(MultiAgentSARSTuple(
      obs.state,
      local_input_features_per_agent,
      global_input_features,
      action_selection_per_agent,
      obs.action_space,
      action_probs_per_agent,
      next_obs.reward,
      next_obs.done,
    ))
    obs = next_obs

    if next_obs.done:
      break

  return MultiAgentEpisode(list(environment.agent_map), steps)

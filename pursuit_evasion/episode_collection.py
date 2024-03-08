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

  for episode_step in range(40):
    # Create feature vectors for each agent
    active_agent_ids = [agent_id for agent_id in obs.state.agent_map.keys() if obs.action_space[agent_id] is not None]
    agent_positions_vector = torch.tensor([obs.state.agent_positions[agent_id].tuple for agent_id in active_agent_ids], device=device)
    agent_teams_vector = torch.tensor([0 if obs.state.agent_map[agent_id].team == 'pursuer' else 1 for agent_id in active_agent_ids], device=device)

    # Simultaneously generate an action for all agents
    active_agent_mask = torch.tensor([0 if (agent_id in environment.caught_evaders or agent_id in environment.successful_evaders) else 1], device=device)
    action_selection_per_agent = torch.zeros(len(obs.state.agent_order), device=device)

    action_probs_per_agent: dict[str, torch.Tensor | None] = {}
    local_input_features_per_agent: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None] = {}

    evader_target_location_tensor = torch.tensor(obs.state.evader_target_location, device=device)

    # Policy rollout
    observability_matrix = obs.observability_matrix
    for (agent_id, agent) in environment.agent_map.items():
      # Check to see if this is a caught or successful evader
      if agent_id in environment.caught_evaders or agent_id in environment.successful_evaders:
        assert obs.action_space[agent_id] is None, "Caught or successful evaders should not have action spaces"
        continue

      action_space = obs.action_space[agent_id]
      assert action_space is not None, "Active agents should have action spaces"

      my_index = obs.state.agent_order.index(agent_id)

      visible_agent_indexes = observability_matrix[my_index].nonzero(as_tuple=True)[0]
      # move self index to front
      visible_agent_indexes = torch.cat([torch.tensor([my_index], device=device), visible_agent_indexes[visible_agent_indexes != my_index]], dim=0)

      # apply policy forward method with batch size of 1 (surely parallelizable at some point in the future)
      with torch.no_grad():
        local_input_features = (
          agent_positions_vector[visible_agent_indexes].unsqueeze(0),
          agent_teams_vector[visible_agent_indexes].unsqueeze(0),
          evader_target_location_tensor.unsqueeze(0),
          obs.state.grid.unsqueeze(0),
        )
        # first 0 is for batch, second 0 is for agent token (token index 0 is "ego")
        logits = policy.forward(*local_input_features)[0, 0]
        # mask out invalid actions
        action_probs_per_agent[agent.id] = F.softmax(logits[action_space], dim=-1)
        action_selection_per_agent[my_index] = action_space[int(torch.multinomial(action_probs_per_agent[agent.id], 1))] # type: ignore # for action_probs_per_agent[agent.id]
        local_input_features_per_agent[agent.id] = local_input_features
    # end agent loop
    # select active agent ids and put them in the global state
    global_input_features = (
      active_agent_ids,
      agent_positions_vector.unsqueeze(0),
      agent_teams_vector.unsqueeze(0),
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
      obs.state.active_mask,
      next_obs.reward,
      next_obs.done,
      discounted_reward=torch.zeros_like(next_obs.reward),
      has_discounted_reward=Falsel,
    ))
    obs = next_obs

    if next_obs.done:
      break

  return MultiAgentEpisode(list(environment.agent_map), steps)

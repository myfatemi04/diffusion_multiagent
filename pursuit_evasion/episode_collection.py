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
    # active_agent_ids = [agent_id for i, agent_id in enumerate(obs.state.agent_order) if obs.state.active_mask[i]]
    # We'll store the positions and teams of all agents, but only care about the agents where active_mask is on.
    agent_positions_vector = torch.tensor([obs.state.agent_positions[agent_id].tuple for agent_id in obs.state.agent_order], device=device)
    agent_teams_vector = torch.tensor([0 if obs.state.agent_map[agent_id].team == 'pursuer' else 1 for agent_id in obs.state.agent_order], device=device)

    # Simultaneously generate an action for all agents
    action_selection_per_agent = torch.zeros(len(obs.state.agent_order), device=device)

    action_probs_per_agent: dict[str, torch.Tensor] = {}
    local_input_features_per_agent: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None] = {}

    evader_target_location_tensor = torch.tensor(obs.state.evader_target_location, device=device)

    # Policy rollout
    observability_matrix = obs.observability_matrix
    for (agent_i, agent_id) in enumerate(obs.state.agent_order):
      agent = obs.state.agent_map[agent_id]

      # Check to see if this is a caught or successful evader
      if not obs.state.active_mask[agent_i]:
        continue

      action_space = obs.action_space[agent_id]
      assert action_space is not None, "Active agents should have action spaces"

      # Create a partial observation of global state
      assert observability_matrix[agent_i, agent_i] == 1, "Agents should always be able to see themselves"
      visible_agent_indexes = observability_matrix[agent_i].nonzero(as_tuple=True)[0].to(device)
      # Move ego agent to index 0
      visible_agent_indexes = torch.cat([
        torch.tensor([agent_i], device=device),
        visible_agent_indexes[visible_agent_indexes != agent_i]],
        dim=0
      )

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

        # Sample an action, masking out
        action_probs_per_agent[agent.id] = F.softmax(logits[action_space], dim=-1)
        selection = torch.multinomial(action_probs_per_agent[agent.id], 1)
        action_selection_per_agent[agent_i] = action_space[int(selection)] # type: ignore # for action_probs_per_agent[agent.id]
        local_input_features_per_agent[agent.id] = local_input_features
    # end agent loop
    # select active agent ids and put them in the global state
    global_input_features = (
      agent_positions_vector[obs.state.active_mask].unsqueeze(0),
      agent_teams_vector[obs.state.active_mask].unsqueeze(0),
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
      has_discounted_reward=False,
    ))
    obs = next_obs

    if next_obs.done:
      break

  return MultiAgentEpisode(list(environment.agent_map), steps)

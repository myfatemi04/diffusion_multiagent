import torch
import torch.nn as nn

from positional_embeddings import PositionalEncoding

class TransformerNetwork(nn.Module):
    def __init__(self, d_model, patch_size=16, num_heads=8, num_layers=6, num_outputs=5):
        super().__init__()

        self.d_model = d_model
        self.patch_size = patch_size

        # Tiles will be converted into tokens
        # The positional encoding will be based on the center coordinate of the tile
        self.patchify = nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size, padding=0)
        self.tile_positional_encoding = PositionalEncoding(n_position_dims=2, n_encoding_dims=d_model)

        # Use a separate positional encoding module for agent positions
        self.agent_positional_encoding = PositionalEncoding(n_position_dims=2, n_encoding_dims=d_model)

        # Have an additive embedding for whether the agent is a pursuer or an evader
        self.team_encoding = nn.Embedding(2, d_model)

        # Have an embedding for the target position, represented as an additional token
        self.target_position_positional_encoding = PositionalEncoding(n_position_dims=2, n_encoding_dims=d_model)

        # Use transformer encoder layers to process the tokens
        # Anything within the receptive field is fair game, so
        # we'll just have a uniform attention mask
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # Project the transformed embeddings into an output space
        # If num_outputs=5, this represents an action space, or a policy
        # If num_outputs=1, this represents a value function
        # We may, at some point, be interested in looking into non-regression approaches
        # for the Q function.
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_outputs)
        )

    def forward(self, agent_locations, agent_teams, target_location, grid):
        """
        shapes:
        - agent_locations: (batch_size, num_agents, 2)
        - agent_teams: (batch_size, num_agents)
        - target_location: (batch_size, 2)
        - grid: (batch_size, grid_size, grid_size)

        only functional for batch_size of 1
        """

        # Sanity checks
        assert grid.shape[1] % self.patch_size == 0, "Grid size must be divisible by patch size. Grid height was " + str(grid.shape[1])
        assert grid.shape[2] % self.patch_size == 0, "Grid size must be divisible by patch size. Grid width was " + str(grid.shape[2])

        # Patchify the grid
        # Creates a tensor of shape (batch_size, d_model, grid_size // patch_size, grid_size // patch_size)
        terrain_tokens = self.patchify(grid.unsqueeze(1))
        terrain_tokens_xy = torch.stack(torch.meshgrid(
            torch.arange(self.patch_size // 2, grid.shape[1], self.patch_size, device=grid.device),
            torch.arange(self.patch_size // 2, grid.shape[2], self.patch_size, device=grid.device),
            indexing='ij',
        ), dim=-1).float()
        
        # Apply positional encoding to the terrain tokens
        terrain_tokens = terrain_tokens.view(terrain_tokens.shape[0], terrain_tokens.shape[1], -1).permute(0, 2, 1)
        terrain_tokens = terrain_tokens + self.tile_positional_encoding(terrain_tokens_xy)

        # Create agent tokens
        agent_tokens = self.agent_positional_encoding(agent_locations) + self.team_encoding(agent_teams)

        # Create target token
        target_tokens = self.target_position_positional_encoding(target_location).unsqueeze(1)

        # print(agent_tokens.shape)
        # print(terrain_tokens.shape)
        # print(target_tokens.shape)

        # Concatenate the tokens
        tokens = torch.cat([agent_tokens, terrain_tokens, target_tokens], dim=1)

        # Apply transformer encoder
        tokens = self.transformer_encoder(tokens)

        # Project the agent tokens to the output space
        # This will be the policy and value function
        output = self.mlp(tokens[:, :agent_locations.shape[1], :])

        return output


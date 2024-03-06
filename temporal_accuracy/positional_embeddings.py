import torch, torch.nn as nn

class PositionalEncoding(nn.Module):
  def __init__(self, n_position_dims: int, n_encoding_dims: int, max_len: int = 10000):
    super().__init__()

    self.n_position_dims = n_position_dims
    self.n_encoding_dims = n_encoding_dims
    self.max_len = torch.tensor(max_len)
    self.projection = nn.Linear(n_encoding_dims * n_position_dims, n_encoding_dims)

  def forward(self, position: torch.Tensor):
    # position: [batch_size, n_position_dims]
    # [batch_size, n_position_dims, n_angles]
    n_angles = self.n_encoding_dims // 2
    # make a geometric sequence of frequencies.
    # these can be cached.
    log_frequencies = torch.linspace(0, -torch.log(self.max_len),
                             n_angles,
                             dtype=torch.float32,
                             device=position.device) \
                   .unsqueeze(0) \
                   .unsqueeze(0) \
                   .expand(position.shape[0], self.n_position_dims, -1)
    frequencies = torch.exp(log_frequencies) * (2 * torch.pi)
    
    # multiply by the positional value
    angles = frequencies * position.unsqueeze(-1)
    sinusoidal_embeddings = torch.cat([angles.sin(), angles.cos()], dim=-1)

    return self.projection(sinusoidal_embeddings)

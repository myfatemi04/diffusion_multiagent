import torch
import torch.nn as nn

class DenseFeatureMapTransformerPolicy(nn.Module):
  def __init__(self, d_model: int, feature_map_dimensions: int, max_input_size: int, patch_size: int, num_actions: int):
    super().__init__()

    self.d_model = d_model
    self.feature_map_dimensions = feature_map_dimensions
    self.max_input_size = max_input_size
    self.patch_size = patch_size
    self.num_actions = num_actions
    # Converts raw feature map into model features
    self.patchify = nn.Conv2d(
      feature_map_dimensions,
      d_model,
      kernel_size=patch_size,
      stride=patch_size,
      padding=0
    )
    self.embeddings = nn.Embedding(int((max_input_size // patch_size) ** 2), d_model)
    self.action_embedding = nn.Parameter(torch.randn(d_model))
    self.transformer_encoder = nn.TransformerEncoder(
      nn.TransformerEncoderLayer(
        d_model,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        batch_first=True,
      ),
      num_layers=6,
    )
    self.action_head = nn.Linear(d_model, num_actions * 2)

  def forward(self, feature_map: torch.Tensor):
    """
    :params:
    - feature_map: torch.Tensor - The feature map to process

    :returns:
    - action_logits: torch.Tensor - The logits for each action
    - action_values: torch.Tensor - The value estimates for each action
    """
    # Patchify the feature map
    batch_size = feature_map.shape[0]
    feature_map = self.patchify(feature_map.permute((0, 3, 1, 2)))
    feature_map = feature_map.reshape(batch_size, -1, self.d_model)
    feature_map = feature_map + self.embeddings(
      torch.arange(feature_map.shape[1], device=feature_map.device)
    )
    action_embedding = self.action_embedding.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
    tokens = torch.cat([action_embedding, feature_map], dim=1)
    tokens = self.transformer_encoder(tokens)
    action_output = self.action_head(tokens[:, 0, :])
    action_logits = action_output[:, :self.num_actions]
    action_values = action_output[:, self.num_actions:]
    return (action_logits, action_values)

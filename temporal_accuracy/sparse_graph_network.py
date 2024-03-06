import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch_geometric.data

from positional_embeddings import PositionalEncoding

# Instead of traversing a graph of nodes, we can use a graph of semantically meaningful
# entities in the environment while still using an unconstrained motion model.
class SparseGraphNetwork(nn.Module):
    def __init__(self, channel_counts, head_dim):
        super().__init__()
        # these are `lazy`, input_channels=-1 are rederived at first forward() pass
        # and are automatically converted to use the correct message passing functions
        # with heterodata
        convs = []
        lins = []

        for i in range(len(channel_counts)):
            convs.append(gnn.SAGEConv((-1, -1) if i == 0 else channel_counts[i - 1], channel_counts[i]))
            lins.append(gnn.Linear(-1, channel_counts[i]))

        self.convs = nn.ModuleList(convs)
        self.lins = nn.ModuleList(lins)

        # action space: [up, down, left, right]
        # slightly different formulation; to use multi-agent actor critic,
        # i put the other agent's outputs as one-hot encoded inputs to the
        # network and then use this qvalue_head to calculate the Q-value
        # for each agent given that joint action
        self.head = gnn.Linear(channel_counts[-1], head_dim)

    def forward(self, x, edge_index):
        for i, (conv, lin) in enumerate(zip(self.convs, self.lins)):
            x = conv(x, edge_index) + lin(x)
            if i != len(self.convs) - 1:
                x = x.relu()
        return self.head(x)

class SparseGraphNetworkWithPositionalEncoding(nn.Module):
    """
    For now, agents and tasks are solely defined by their coordinates.
    """
    def __init__(self, channel_counts, head_dim, n_encoding_dims=64, positional_encoding_max_value=100):
        self.positional_encoding = PositionalEncoding(
            n_position_dims=2,
            n_encoding_dims=n_encoding_dims,
            max_len=positional_encoding_max_value
        )
        self.net = SparseGraphNetwork(channel_counts, head_dim)

    def make_heterogeneous(self, dummy_features: torch_geometric.data.HeteroData):
        gnn.to_hetero(self.net, dummy_features.metadata(), aggr='sum')
        # Populate lazy-loaded channels
        with torch.no_grad():
            _ = self.net(dummy_features.x_dict, dummy_features.edge_index_dict)
        # Return self to allow this to be called in the same line as it is instantiated in
        return self

    def forward(self, x, edge_index):
        x = {
            node_type_name: self.positional_encoding(x[node_type_name])
            for node_type_name in x.keys()
        }
        return self.net(x, edge_index)

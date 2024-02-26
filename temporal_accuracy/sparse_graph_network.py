import torch.nn as nn
import torch_geometric.nn as gnn

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
            convs.append(gnn.GATConv((-1, -1) if i == 0 else channel_counts[i - 1], channel_counts[i], heads=1, dropout=0.1, add_self_loops=False))
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

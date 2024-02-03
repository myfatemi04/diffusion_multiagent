import torch
import torch.nn.functional as F
import torch_geometric.nn

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_classes, num_layers, dropout, GNNLayer=torch_geometric.nn.GCNConv):
        super(GNN, self).__init__()

        self.conv1 = GNNLayer(num_features, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(GNNLayer(hidden_channels, hidden_channels))
        self.conv2 = GNNLayer(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

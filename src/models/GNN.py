# GNN model for graph classification
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool

class GNN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, conv_type):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.lin = nn.Linear(in_features, out_features)
        self.conv_type = conv_type #save the argument conv_type as an object atribute

        if conv_type == 'gcn':
            self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)

        elif conv_type == 'gat':
            self.conv1 = GATConv(dataset.num_node_features, hidden_channels)
            self.conv2 = GATConv(hidden_channels, hidden_channels)
            self.conv3 = GATConv(hidden_channels, hidden_channels)

        elif conv_type == 'graphsage':
            self.conv1 = SAGEConv(dataset.num_node_features, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, batch):
        # 1. get node embedding
        if self.conv_type == 'gat':
            x = F.elu(self.conv1(x, edge_index))
            x = F.elu(self.conv2(x, edge_index))
            embeddings = x.detach() # save the embeddings after the second layer
            x = self.conv3(x, edge_index)
        else:
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index).relu()
            embeddings = x.detach()
            x = self.conv3(x, edge_index)

        # 2. redout layer, aggregate info
        x = global_mean_pool(x, batch)

        # 3. classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return {"output": x, "embeddings": embeddings}


import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class HomoGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=256, heads=4, dropout=0.2):
        super().__init__()
        self.conv1 = GATv2Conv(
            in_dim, hidden_dim, heads=heads,
            concat=True, dropout=dropout, add_self_loops=True
        )
        self.lin1 = nn.Linear(in_dim, hidden_dim * heads)

        self.conv2 = GATv2Conv(
            hidden_dim * heads, out_dim, heads=1,
            concat=False, dropout=dropout, add_self_loops=True
        )
        self.lin2 = nn.Linear(hidden_dim * heads, out_dim)

        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        x = F.elu(h + self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        h = self.conv2(x, edge_index)
        x = h + self.lin2(x)
        return x
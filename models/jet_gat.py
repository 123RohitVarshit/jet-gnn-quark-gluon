import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool

class JetGAT(nn.Module):
    def __init__(self, in_channels=6, hidden_dim=64, heads=4,
                 num_layers=3, fc_channels=256, num_classes=2,
                 dropout=0.1, edge_dim=4):
        super().__init__()
        self.dropout = dropout

        self.input_proj = nn.Linear(in_channels, hidden_dim * heads)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        ch = hidden_dim * heads
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            self.convs.append(GATv2Conv(
                in_channels   = ch,
                out_channels  = hidden_dim,
                heads         = 1 if is_last else heads,
                concat        = not is_last,
                dropout       = dropout,
                edge_dim      = edge_dim,
                share_weights = False
            ))
            out_ch = hidden_dim if is_last else hidden_dim * heads
            self.norms.append(nn.LayerNorm(out_ch))
            ch = out_ch

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, fc_channels),
            nn.LayerNorm(fc_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_channels, fc_channels // 2),
            nn.LayerNorm(fc_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_channels // 2, num_classes)
        )

    def forward(self, data):
        x          = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr
        batch      = data.batch

        x = F.gelu(self.input_proj(x))

        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index, edge_attr=edge_attr)
            x_new = norm(x_new)
            x_new = F.gelu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new if x.shape == x_new.shape else x_new

        x = torch.cat([global_mean_pool(x, batch),
                        global_max_pool(x, batch)], dim=-1)
        return self.classifier(x)

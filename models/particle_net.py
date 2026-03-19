import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv, global_mean_pool, global_max_pool

def mlp_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Linear(in_ch, out_ch),
        nn.BatchNorm1d(out_ch),
        nn.ReLU()
    )

def make_edge_mlp(in_ch, out_ch):
    return nn.Sequential(
        nn.Linear(2 * in_ch, out_ch),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(),
        nn.Linear(out_ch, out_ch),
        nn.BatchNorm1d(out_ch),
        nn.ReLU()
    )

class ParticleNet(nn.Module):
    def __init__(self, in_channels=6, k=16,
                 edge_channels=(64, 128, 256),
                 fc_channels=256, num_classes=2, dropout=0.1):
        super().__init__()
        self.k = k
        ch = in_channels
        self.convs = nn.ModuleList()
        for out_ch in edge_channels:
            self.convs.append(
                DynamicEdgeConv(make_edge_mlp(ch, out_ch), k=k, aggr='max')
            )
            ch = out_ch

        self.classifier = nn.Sequential(
            nn.Linear(2 * edge_channels[-1], fc_channels),
            nn.BatchNorm1d(fc_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_channels, fc_channels // 2),
            nn.BatchNorm1d(fc_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_channels // 2, num_classes)
        )

    def forward(self, data):
        x, batch = data.x, data.batch
        for conv in self.convs:
            x = conv(x, batch)
        x = torch.cat([global_mean_pool(x, batch),
                        global_max_pool(x, batch)], dim=-1)
        return self.classifier(x)

import torch
import torch.nn as nn
from BiCrossAttentionGate import BidirectionalCrossAttentionGate
from RSAM import rsam


class method(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.2, norm_type=None,
                 conv_type='rsam', gate_type='AttentionGate'):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.Linear(in_channels, hidden_channels)
        self.decoder = nn.Linear(hidden_channels, out_channels)

        self.convs = nn.ModuleList([
            rsam(hidden_channels, hidden_channels, hidden_channels, 0.1)
            if conv_type == 'rsam' else NotImplementedError()
            for _ in range(num_layers)
        ])

        self.norm_self = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) if norm_type == 'bn' else
            nn.LayerNorm(hidden_channels) if norm_type == 'ln' else
            nn.Identity()
            for _ in range(num_layers)
        ])

        self.norm_neigh = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) if norm_type == 'bn' else
            nn.LayerNorm(hidden_channels) if norm_type == 'ln' else
            nn.Identity()
            for _ in range(num_layers)
        ])

        if gate_type == 'AttentionGate':
            self.gate = BidirectionalCrossAttentionGate(hidden_channels)
        elif gate_type == 'directlyFusion':
            self.gate = nn.Linear(hidden_channels * 2, hidden_channels)
        else:
            self.gate = nn.Identity()

        self.gate_type = gate_type

    def forward(self, x, edge_index):
        x = self.encoder(self.dropout(x)).relu()
        h = self.norm_self[0](x)

        for i in range(self.num_layers):
            m = self.dropout(h)
            m = self.convs[i](m, edge_index)
            m = self.norm_neigh[i](m)

            if isinstance(self.gate, nn.Identity):
                h = m
            elif self.gate_type == 'directlyFusion':
                h = self.gate(torch.cat([h, m], dim=1))
            else:
                h = self.gate(h, m)

            h = self.norm_self[i](h)

        out = self.decoder(self.dropout(h))
        return out

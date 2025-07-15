import torch.nn as nn
from torch_sparse import SparseTensor
import torch

from AttentionGate import AttentionGate
from BiCrossAttentionGate3 import BidirectionalCrossAttentionGate
from layers_CAGNN import GCNConv_override, GATConv_override, GINConv_override
from DeProp import DeProp_method
from GCN import SimpleGCN


class CAGNN_method(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.3,
                 norm_type=None, conv_type='DeProp', gate_type='AttentionGate'):
        super().__init__()
        self.transforms = None
        self.learnable_norm_neighb = None
        self.learnable_norm_self = None
        self.conv_layers = None
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.conv_type = conv_type
        self.dropout = nn.Dropout(dropout)
        self.gate_type = gate_type

        # encoder, message passing, decoder
        self.encoder = nn.Linear(in_channels, hidden_channels)
        self.build_message_passing_layers(num_layers, hidden_channels)
        self.decoder = nn.Linear(hidden_channels, out_channels)

        if gate_type == 'convex':
            self.gate = nn.Linear(hidden_channels * 2, 1)
        elif gate_type == 'convex_MLP_2':
            self.gate = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1)
            )
        elif gate_type == 'convex_MLP_3':
            self.gate = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels * 2),
                nn.ReLU(),
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1)
            )
        elif gate_type == 'AttentionGate':
            self.gate = BidirectionalCrossAttentionGate(hidden_channels)
        elif gate_type == 'directlyFusion':
            self.gate = nn.Linear(hidden_channels * 2, hidden_channels)

    def set_adj(self, adj):
        if self.conv_type == 'gat':
            if isinstance(adj, SparseTensor):
                row, col, _ = adj.coo()
                adj = torch.cat([row.unsqueeze(0), col.unsqueeze(0)], dim=0)
            elif isinstance(adj, torch.Tensor):
                if adj.shape[0] == adj.shape[1]:
                    adj = adj.nonzero().t()
        return adj

    def build_message_passing_layers(self, num_layers, hid_channels):
        self.conv_layers = nn.ModuleList()
        self.learnable_norm_self = nn.ModuleList()
        self.learnable_norm_neighb = nn.ModuleList()
        self.transforms = nn.ModuleList()

        for i in range(num_layers):
            if self.conv_type == 'gcn':
                layer = SimpleGCN(64, 64, 64, 0.2)
            elif self.conv_type == 'gin':
                layer = GINConv_override(hid_channels, hid_channels)
            elif self.conv_type == 'gat':
                layer = GATConv_override(hid_channels, hid_channels, dropout=0, drop_edge=0, alpha=0.2, nheads=1)
            elif self.conv_type == 'DeProp':
                layer = DeProp_method(64, 64, 64, 0.1)
            else:
                raise 'not implement'

            self.conv_layers.append(layer)
            if self.norm_type == 'bn':
                self.learnable_norm_self.append(nn.BatchNorm1d(hid_channels))
                self.learnable_norm_neighb.append(nn.BatchNorm1d(hid_channels))
            elif self.norm_type == 'ln':
                self.learnable_norm_self.append(nn.LayerNorm(hid_channels))
                self.learnable_norm_neighb.append(nn.LayerNorm(hid_channels))

    def propagate(self, adj, x, layer_index):
        # x = self.conv_layers[layer_index](adj, x)
        x = self.conv_layers[layer_index](x, adj)
        return x

    def update1(self, self_x, conv_x, edge_index, layer_index=0):
        if self.gate_type in ['convex', 'convex_MLP_2', 'convex_MLP_3']:
            a = self.gate(torch.cat([self_x, conv_x], dim=1)).sigmoid()
            self_x = a * self_x + (1 - a) * conv_x
        elif self.gate_type == 'AttentionGate':
            # a = self.gate(self_x, conv_x)
            # self_x = a * self_x + (1 - a) * conv_x
            self_x = self.gate(self_x, conv_x)
        elif self.gate_type == 'directlyFusion':
            self_x = self.gate(torch.cat([self_x, conv_x], dim=1))
        else:
            self_x = conv_x
        return self_x, conv_x

    def norm(self, x, layer_index=0, is_self=False):
        if self.norm_type in ['bn', 'ln']:
            if is_self:
                x = self.learnable_norm_self[layer_index](x)
            else:
                x = self.learnable_norm_neighb[layer_index](x)
        return x

    def forward(self, x, edge_index):
        # adj = self.set_adj(adj)
        adj = edge_index
        x = self.dropout(x)
        init_x = self.encoder(x).relu()
        init_x = self.norm(init_x, 0, is_self=True)
        self_x, conv_x = init_x, init_x
        for i in range(self.num_layers):
            conv_x = self.dropout(conv_x)
            conv_x = self.propagate(adj, conv_x, i)
            self_x, conv_x = self.update1(self_x, conv_x, adj, i)
            self_x = self.norm(self_x, i, is_self=True)
            conv_x = self.norm(conv_x, i)
        h = self_x
        h = self.dropout(h)
        o = self.decoder(h)
        return o

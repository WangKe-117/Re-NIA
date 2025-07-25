import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor


class rsam(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1,
                 dropout=0.2, use_bn=True, use_norm=False):
        super().__init__()
        self.dropout = dropout

        self.input_block = SingleLayerBlock(in_channels, hidden_channels, dropout, use_bn, use_norm)
        self.hidden_blocks = nn.ModuleList([
            SingleLayerBlock(hidden_channels, hidden_channels, dropout, use_bn, use_norm)
            for _ in range(num_layers)
        ])
        self.output_block = RSAMProp(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.input_block(x, edge_index)
        for block in self.hidden_blocks:
            x = block(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output_block(x, edge_index)
        return x


class SingleLayerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, use_bn, use_norm):
        super().__init__()
        self.prop = RSAMProp(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
        self.use_norm = use_norm
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.prop(x, edge_index)
        x = self.bn(x)
        if self.use_norm:
            x = F.normalize(x, p=2, dim=0)
        x = F.relu(x)
        return x


class RSAMProp(MessagePassing):
    def __init__(self, in_channels, out_channels,
                 improved=False, cached=True,
                 add_self_loops=True, normalize=True,
                 bias=True):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.improved = improved

        self.linear = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')
        self.bias = Parameter(torch.Tensor(out_channels)) if bias else None
        self.weights = nn.Parameter(torch.randn(3))

        self._cached_edge_index = None
        self._cached_adj_t = None
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()
        zeros(self.bias)
        self.weights.data = torch.randn_like(self.weights)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        if self.normalize:
            if isinstance(edge_index, Tensor):
                if self._cached_edge_index is None:
                    edge_index, edge_weight = gcn_norm(
                        edge_index, edge_weight, x.size(0),
                        self.improved, self.add_self_loops
                    )
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = self._cached_edge_index
            elif isinstance(edge_index, SparseTensor):
                if self._cached_adj_t is None:
                    edge_index = gcn_norm(
                        edge_index, edge_weight, x.size(0),
                        self.improved, self.add_self_loops
                    )
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = self._cached_adj_t

        ζ, η, θ = F.softmax(self.weights, dim=0)
        x_proj = self.linear(x)
        res = ζ * x_proj
        agg = η * self.propagate(edge_index, x=x_proj, edge_weight=edge_weight)
        red = θ * x_proj @ (x_proj.t() @ x_proj)
        out = res + agg - red
        if self.bias is not None:
            out += self.bias
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

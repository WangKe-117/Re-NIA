import dgl
import numpy as np
import torch as th
import torch.nn as nn
from typing import Optional, Tuple

from scipy import spatial
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch
import torch.nn.functional as F
from layers_CAGNN import GCNConv_override, GATConv_override, GINConv_override
from torch_geometric.utils import dense_to_sparse
from utils_32 import get_pairwise_sim, torch_corr
import argparse


# def drop_node(feats, drop_rate, training):
#     n = feats.shape[0]
#     drop_rates = th.FloatTensor(np.ones(n) * drop_rate)
#
#     if training:
#
#         masks = th.bernoulli(1. - drop_rates).unsqueeze(1)
#         feats = masks.to(feats.device) * feats
#
#     else:
#         feats = feats * (1. - drop_rate)
#
#     return feats

class KNNGNN_method(torch.nn.Module):
    def __init__(self, num_features, hidden, proj, dropout):
        super().__init__()
        self.lin1 = nn.Linear(num_features, hidden)
        self.gatv1 = GATConv(
            num_features,
            hidden,
            heads=8,
            dropout=dropout, add_self_loops=False)
        self.gatv2 = GATConv(
            hidden * 8,
            hidden,
            heads=1,
            concat=False,
            dropout=dropout, add_self_loops=False)
        self.proj = nn.Linear(2 * hidden, proj)
        self.output_normalization = nn.LayerNorm(2 * hidden)
        self.lin = nn.Linear(4 * hidden, 64)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.gatv1.reset_parameters()
        self.gatv2.reset_parameters()
        self.proj.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        x_MLP = F.dropout(x, p=0, training=self.training)
        x1 = F.relu(self.lin1(x_MLP))
        x_GAT = F.relu(self.gatv1(x, edge_index))
        x2 = F.dropout(x_GAT, p=0, training=self.training)
        x2 = self.gatv2(x2, edge_index)
        final = torch.cat([x1, x2], dim=1)
        g_score = self.proj(final)  # [num_nodes, 1]
        sim_matrix = []
        tree = spatial.KDTree(g_score.cpu().detach().numpy())
        # 查询最近的点
        for i in g_score:
            sim_matrix.append(torch.mean(final[tree.query(torch.tensor(i).cpu(), k=50)[1]], dim=0))
        sim_matrix = torch.stack(sim_matrix)
        sim_matrix = self.output_normalization(sim_matrix)
        out = torch.cat([final, sim_matrix], dim=1)
        out = self.lin(out)
        return out



class KNNGNN(nn.Module):
    def __init__(self, G, hid_dim, n_class, batchnorm, num_diseases, num_mirnas, out_dim, dropout):
        super(KNNGNN, self).__init__()
        self.G = G
        self.hid_dim = hid_dim
        self.n_class = n_class
        self.num_diseases = num_diseases
        self.num_mirnas = num_mirnas
        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)
        self.mirna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0)
        self.m_fc = nn.Linear(G.ndata['m_sim'].shape[1], hid_dim, bias=False)
        self.d_fc = nn.Linear(G.ndata['d_sim'].shape[1], hid_dim, bias=False)
        self.m_fc1 = nn.Linear(out_dim + n_class, out_dim)
        self.d_fc1 = nn.Linear(out_dim + n_class, out_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout = dropout
        self.PROCESS_H = KNNGNN_method(64, 256, 4, 0)
        self.predict = nn.Linear(out_dim * 2, 1)

    def forward(self, diseases, mirnas, training=True):
        self.G.apply_nodes(lambda nodes: {'z': self.dropout1(self.d_fc(nodes.data['d_sim']))}, self.disease_nodes)
        self.G.apply_nodes(lambda nodes: {'z': self.dropout1(self.m_fc(nodes.data['m_sim']))}, self.mirna_nodes)
        feats = self.G.ndata.pop('z')
        x = feats

        diseases1 = diseases.unsqueeze(0)
        mirnas1 = mirnas.unsqueeze(0)
        edge_index = torch.cat((diseases1, mirnas1), dim=0)

        if training:
            h = self.PROCESS_H(x, edge_index)
            feat0 = h
            h_d = th.cat((feat0[:self.num_diseases], feats[:self.num_diseases]), dim=1)
            h_m = th.cat((feat0[self.num_diseases:], feats[self.num_diseases:]), dim=1)
            h_m = self.dropout1(F.elu(self.m_fc1(h_m)))  # (495,64)
            h_d = self.dropout1(F.elu(self.d_fc1(h_d)))
            # (878,64)
            h = th.cat((h_d, h_m), dim=0)
            # 这里的disease和mirnas就是顶点，其对应位置就顶点之间存在边的label：0或者1
            # 疾病顶点特征
            h_diseases = h[diseases]  # disease中有重复的疾病名称;(17376,64)
            # mirnas顶点的特征
            h_mirnas = h[mirnas]
            h_concat = th.cat((h_diseases, h_mirnas), 1)  # (17376,128)
            predict_score = th.sigmoid(self.predict(h_concat))
            return predict_score
        else:
            h = self.PROCESS_H(x, edge_index)
            feat0 = h
            h_d = th.cat((feat0[:self.num_diseases], feats[:self.num_diseases]), dim=1)
            h_m = th.cat((feat0[self.num_diseases:], feats[self.num_diseases:]), dim=1)
            h_m = self.dropout1(F.elu(self.m_fc1(h_m)))  # (495,64)
            h_d = self.dropout1(F.elu(self.d_fc1(h_d)))
            # (878,64)
            h = th.cat((h_d, h_m), dim=0)
            # 这里的disease和mirnas就是顶点，其对应位置就顶点之间存在边的label：0或者1
            # 疾病顶点特征
            h_diseases = h[diseases]  # disease中有重复的疾病名称;(17376,64)
            # mirnas顶点的特征
            h_mirnas = h[mirnas]
            h_concat = th.cat((h_diseases, h_mirnas), 1)  # (17376,128)
            predict_score = th.sigmoid(self.predict(h_concat))
            return predict_score

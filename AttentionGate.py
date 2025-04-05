import math

import torch.nn as nn
import torch
import torch.nn.functional as F


class AttentionGate(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.q = nn.Linear(in_dim, in_dim)
        self.k = nn.Linear(in_dim, in_dim)
        self.v = nn.Linear(in_dim, in_dim)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 全局池化
        self.softmax = nn.Softmax(dim=1)

    def forward(self, self_x, conv_x):
        Q = self.q(self_x)  # (N, d)
        K = self.k(conv_x)  # (N, d)
        V = self.v(conv_x)  # (N, d)

        # 计算注意力分数
        attn_weights = self.softmax(Q @ K.T)  # (N, N)

        # 加权融合 + 全局池化
        attn_out = attn_weights @ V  # (N, d)
        global_info = self.global_pool(attn_out.unsqueeze(-1)).squeeze(-1)  # (N, d)

        return global_info  # 映射到 (0,1)








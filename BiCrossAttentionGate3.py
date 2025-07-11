import torch
import torch.nn as nn


class BidirectionalCrossAttentionGate(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # 双向注意力组件
        # 方向1：self_x -> conv_x
        self.q1 = nn.Linear(in_dim, in_dim)
        self.k1 = nn.Linear(in_dim, in_dim)
        self.v1 = nn.Linear(in_dim, in_dim)

        # 方向2：conv_x -> self_x
        self.q2 = nn.Linear(in_dim, in_dim)
        self.k2 = nn.Linear(in_dim, in_dim)
        self.v2 = nn.Linear(in_dim, in_dim)

        self.fusion_final = nn.Linear(in_dim * 2, in_dim)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, self_x, conv_x):
        # ========== 双向注意力计算 ==========
        # 方向1：self_x关注conv_x
        Q1 = self.q1(self_x)
        K1 = self.k1(conv_x)
        V1 = self.v1(conv_x)
        scale = K1.size(-1) ** 0.5
        attn1 = (Q1 @ K1.transpose(-2, -1)) / scale
        attn_weights1 = self.softmax(attn1)
        attn_out1 = attn_weights1 @ V1  # (N, d)

        # 方向2：conv_x关注self_x
        Q2 = self.q2(conv_x)
        K2 = self.k2(self_x)
        V2 = self.v2(self_x)
        scale = K2.size(-1) ** 0.5
        attn2 = (Q2 @ K2.transpose(-2, -1)) / scale
        attn_weights2 = self.softmax(attn2)
        attn_out2 = attn_weights2 @ V2  # (N, d)

        a1 = torch.sigmoid(torch.tanh(attn_out1))
        a2 = torch.sigmoid(torch.tanh(attn_out2))

        OutInfo1 = a2 * self_x + (1 - a2) * conv_x
        OutInfo2 = a1 * conv_x + (1 - a1) * self_x

        OutInfo = self.fusion_final(torch.cat([OutInfo1, OutInfo2], dim=-1))

        return OutInfo

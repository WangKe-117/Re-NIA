import torch
import torch.nn as nn


class MultiHeadBilinearFusion(nn.Module):
    def __init__(self, input_dim, num_heads=4, rank=16):
        super().__init__()
        self.num_heads = num_heads
        self.rank = rank
        self.input_dim = input_dim
        self.U = nn.Parameter(torch.randn(num_heads, input_dim, rank))
        self.V = nn.Parameter(torch.randn(num_heads, input_dim, rank))
        self.FC = nn.Sequential(
            nn.Linear(num_heads * rank, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, self_x, conv_x):
        proj1 = torch.einsum('nd,hdr->nhr', self_x, self.U)  # [N, H, R]
        proj2 = torch.einsum('nd,hdr->nhr', conv_x, self.V)
        interaction = proj1 * proj2  # 各头独立交互
        fused = self.FC(interaction.flatten(1))  # 拼接后融合
        return fused


class DynamicFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        # 动态权重生成器
        self.weight_generator = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 输出3个权重：对应 AttentionInfo、HighOrderInfo、ResidualInfo
            nn.Softmax(dim=-1)  # 保证权重归一
        )

    def forward(self, AttentionInfo, HighOrderInfo, ResidualInfo):
        # 拼接特征
        concat_features = torch.cat([AttentionInfo, HighOrderInfo, ResidualInfo], dim=-1)  # [N, 3d]
        # 动态生成权重
        weights = self.weight_generator(concat_features)  # [N, 3]
        w1, w2, w3 = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]  # [N, 1]
        # 按节点加权融合
        fused_info = w1 * AttentionInfo + w2 * HighOrderInfo + w3 * ResidualInfo  # [N, d]

        return fused_info


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

        self.fusion_pool = nn.Linear(in_dim * 2, in_dim)
        self.pool_fc1 = nn.Linear(2, in_dim)
        self.pool_fc2 = nn.Linear(2, in_dim)
        self.fusion_res = nn.Linear(in_dim * 2, in_dim)
        self.fusion_attention = nn.Linear(in_dim * 2, in_dim)
        self.highorder = MultiHeadBilinearFusion(in_dim)
        self.agg = DynamicFusion(in_dim)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, self_x, conv_x):
        HighOrderInfo = self.highorder(self_x, conv_x)

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

        fusion_input = torch.cat([attn_out1, attn_out2], dim=-1)
        a = self.sigmoid(self.fusion_attention(fusion_input))
        AttentionInfo = a * attn_out1 + (1 - a) * attn_out2

        res_combined = torch.cat([self_x, conv_x], dim=-1)
        r = self.sigmoid(self.fusion_res(res_combined))
        ResidualInfo = r * self_x + (1 - r) * conv_x

        OutInfo = self.agg(HighOrderInfo, AttentionInfo, ResidualInfo)

        return OutInfo

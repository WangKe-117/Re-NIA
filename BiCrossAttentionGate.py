import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # 全局信息处理组件（新增）
        self.pool_fc1 = nn.Linear(2, in_dim)  # 方向1的统计池化映射
        self.pool_fc2 = nn.Linear(2, in_dim)  # 方向2的统计池化映射

        # 高阶融合组件
        self.tensor_fc = nn.Linear(in_dim ** 2, in_dim)
        self.gate = nn.Linear(in_dim * 3, in_dim * 3)
        self.fusion = nn.Linear(in_dim * 3, in_dim)
        self.res_transform = nn.Linear(in_dim * 2, in_dim)  # 直接处理双注意力输出

        # 输出组件
        self.output = nn.Linear(in_dim, 1)

        # 工具函数
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, self_x, conv_x):
        """
        Args:
            self_x: 自身特征 (N, d)
            conv_x: 邻居聚合特征 (N, d)
        Returns:
            a: 融合权重系数 (N, 1)
        """
        # ========== 双向注意力计算 ==========
        # 方向1：self_x关注conv_x
        Q1 = self.q1(self_x)
        K1 = self.k1(conv_x)
        V1 = self.v1(conv_x)

        # 修正注意力缩放（新增）
        scale = K1.size(-1) ** 0.5
        attn1 = (Q1 @ K1.transpose(-2, -1)) / scale
        attn_weights1 = self.softmax(attn1)
        attn_out1 = attn_weights1 @ V1  # (N, d)

        # 方向2：conv_x关注self_x
        Q2 = self.q2(conv_x)
        K2 = self.k2(self_x)
        V2 = self.v2(self_x)

        # 修正注意力缩放（新增）
        scale = K2.size(-1) ** 0.5
        attn2 = (Q2 @ K2.transpose(-2, -1)) / scale
        attn_weights2 = self.softmax(attn2)
        attn_out2 = attn_weights2 @ V2  # (N, d)

        # 原始残差路径（直接使用注意力输出）
        res_combined = torch.cat([attn_out1, attn_out2], dim=1)  # (N,2d)
        residual = self.res_transform(res_combined)  # (N,d)

        # ========== 全局特征提取 ==========
        # 方向1的全局特征（修正维度处理）
        global_avg1 = attn_out1.mean(dim=1, keepdim=True)  # (N,1)
        global_max1 = attn_out1.max(dim=1, keepdim=True)[0]  # (N,1)
        global_info1 = self.pool_fc1(
            torch.cat([global_avg1, global_max1], dim=1)  # (N,2)
        )  # -> (N,d)

        # 方向2的全局特征（修正维度处理）
        global_avg2 = attn_out2.mean(dim=1, keepdim=True)  # (N,1)
        global_max2 = attn_out2.max(dim=1, keepdim=True)[0]  # (N,1)
        global_info2 = self.pool_fc2(
            torch.cat([global_avg2, global_max2], dim=1)  # (N,2)
        )  # -> (N,d)

        # ========== 高阶融合 ==========
        # 外积交互（保持原设计）
        g1 = global_info1.unsqueeze(-1)  # (N,d,1)
        g2 = global_info2.unsqueeze(1)  # (N,1,d)
        tensor = torch.matmul(g1, g2).flatten(start_dim=1)  # (N,d*d)
        tensor = F.gelu(self.tensor_fc(tensor))  # (N,d)

        combined = torch.cat([global_info1, global_info2, tensor], dim=1)
        gate = self.sigmoid(self.gate(combined))
        gated_combined = gate * combined

        fused = self.fusion(gated_combined)  # (N,d)

        # ========== 改进的残差相加 ==========
        final_fused = fused + residual  # 直接添加原始注意力信息

        a = self.sigmoid(self.output(final_fused))
        return a

import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalCrossAttentionGate(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.q1 = nn.Linear(in_dim, in_dim)
        self.k1 = nn.Linear(in_dim, in_dim)
        self.v1 = nn.Linear(in_dim, in_dim)

        self.q2 = nn.Linear(in_dim, in_dim)
        self.k2 = nn.Linear(in_dim, in_dim)
        self.v2 = nn.Linear(in_dim, in_dim)

        self.pool_fc1 = nn.Linear(2, in_dim)
        self.pool_fc2 = nn.Linear(2, in_dim)

        self.tensor_fc = nn.Linear(in_dim ** 2, in_dim)
        self.gate = nn.Linear(in_dim * 3, in_dim * 3)
        self.fusion = nn.Linear(in_dim * 3, in_dim)
        self.res_transform = nn.Linear(in_dim * 2, in_dim)

        self.output = nn.Linear(in_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, self_x, conv_x):

        Q1 = self.q1(self_x)
        K1 = self.k1(conv_x)
        V1 = self.v1(conv_x)

        scale = K1.size(-1) ** 0.5
        attn1 = (Q1 @ K1.transpose(-2, -1)) / scale
        attn_weights1 = self.softmax(attn1)
        attn_out1 = attn_weights1 @ V1

        Q2 = self.q2(conv_x)
        K2 = self.k2(self_x)
        V2 = self.v2(self_x)

        scale = K2.size(-1) ** 0.5
        attn2 = (Q2 @ K2.transpose(-2, -1)) / scale
        attn_weights2 = self.softmax(attn2)
        attn_out2 = attn_weights2 @ V2

        res_combined = torch.cat([attn_out1, attn_out2], dim=1)
        residual = self.res_transform(res_combined)

        global_avg1 = attn_out1.mean(dim=1, keepdim=True)
        global_max1 = attn_out1.max(dim=1, keepdim=True)[0]
        global_info1 = self.pool_fc1(
            torch.cat([global_avg1, global_max1], dim=1)
        )

        global_avg2 = attn_out2.mean(dim=1, keepdim=True)
        global_max2 = attn_out2.max(dim=1, keepdim=True)[0]
        global_info2 = self.pool_fc2(
            torch.cat([global_avg2, global_max2], dim=1)
        )

        g1 = global_info1.unsqueeze(-1)
        g2 = global_info2.unsqueeze(1)
        tensor = torch.matmul(g1, g2).flatten(start_dim=1)
        tensor = F.gelu(self.tensor_fc(tensor))

        combined = torch.cat([global_info1, global_info2, tensor], dim=1)
        gate = self.sigmoid(self.gate(combined))
        gated_combined = gate * combined

        fused = self.fusion(gated_combined)

        final_fused = fused + residual

        a = self.sigmoid(self.output(final_fused))
        return a

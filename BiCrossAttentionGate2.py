import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree


class TopoAwareMultiHeadFusion(nn.Module):
    def __init__(self, input_dim, num_heads=4, hidden_dim=64, topo_dim=4):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList()

        for _ in range(num_heads):
            self.heads.append(
                nn.Sequential(
                    nn.Linear(input_dim * 2 + topo_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
            )

        self.structure_embed = nn.Linear(1, topo_dim)
        self.residual_proj = nn.Linear(input_dim * 2, input_dim)

    def forward(self, s_prev, h_cur, edge_index):
        num_nodes = s_prev.size(0)
        # 计算结构特征（如度）
        deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float).unsqueeze(-1)  # [N, 1]
        deg_embed = self.structure_embed(deg)  # [N, topo_dim]

        fused = []
        alpha_weights = []

        for head in self.heads:
            # 拼接：节点自身、邻居信息、结构信息
            x = torch.cat([s_prev, h_cur, deg_embed], dim=-1)  # [N, 2d + topo_dim]
            score = torch.sigmoid(head(x))  # [N, 1]
            fused.append(score * h_cur + (1 - score) * s_prev)
            alpha_weights.append(score)

        # 所有 head 融合后平均，或者加权求和
        out = torch.stack(fused, dim=0).mean(dim=0)  # [N, d]
        out = self.residual_proj(torch.cat([out, s_prev], dim=-1))  # 加残差维度映射
        return out

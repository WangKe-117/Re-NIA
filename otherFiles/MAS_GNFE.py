import torch
import torch.nn as nn
import torch.nn.functional as F


class MAS(nn.Module):
    """ Multilayer Attention Score (MAS) Module """

    def __init__(self, in_dim, num_layers=3, alpha_leaky=0.2):
        super(MAS, self).__init__()
        self.num_layers = num_layers
        self.att_layers = nn.ModuleList([
            nn.Linear(in_dim * 2, 1) for _ in range(num_layers)  # 注意力层
        ])
        self.W_layers = nn.ModuleList([
            nn.Linear(in_dim, in_dim) for _ in range(num_layers)  # 线性变换层
        ])
        self.leaky_relu = nn.LeakyReLU(alpha_leaky)  # LeakyReLU 激活函数

    def forward(self, X, A):
        """
        X: 节点特征矩阵 (N, 64)
        A: 邻接矩阵 (N, N)
        return: MAS 计算后的节点特征矩阵 (N, 64)
        """
        N = X.shape[0]  # 节点数量
        A_hat = A + torch.eye(N).to(A.device)  # 加入自环
        D_inv_sqrt = torch.diag(torch.pow(A_hat.sum(1), -0.5))  # 归一化因子

        # 计算每一层的注意力矩阵
        attention_matrices = []
        for l in range(self.num_layers):
            W_h = self.W_layers[l](X)  # 线性变换
            expanded_X = W_h.unsqueeze(1).repeat(1, N, 1)  # 复制 (N, N, 64)
            attention_input = torch.cat([expanded_X, expanded_X.transpose(0, 1)], dim=-1)  # 拼接 (N, N, 128)
            e_ij = self.att_layers[l](attention_input).squeeze(-1)  # 计算注意力分数 (N, N)
            alpha_ij = F.softmax(self.leaky_relu(e_ij), dim=1)  # 归一化注意力
            attention_matrices.append(alpha_ij)

        # 计算 MAS
        A_MAS = torch.ones_like(A).to(A.device)  # 初始化 (N, N)
        for att in attention_matrices:
            A_MAS *= att  # 乘积计算 MAS

        # 生成 MAS 加权特征
        X_MAS = torch.matmul(A_MAS, X)  # (N, 64)

        return X_MAS


class GNFE(nn.Module):
    """ Graph Node Feature Extraction (GNFE) Module """

    def __init__(self, in_dim, out_dim):
        super(GNFE, self).__init__()
        self.gc1 = nn.Linear(in_dim, out_dim)  # 第一层 GCN
        self.gc2 = nn.Linear(out_dim, out_dim)  # 第二层 GCN
        self.gumbel_softmax = nn.Linear(out_dim, out_dim)  # Gumbel SoftMax 特征选择

    def forward(self, X, A):
        """
        X: MAS 处理后的节点特征矩阵 (N, 64)
        A: 邻接矩阵 (N, N)
        return: GNFE 处理后的节点特征矩阵 (N, 64)
        """
        N = X.shape[0]
        A_hat = A + torch.eye(N).to(A.device)  # 加入自环
        D_inv_sqrt = torch.diag(torch.pow(A_hat.sum(1), -0.5))  # 归一化

        # 图卷积
        X = F.relu(self.gc1(torch.matmul(A_hat, X)))  # 第一层 GCN
        X = F.relu(self.gc2(torch.matmul(A_hat, X)))  # 第二层 GCN

        # Gumbel SoftMax 进行特征选择
        logits = self.gumbel_softmax(X)  # 计算 logits
        selection = F.gumbel_softmax(logits, tau=0.5, hard=True)  # 选择重要特征
        X_selected = X * selection  # 应用特征选择

        return X_selected


class MAS_GNFE(nn.Module):
    """ 整合 MAS + GNFE 处理 """

    def __init__(self, in_dim=64, out_dim=64, num_layers=3):
        super(MAS_GNFE, self).__init__()
        self.mas = MAS(in_dim, num_layers)
        self.gnfe = GNFE(in_dim, out_dim)

    def forward(self, X, A):
        X_MAS = self.mas(X, A)  # 先进行 MAS 处理
        X_out = self.gnfe(X_MAS, A)  # 再进行 GNFE 处理
        return X_out


# 测试代码
if __name__ == "__main__":
    N, F = 100, 64  # 假设有 100 个节点，每个节点 64 维特征
    X = torch.rand(N, F)  # 随机生成节点特征矩阵
    A = (torch.rand(N, N) > 0.8).float()  # 生成随机邻接矩阵（稀疏图）

    model = MAS_GNFE()
    X_out = model(X, A)  # 通过模型
    print("输出形状:", X_out.shape)  # 应该是 (100, 64)

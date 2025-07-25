import torch

from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx


def detect_change(pred_label1, pred_label2):
    pulse_matrix1 = (pred_label1[:, 1:] != pred_label1[:, :-1]).long()  # [N, E-1]
    p1 = pulse_matrix1.cpu().numpy()

    pulse_matrix2 = (pred_label2[:, 1:] != pred_label2[:, :-1]).long()  # [N, E-1]
    p2 = pulse_matrix2.cpu().numpy()

    r1 = get_unstable_index(pulse_matrix1)
    r2 = get_unstable_index(pulse_matrix2)

    result = list(set(r1) & set(r2))

    return result


def get_unstable_index(pulse_matrix):
    # 每个节点跳变次数
    jump_counts = pulse_matrix.sum(dim=1).long()  # [N]，值域 0~E

    max_jump = int(pulse_matrix.shape[1])  # E

    # 统计直方图（GPU实现）
    hist = torch.bincount(jump_counts, minlength=max_jump + 1).float()  # [E+1]
    total = hist.sum()
    prob = hist / total

    omega = torch.cumsum(prob, dim=0)  # 前缀和
    mu = torch.cumsum(prob * torch.arange(max_jump + 1, device=prob.device), dim=0)
    mu_t = mu[-1]

    # 类间方差公式
    sigma_b_squared = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-6)

    # 找最大类间方差对应的跳变次数阈值
    _, best_thresh = torch.max(sigma_b_squared, dim=0)

    # 判定：跳变次数 > 阈值 → 不稳定
    unstable_mask = (jump_counts > best_thresh)
    unstable_indices = unstable_mask.nonzero(as_tuple=False).squeeze()

    r = unstable_indices.tolist()
    return r


def relearn(model, optimizer, src_unstable, dst_unstable, label_relearn, cul_loss):
    model.train()
    score_relearn = model(src_unstable, dst_unstable, True)
    loss = cul_loss(score_relearn, label_relearn, model)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach()
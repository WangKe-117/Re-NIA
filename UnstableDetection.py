import torch


def detect_change(pred_label1, pred_label2):
    pulse_matrix1 = (pred_label1[:, 1:] != pred_label1[:, :-1]).long()  # [N, E-1]

    pulse_matrix2 = (pred_label2[:, 1:] != pred_label2[:, :-1]).long()  # [N, E-1]

    r1 = get_unstable_index(pulse_matrix1)
    r2 = get_unstable_index(pulse_matrix2)

    result = list(set(r1) & set(r2))

    return result


def get_unstable_index(pulse_matrix):
    jump_counts = pulse_matrix.sum(dim=1).long()

    max_jump = int(pulse_matrix.shape[1])

    hist = torch.bincount(jump_counts, minlength=max_jump + 1).float()
    total = hist.sum()
    prob = hist / total

    omega = torch.cumsum(prob, dim=0)
    mu = torch.cumsum(prob * torch.arange(max_jump + 1, device=prob.device), dim=0)
    mu_t = mu[-1]

    sigma_b_squared = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-6)

    _, best_thresh = torch.max(sigma_b_squared, dim=0)

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

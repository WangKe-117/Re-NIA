import numpy as np


def evalaution(pred_label):
    pl = pred_label.cpu().numpy()

    pulse_matrix = (pred_label[:, 1:] != pred_label[:, :-1]).long()  # [N, E-1]
    p = pulse_matrix.cpu().numpy()

    WFRL = computeWFRL(p)

    return WFRL


def computeWFRL(pulse_matrix):
    E, K_minus_1 = pulse_matrix.shape
    wfrl_list = []

    for i in range(E):
        seq = pulse_matrix[i]
        # 找连续1段长度
        lengths = []
        count = 0.0
        for val in seq:
            if val == 1:
                count += 1
            else:
                if count > 0:
                    lengths.append(count)
                    count = 0.0
        # 末尾如果有连续1段
        if count > 0:
            lengths.append(count)

        if len(lengths) == 0:
            # 无翻转段，稳定，长度为0
            wfrl_list.append(0.0)
        else:
            # 计算加权平均 R_i = sum(L^2) / sum(L)
            numerator = sum([l ** 2 for l in lengths])
            denominator = sum(lengths)
            wfrl = numerator / denominator
            wfrl_list.append(wfrl)
    rlist = np.array(wfrl_list)
    # 全局平均
    return np.mean(rlist)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp
from sklearn import metrics
import torch
import torch.nn as nn
import dgl
from numpy.linalg import eig, eigh
import pandas as pd


def load_data(directory, random_seed):
    D_SSM1 = np.loadtxt(directory + '/D_SSM1.txt')
    D_SSM2 = np.loadtxt(directory + '/D_SSM2.txt')
    D_GSM = np.loadtxt(directory + '/D_GSM.txt')
    M_FSM = np.loadtxt(directory + '/M_FSM.txt')
    M_GSM = np.loadtxt(directory + '/M_GSM.txt')
    all_associations = pd.read_csv(directory + '/all_mirna_disease_pairs.csv', names=['miRNA', 'disease', 'label'])

    D_SSM = (D_SSM1 + D_SSM2) / 2
    ID = D_SSM
    IM = M_FSM
    for i in range(D_SSM.shape[0]):
        for j in range(D_SSM.shape[1]):
            if ID[i][j] == 0:
                ID[i][j] = D_GSM[i][j]

    for i in range(M_FSM.shape[0]):
        for j in range(M_FSM.shape[1]):
            if IM[i][j] == 0:
                IM[i][j] = M_GSM[i][j]

    # D_SSM1 = np.loadtxt(directory + '/D_SSM1.txt')
    # D_SSM2 = np.loadtxt(directory + '/D_SSM2.txt')
    # D_GSM = np.loadtxt(directory + '/D_GSM.txt')
    # M_FSM = np.loadtxt(directory + '/M_FSM.txt')
    # M_GSM = np.loadtxt(directory + '/M_GSM.txt')
    # all_associations = pd.read_csv(directory + '/all_mirna_disease_pairs.csv', names=['miRNA', 'disease', 'label'])
    #
    # D_SSM = (D_SSM1 + D_SSM2) / 2
    # ID = D_GSM
    # IM = M_GSM
    # for i in range(D_SSM.shape[0]):
    #     for j in range(D_SSM.shape[1]):
    #         if ID[i][j] == 0:
    #             ID[i][j] = D_GSM[i][j]
    #
    # for i in range(M_FSM.shape[0]):
    #     for j in range(M_FSM.shape[1]):
    #         if IM[i][j] == 0:
    #             IM[i][j] = M_GSM[i][j]
    known_associations = all_associations.loc[all_associations['label'] == 1]
    unknown_associations = all_associations.loc[all_associations['label'] == 0]
    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0)

    sample_df = known_associations.append(random_negative)
    sample_df.reset_index(drop=True, inplace=True)

    samples = sample_df.values  # 获得重新编号的新样本

    return ID, IM, samples  # 未知关联数量较多，选择和已知关联数目相同的未知关联组成样本


def build_graph(directory, random_seed):
    ID, IM, samples = load_data(directory, random_seed)

    # print(adj)
    # k=0
    # for i in range(adj.shape[0]):
    #     if(adj[0][i]==1):
    #         k=k+1
    # print(k)
    # miRNA和disease二元异质图
    g = dgl.DGLGraph()  # 创建一个空的DGLGraph对象
    g.add_nodes(ID.shape[0] + IM.shape[0])  # 添加节点到图中，节点的数量为miRNA和disease的总数
    node_type = torch.zeros(g.number_of_nodes(), dtype=torch.int64)  # 创建一个张量来表示节点类型，初始值为0
    node_type[: ID.shape[0]] = 1  # 将前ID.shape[0]个节点标记为1，表示它们是disease节点
    #0-374 disease
    g.ndata['type'] = node_type  # 将节点类型数据存储在图的节点特征'data'中

    d_sim = torch.zeros(g.number_of_nodes(), ID.shape[1])  # 创建一个张量来存储disease相i似性特征，初始值为0
    d_sim[: ID.shape[0], :] = torch.from_numpy(ID.astype('float32'))  # 将从文件加载的dsease相似性特征存储在d_sim张量中
    g.ndata['d_sim'] = d_sim  # 将d_sim存储在图的节点特征'data'中

    m_sim = torch.zeros(g.number_of_nodes(), IM.shape[1])  # 创建一个张量来存储miRNA相似性特征，初始值为0
    m_sim[ID.shape[0]: ID.shape[0] + IM.shape[0], :] = torch.from_numpy(
        IM.astype('float32'))  # 将从文件加载的miRNA相似性特征存储在m_sim张量中
    g.ndata['m_sim'] = m_sim  # 将m_sim存储在图的节点特征'data'中

    disease_ids = list(range(1, ID.shape[0] + 1))  # 创建一个包含disease节点ID的列表
    mirna_ids = list(range(1, IM.shape[0] + 1))  # 创建一个包含miRNA节点ID的列表

    disease_ids_invmap = {id_: i for i, id_ in enumerate(disease_ids)}  # 创建一个字典，将disease节点ID映射到索引
    mirna_ids_invmap = {id_: i for i, id_ in enumerate(mirna_ids)}  # 创建一个字典，将miRNA节点ID映射到索引

    sample_disease_vertices = [disease_ids_invmap[id_] for id_ in samples[:, 1]]  # 从samples中获取disease节点的索引
    sample_mirna_vertices = [mirna_ids_invmap[id_] + ID.shape[0] for id_ in samples[:, 0]]  # 从samples中获取miRNA节点的索引

    g.add_edges(sample_disease_vertices, sample_mirna_vertices,
                data={'label': torch.from_numpy(
                    samples[:, 2].astype('float32'))})  # 在图中添加从disease到miRNA的边，带有'label'特征，特征值为samples中的第三列
    g.add_edges(sample_mirna_vertices, sample_disease_vertices,
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})
    # e,u = eigen_decompositon(Adj)
    # e = torch.FloatTensor(e)
    # u = torch.FloatTensor(u)

    # 在图中添加从miRNA到disease的边，带有'label'特征，特征值为samples中的第三列
    # g.readonly()  # 将图设置为只读，禁止修改图的结构和特征

    return g, sample_disease_vertices, sample_mirna_vertices, ID, IM, samples  # 返回构建的图以及其他相关变量


def feature_normalize(x):
    x = np.array(x)
    rowsum = x.sum(axis=1, keepdims=True)
    rowsum = np.clip(rowsum, 1, 1e10)
    return x / rowsum


def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()


def plot_auc_curves(fprs, tprs, auc, directory, name):
    mean_fpr = np.linspace(0, 1, 20000)
    tpr = []

    for i in range(len(fprs)):
        tpr.append(interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.4, linestyle='--', label='Fold %d AUC: %.4f' % (i + 1, auc[i]))

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    # mean_auc = metrics.auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(auc)
    auc_std = np.std(auc)
    plt.plot(mean_fpr, mean_tpr, color='BlueViolet', alpha=0.9, label='Mean AUC: %.4f $\pm$ %.4f' % (mean_auc, auc_std))

    plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.4)

    # std_tpr = np.std(tpr, axis=0)
    # tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='LightSkyBlue', alpha=0.3, label='$\pm$ 1 std.dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.legend(loc='lower right')
    plt.savefig(directory + '/%s.jpg' % name, dpi=1200, bbox_inches='tight')
    plt.close()


def plot_prc_curves(precisions, recalls, prc, directory, name):
    mean_recall = np.linspace(0, 1, 20000)
    precision = []

    for i in range(len(recalls)):
        precision.append(interp(1 - mean_recall, 1 - recalls[i], precisions[i]))
        precision[-1][0] = 1.0
        plt.plot(recalls[i], precisions[i], alpha=0.4, linestyle='--', label='Fold %d AP: %.4f' % (i + 1, prc[i]))

    mean_precision = np.mean(precision, axis=0)
    mean_precision[-1] = 0
    # mean_prc = metrics.auc(mean_recall, mean_precision)
    mean_prc = np.mean(prc)
    prc_std = np.std(prc)
    plt.plot(mean_recall, mean_precision, color='BlueViolet', alpha=0.9,
             label='Mean AP: %.4f $\pm$ %.4f' % (mean_prc, prc_std))  # AP: Average Precision

    plt.plot([1, 0], [0, 1], linestyle='--', color='black', alpha=0.4)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('P-R curves')
    plt.legend(loc='lower left')
    plt.savefig(directory + '/%s.jpg' % name, dpi=1200, bbox_inches='tight')
    plt.close()


def normalize_graph(g):
    g = np.array(g)
    g = g + g.T
    g[g > 0.] = 1.0
    deg = g.sum(axis=1).reshape(-1)
    deg[deg == 0.] = 1.0
    deg = np.diag(deg ** -0.5)
    adj = np.dot(np.dot(deg, g), deg)
    L = np.eye(g.shape[0]) - adj
    return L


def eigen_decompositon(g):
    "The normalized (unit “length”) eigenvectors, "
    "such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]."
    g = normalize_graph(g)
    e, u = eigh(g)
    return e, u


import argparse
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import scipy.sparse as sp
import torch


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool).cuda()
    mask[index] = 1
    return mask


def mask_to_index(mask):
    index = torch.where(mask == True)[0].cuda()
    return index


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]  # 创建一个空列表用来存每次实验的结果

    def add_result(self, run, result):
        assert len(result) == 3  # 确保实验结果是包含三个元素的列表[train_result, valid_result, test_result]。
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'{self.info} Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            # import ipdb; ipdb.set_trace()
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                argmax = r[:, 1].argmax()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'{self.info} All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            br = r.mean()
            cr = r.std()
            return br, cr

    def best_result(self, run=None, with_var=False):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            train1 = result[:, 0].max()
            valid = result[:, 1].max()
            train2 = result[argmax, 0]
            test = result[argmax, 2]
            return (train1, valid, train2, test)
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                argmax = r[:, 1].argmax()
                train2 = r[argmax, 0].item()
                test = r[argmax, 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            r = best_result[:, 0]
            train1 = r.mean().item()
            train1_var = f'{r.mean():.2f} ± {r.std():.2f}'

            r = best_result[:, 1]
            valid = r.mean().item()
            valid_var = f'{r.mean():.2f} ± {r.std():.2f}'

            r = best_result[:, 2]
            train2 = r.mean().item()
            train2_var = f'{r.mean():.2f} ± {r.std():.2f}'

            r = best_result[:, 3]
            test = r.mean().item()
            test_var = f'{r.mean():.2f} ± {r.std():.2f}'

            if with_var:
                return (train1, valid, train2, test, train1_var, valid_var, train2_var, test_var)
            else:
                return (train1, valid, train2, test)  # 如果指定了 run，返回该实验的最佳结果（最高训练分数、最高验证分数、最佳训练分数和最终测试分数）。


# 如果没有指定 run，返回所有实验的平均最佳结果：


def torch_corr(x):
    mean_x = torch.mean(x, 1)  # 计算特征维度上的均值
    # xm = x.sub(mean_x.expand_as(x))
    xm = x - mean_x.view(-1, 1)
    c = xm.mm(xm.t())  # 协方差矩阵
    # c = c / (x.size(1) - 1)

    # normalize covariance matrix
    d = torch.diag(c)  # 对角线是方差
    stddev = torch.pow(d, 0.5)  # 标准差
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)
    return c


def get_pairwise_sim(x):  # 目的是计算输入矩阵 x 中每一对样本（行）之间的欧氏距离，并返回一个代表样本对相似性的度量
    try:
        x = x.detach().cpu().numpy()  # 将张量x从计算图中分离，转移到 CPU 上，并转换为 NumPy 数组不再参与梯度计算
    except:
        pass

    if sp.issparse(x):
        x = x.todense()
        x = x / (np.sqrt(np.square(x).sum(1))).reshape(-1, 1)  # 计算方根，对每一行进行归一化
        x = sp.csr_matrix(x)  # 在转回稀疏矩阵
    else:
        x = x / (np.sqrt(np.square(x).sum(1)) + 1e-10).reshape(-1, 1)
    # x = x / x.sum(1).reshape(-1,1)
    try:
        dis = euclidean_distances(x)  # 计算欧式距离
        return 0.5 * (dis.sum(1) / (dis.shape[1] - 1)).mean()  # 计算相似度
    except:
        return -1

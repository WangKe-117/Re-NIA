# import time
# import numpy as np
# import pandas as pd
# import random
# import torch
# from sklearn.model_selection import KFold
# from sklearn import metrics
# from cul_loss import CustomLossWithRegularization
# import logging
# from MainModel import PREDICTOR
# from utils import build_graph, weight_reset
# from sklearn.cluster import SpectralClustering
#
#
# # 配置日志输出到文件
# logging.basicConfig(filename='output.log', level=logging.INFO, format='%(message)s')
#
#
# def PreTrain(directory, epochs, n_classes, in_size, out_dim, dropout, lr, wd, random_seed, cuda):
#     random.seed(random_seed)
#     np.random.seed(random_seed)
#     torch.manual_seed(random_seed)
#     if torch.cuda.is_available():
#         if not cuda:
#             print('WARNING: You have a CUDA device, so you should probably run with --cuda')
#         else:
#             torch.cuda.manual_seed(random_seed)
#     context = torch.device('cuda')
#     g, disease_vertices, mirna_vertices, ID, IM, samples = build_graph(directory, random_seed)
#     samples_df = pd.DataFrame(samples, columns=['miRNA', 'disease', 'label'])
#     auc_result = []
#     acc_result = []
#     pre_result = []
#     recall_result = []
#     f1_result = []
#     prc_result = []
#     fprs = []
#     tprs = []
#     precisions = []
#     recalls = []
#     patience = 200
#     i = 0
#     kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
#     for train_idx, test_idx in kf.split(samples[:, 2]):
#         # 返回训练集和测试集的索引train：test 4:1
#         best_val_auc = 0.0
#         i += 1
#         print('Training for Fold', i)
#         samples_df['train'] = 0
#         samples_df['train'].iloc[train_idx] = 1  # 多加一列，将训练集记为1
#         train_tensor = torch.from_numpy(samples_df['train'].values.astype('int64'))
#         edge_data = {'train': train_tensor}
#         g = g.to('cpu')
#         g.edges[disease_vertices, mirna_vertices].data.update(edge_data)  # 正向反向加边，更新边上的数据
#         g.edges[mirna_vertices, disease_vertices].data.update(edge_data)
#         train_eid = g.filter_edges(lambda edges: edges.data['train'])  # 过滤出被记为train的边
#         g_train = g.edge_subgraph(train_eid, relabel_nodes=False)  # 从异构图中创建子图，train集的子图
#         # g_train.copy_from_parent()
#         label_train = g_train.edata['label'].unsqueeze(1)
#
#         src_train, dst_train = g_train.all_edges()  # 训练集的边
#
#         test_eid = g.filter_edges(lambda edges: edges.data['train'] == 0)  # 原图中选出标记为0的记为测试集
#         src_test, dst_test = g.find_edges(test_eid)
#         label_test = g.edges[test_eid].data['label'].unsqueeze(1)  # 测试集的边
#         print('## Training edges:', len(train_eid))
#         print('## Testing edges:', len(test_eid))
#
#         g = g.to(context)
#         g_train = g_train.to(context)
#         label_train = label_train.to(context)
#         src_train = src_train.to(context)
#         dst_train = dst_train.to(context)
#         label_test = label_test.to(context)
#         src_test = src_test.to(context)
#         dst_test = dst_test.to(context)
#
#         model = PREDICTOR(G_train=g_train,
#                           G_all=g,
#                           hid_dim=in_size,
#                           n_class=n_classes,
#                           batchnorm=True,
#                           num_diseases=ID.shape[0],
#                           num_mirnas=IM.shape[0],
#                           out_dim=out_dim,
#                           dropout=dropout)
#         model.apply(weight_reset)
#         model = model.to(context)
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
#
#         lambda_reg = 0.001  # 调整此值以控制正则化的强度
#         cul_loss = CustomLossWithRegularization(lambda_reg)
#         patience_counter = 0
#         best_model_state = {}
#
#         pmatrix = list()
#         for epoch in range(10):
#             start = time.time()
#
#             model.train()
#             with torch.autograd.set_detect_anomaly(True):
#                 score_train = model(src_train, dst_train, True)  # train集子图进入model训练
#                 loss_train = cul_loss(score_train, label_train, model)
#
#                 score_train_cpu = np.squeeze(score_train.cpu().detach().numpy())
#                 pred_label = [0 if j < 0.5 else 1 for j in score_train_cpu]
#                 pmatrix.append((torch.tensor(pred_label)).numpy())
#                 last_epochs_pred_matrix = np.array(pmatrix).T
#
#                 optimizer.zero_grad()  # 梯度置零
#                 loss_train.backward()  # 反向传播
#                 optimizer.step()
#
#         changable_nodes = detect_change(last_epochs_pred_matrix)
#
#         continue
#
#
# def detect_change(pred_label_array):
#     """
#     输入：
#         pred_label_array: np.ndarray, shape = [N, E]
#             每行是一个 miRNA-疾病对在最近 E 个 epoch 的预测标签（0 或 1）
#
#     输出：
#         changable_nodes: list[int]
#             被判定为预测不稳定的样本索引（行号）
#     """
#     N, E = pred_label_array.shape
#
#     # 构建脉冲矩阵（每一行记录 label 是否发生变化）
#     changable_array = []
#     for i in range(N):
#         temp_list = []
#         for j in range(1, E):
#             if pred_label_array[i][j] == pred_label_array[i][j - 1]:
#                 temp_list.append(0)
#             else:
#                 temp_list.append(1)
#         changable_array.append(temp_list)  # shape = [N, E-1]
#
#     changable_array = np.array(changable_array)
#
#     # 使用谱聚类将样本划分为稳定 / 不稳定 两类
#     model = SpectralClustering(n_clusters=2, affinity='rbf', random_state=42)
#     model.fit(changable_array)
#     result = model.labels_  # shape = [N]
#
#     # 将较小类别视为“不稳定”类
#     count_0 = np.sum(result == 0)
#     count_1 = np.sum(result == 1)
#     unstable_label = 0 if count_0 < count_1 else 1
#
#     # 获取不稳定节点的索引
#     changable_nodes = list(np.where(result == unstable_label)[0])
#
#     return changable_nodes
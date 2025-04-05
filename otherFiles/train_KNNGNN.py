import time
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn import metrics
import scipy.sparse as sp
from scipy.sparse import csc_matrix, coo_matrix
import math
from cul_loss import CustomLossWithRegularization
import logging
import matplotlib.pyplot as plt
from MainModel import CAGNN
from model_KNNGNN import KNNGNN

# 配置日志输出到文件
logging.basicConfig(filename='../output.log', level=logging.INFO, format='%(message)s')
from utils_32 import build_graph, weight_reset


def Train(directory, epochs, n_classes, in_size, out_dim, dropout, lr, wd, random_seed, cuda):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        if not cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')
        else:
            torch.cuda.manual_seed(random_seed)
    context = torch.device('cpu')
    g, disease_vertices, mirna_vertices, ID, IM, samples = build_graph(directory, random_seed)
    samples_df = pd.DataFrame(samples, columns=['miRNA', 'disease', 'label'])
    g.to(context)
    auc_result = []
    acc_result = []
    pre_result = []
    recall_result = []
    f1_result = []
    prc_result = []
    fprs = []
    tprs = []
    precisions = []
    recalls = []
    patience = 100
    i = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    for train_idx, test_idx in kf.split(samples[:, 2]):  # 返回训练集和测试集的索引train：test 4:1
        best_val_auc = 0.0
        i += 1
        print('Training for Fold', i)
        samples_df['train'] = 0
        samples_df['train'].iloc[train_idx] = 1  # 多加一列，将训练集记为1
        train_tensor = torch.from_numpy(samples_df['train'].values.astype('int64'))
        edge_data = {'train': train_tensor}
        g.edges[disease_vertices, mirna_vertices].data.update(edge_data)  # 正向反向加边，更新边上的数据
        g.edges[mirna_vertices, disease_vertices].data.update(edge_data)
        train_eid = g.filter_edges(lambda edges: edges.data['train'])  # 过滤出被记为train的边
        g_train = g.edge_subgraph(train_eid, relabel_nodes=False)  # 从异构图中创建子图，train集的子图
        # g_train.copy_from_parent()
        label_train = g_train.edata['label'].unsqueeze(1)
        src_train, dst_train = g_train.all_edges()  # 训练集的边
        test_eid = g.filter_edges(lambda edges: edges.data['train'] == 0)  # 原图中选出标记为0的记为测试集
        src_test, dst_test = g.find_edges(test_eid)
        label_test = g.edges[test_eid].data['label'].unsqueeze(1)  # 测试集的边
        print('## Training edges:', len(train_eid))
        print('## Testing edges:', len(test_eid))
        model = KNNGNN(G=g_train,
                       hid_dim=in_size,
                       n_class=n_classes,
                       batchnorm=True,
                       num_diseases=ID.shape[0],
                       num_mirnas=IM.shape[0],
                       out_dim=out_dim,
                       dropout=dropout)
        model.apply(weight_reset)
        model.to(context)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        lambda_reg = 0.001  # 调整此值以控制正则化的强度
        cul_loss = CustomLossWithRegularization(lambda_reg)

        for epoch in range(epochs):
            start = time.time()

            model.train()
            with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad()
                score_train = model(src_train, dst_train, True)  # train集子图进入model训练
                loss_train = cul_loss(score_train, label_train, model)
                # loss_train += 0.001 * model.ortho_loss()

                # optimizer.zero_grad()   # 梯度置零
                loss_train.backward()  # 反向传播
                optimizer.step()

            model.eval()
            with torch.no_grad():  # with torch.no_grad()或者@torch.no_grad()中的数据不需要计算梯度，也不会进行反向传播
                score_val = model(src_test, dst_test, False)  # 注意在整个图g中训练测试集
                loss_val = cul_loss(score_val, label_test, model)

            score_train_cpu = np.squeeze(score_train.cpu().detach().numpy())  # 在深度学习训练后，需要计算每个epoch得到的模型的训练效果的时候，
            score_val_cpu = np.squeeze(score_val.cpu().detach().numpy())  # 一般会用到detach() item() cpu() numpy()等函数。
            label_train_cpu = np.squeeze(label_train.cpu().detach().numpy())
            label_val_cpu = np.squeeze(label_test.cpu().detach().numpy())

            train_auc = metrics.roc_auc_score(label_train_cpu, score_train_cpu)
            val_auc = metrics.roc_auc_score(label_val_cpu, score_val_cpu)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                # 检查是否应该停止
            if patience_counter >= patience:
                break

            pred_val = [0 if j < 0.5 else 1 for j in score_val_cpu]
            acc_val = metrics.accuracy_score(label_val_cpu, pred_val)
            pre_val = metrics.precision_score(label_val_cpu, pred_val)
            recall_val = metrics.recall_score(label_val_cpu, pred_val)
            f1_val = metrics.f1_score(label_val_cpu, pred_val)

            end = time.time()
            if (epoch + 1) % 10 == 0:
                print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.item(),
                      'Val Loss: %.4f' % loss_val.cpu().detach().numpy(),
                      'Acc: %.4f' % acc_val, 'Pre: %.4f' % pre_val, 'Recall: %.4f' % recall_val, 'F1: %.4f' % f1_val,
                      'Train AUC: %.4f' % train_auc, 'Val AUC: %.4f' % val_auc, 'Time: %.2f' % (end - start))
            # if (epoch + 1) % 10 == 0:
            #     print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.item(),
            #           'Val Loss: %.4f' % loss_val.cpu().detach().numpy(),
            #           'Acc: %.4f' % acc_val, 'Pre: %.4f' % pre_val, 'Recall: %.4f' % recall_val, 'F1: %.4f' % f1_val,
            #           'Train AUC: %.4f' % train_auc, 'Val AUC: %.4f' % val_auc)
            #     message = f"Fold {i}: {val_auc:.4f}"
            #     logging.info(message)
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            model.load_state_dict(best_model_state)
            score_test = model(src_test, dst_test, False)  # 测试分数和验证分数相同？？？
        score_test_cpu = np.squeeze(score_test.cpu().detach().numpy())  # np.squeeze删除指定的维度
        label_test_cpu = np.squeeze(label_test.cpu().detach().numpy())

        fpr, tpr, thresholds = metrics.roc_curve(label_test_cpu, score_test_cpu)
        precision, recall, _ = metrics.precision_recall_curve(label_test_cpu, score_test_cpu)
        test_auc = metrics.auc(fpr, tpr)
        test_prc = metrics.auc(recall, precision)

        pred_test = [0 if j < 0.5 else 1 for j in score_test_cpu]
        acc_test = metrics.accuracy_score(label_test_cpu, pred_test)
        pre_test = metrics.precision_score(label_test_cpu, pred_test)
        recall_test = metrics.recall_score(label_test_cpu, pred_test)
        f1_test = metrics.f1_score(label_test_cpu, pred_test)
        print(f"Fold completed. Best validation AUC: {best_val_auc}")
        print('Fold: ', i, 'Test acc: %.4f' % acc_test, 'Test Pre: %.4f' % pre_test,
              'Test Recall: %.4f' % recall_test, 'Test F1: %.4f' % f1_test, 'Test PRC: %.4f' % test_prc,
              'Test AUC: %.4f' % best_val_auc)

        auc_result.append(best_val_auc)
        acc_result.append(acc_test)
        pre_result.append(pre_test)
        recall_result.append(recall_test)
        f1_result.append(f1_test)
        prc_result.append(test_prc)

        fprs.append(fpr)
        tprs.append(tpr)
        precisions.append(precision)
        recalls.append(recall)

    print('## Training Finished !')
    print('-----------------------------------------------------------------------------------------------')
    print('-AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc_result), np.std(auc_result)),
          'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc_result), np.std(acc_result)),
          'Precision mean: %.4f, variance: %.4f \n' % (np.mean(pre_result), np.std(pre_result)),
          'Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall_result), np.std(recall_result)),
          'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1_result), np.std(f1_result)),
          'PRC mean: %.4f, variance: %.4f \n' % (np.mean(prc_result), np.std(prc_result)))

    return fprs, tprs, auc_result, precisions, recalls, prc_result

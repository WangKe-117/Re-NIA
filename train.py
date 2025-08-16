import time
import numpy as np
import pandas as pd
import random
import torch
from sklearn.model_selection import KFold
from sklearn import metrics
from cul_loss import CustomLossWithRegularization
import logging
from MainModel import PREDICTOR
from ReLearnModel import PREDICTOR_RELEARN
from utils import build_graph, weight_reset
from UnstableDetection import detect_change, relearn


def Train(directory, epochs, n_classes, in_size, out_dim, dropout, lr, wd, random_seed, cuda):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        if not cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')
        else:
            torch.cuda.manual_seed(random_seed)
    context = torch.device('cuda')
    g, disease_vertices, mirna_vertices, ID, IM, samples = build_graph(directory, random_seed)
    samples_df = pd.DataFrame(samples, columns=['miRNA', 'disease', 'label'])
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
    patience = 200
    i = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    for train_idx, test_idx in kf.split(samples[:, 2]):
        best_val_auc = 0.0
        i += 1
        print('Training for Fold', i)
        samples_df['train'] = 0
        samples_df['train'].iloc[train_idx] = 1
        train_tensor = torch.from_numpy(samples_df['train'].values.astype('int64'))
        edge_data = {'train': train_tensor}
        g.edges[disease_vertices, mirna_vertices].data.update(edge_data)
        g.edges[mirna_vertices, disease_vertices].data.update(edge_data)
        train_eid = g.filter_edges(lambda edges: edges.data['train'])
        g_train = g.edge_subgraph(train_eid, relabel_nodes=False)
        label_train = g_train.edata['label'].unsqueeze(1)
        src_train, dst_train = g_train.all_edges()

        test_eid = g.filter_edges(lambda edges: edges.data['train'] == 0)
        src_test, dst_test = g.find_edges(test_eid)
        label_test = g.edges[test_eid].data['label'].unsqueeze(1)
        print('## Training edges:', len(train_eid))
        print('## Testing edges:', len(test_eid))

        model = PREDICTOR(G_train=g_train,
                          hid_dim=in_size,
                          n_class=n_classes,
                          batchnorm=True,
                          num_diseases=ID.shape[0],
                          num_mirnas=IM.shape[0],
                          out_dim=out_dim,
                          dropout=dropout)
        model.apply(weight_reset)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        model_relearn = PREDICTOR_RELEARN(G_train=g_train,
                                          hid_dim=in_size,
                                          n_class=n_classes,
                                          batchnorm=True,
                                          num_diseases=ID.shape[0],
                                          num_mirnas=IM.shape[0],
                                          out_dim=out_dim,
                                          dropout=dropout)
        model_relearn.apply(weight_reset)
        optimizer_relearn = torch.optim.Adam(model_relearn.parameters(), lr=lr, weight_decay=wd)

        lambda_reg = 0.001
        cul_loss = CustomLossWithRegularization(lambda_reg)
        patience_counter = 0
        best_model_state = {}

        last_epochs_pred_matrix = []

        for epoch in range(200):
            model.train()
            with torch.autograd.set_detect_anomaly(True):
                score_train_pre = model(src_train, dst_train, True)
                loss_train_pre = cul_loss(score_train_pre, label_train, model)

                pred_label_tensor = (score_train_pre.view(-1) >= 0.5).long()
                last_epochs_pred_matrix.append(pred_label_tensor.unsqueeze(1))

                optimizer.zero_grad()
                loss_train_pre.backward()
                optimizer.step()

                print('PreTraining... :', epoch + 1)

        last_epochs_pred_matrix = torch.cat(last_epochs_pred_matrix, dim=1)

        N = last_epochs_pred_matrix.shape[0]
        half = N // 2

        first_half = last_epochs_pred_matrix[:half]
        second_half = last_epochs_pred_matrix[half:]

        unstable_index = detect_change(first_half, second_half)
        unstable_index.sort()

        src_unstable = torch.cat([src_train[unstable_index], dst_train[unstable_index]], dim=0)
        dst_unstable = torch.cat([dst_train[unstable_index], src_train[unstable_index]], dim=0)

        label_relearn_half = torch.from_numpy(samples_df.loc[unstable_index, 'label'].values.astype('int64')).unsqueeze(1)
        label_relearn = torch.cat([label_relearn_half, label_relearn_half], dim=0).float().to(context)

        log_var_pretrain = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))
        log_var_relearn = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))

        for epoch in range(epochs):
            start = time.time()

            model.train()
            with torch.autograd.set_detect_anomaly(True):
                score_train = model(src_train, dst_train, True)
                loss_train = cul_loss(score_train, label_train, model)

                loss_relearn = relearn(model_relearn, optimizer_relearn, src_unstable, dst_unstable, label_relearn,
                                       cul_loss)

                loss_P = (1.0 / (2.0 * torch.exp(log_var_pretrain))) * loss_train + 0.5 * log_var_pretrain
                loss_S = (1.0 / (2.0 * torch.exp(log_var_relearn))) * loss_relearn + 0.5 * log_var_relearn

                loss = loss_P + loss_S

                optimizer.zero_grad()
                optimizer.step()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                score_val = model(src_test, dst_test, False)
                loss_val = cul_loss(score_val, label_test, model)

            score_train_cpu = np.squeeze(score_train.cpu().detach().numpy())
            score_val_cpu = np.squeeze(score_val.cpu().detach().numpy())
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
            if patience_counter >= patience:
                break

            pred_val = [0 if j < 0.5 else 1 for j in score_val_cpu]
            acc_val = metrics.accuracy_score(label_val_cpu, pred_val)
            pre_val = metrics.precision_score(label_val_cpu, pred_val)
            recall_val = metrics.recall_score(label_val_cpu, pred_val)
            f1_val = metrics.f1_score(label_val_cpu, pred_val)

            end = time.time()
            if (epoch + 1) % 10 == 0:
                print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss.item(),
                      'Val Loss: %.4f' % loss_val.cpu().detach().numpy(),
                      'Acc: %.4f' % acc_val, 'Pre: %.4f' % pre_val, 'Recall: %.4f' % recall_val, 'F1: %.4f' % f1_val,
                      'Train AUC: %.4f' % train_auc, 'Val AUC: %.4f' % val_auc, 'Time: %.2f' % (end - start))
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            model.load_state_dict(best_model_state)
            score_test = model(src_test, dst_test, False)
        score_test_cpu = np.squeeze(score_test.cpu().detach().numpy())
        label_test_cpu = np.squeeze(label_test.cpu().detach().numpy())

        fpr, tpr, thresholds = metrics.roc_curve(label_test_cpu, score_test_cpu)
        precision, recall, _ = metrics.precision_recall_curve(label_test_cpu, score_test_cpu)
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

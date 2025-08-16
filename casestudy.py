import time
import warnings
import numpy as np
import pandas as pd
import dgl
import random
import torch
import torch.nn as nn

from MainModel import PREDICTOR
from ReLearnModel import PREDICTOR_RELEARN
from TRY import detect_change, relearn
from cul_loss import CustomLossWithRegularization

case_study_directory = 'case_study_data'
case_study_result_directory = 'case_study_result'
directory = 'data'
epochs = 1000
n_classes = 3
in_size = 64
out_dim = 64
dropout = 0.0
slope = 0.2
lr = 0.001
wd = 5e-3
random_seed = 1235
cuda = True

warnings.filterwarnings("ignore")
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    if not cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed(random_seed)

context = torch.device('cpu')

toVerifyDiseases = pd.read_csv(case_study_directory + '/toVerifyDiseases1.csv').values
miRNA_name = pd.read_csv(case_study_directory + '/miRNA_name.csv', names=['miRNA'])
dbDEMC = pd.read_csv(case_study_directory + '/dbDEMC.csv')
miR2Disease = pd.read_csv(case_study_directory + '/miR2Disease.csv')

disease_number = toVerifyDiseases[:, -1]
disease_name = toVerifyDiseases[:, 2]
toverifydisease_dict = dict(zip(disease_number, disease_name))

statistics = pd.DataFrame(toverifydisease_dict.values(), columns=['disease_name'])

index_number = 0


def sample(directory, disease_number, random_seed):
    all_associations = pd.read_csv(directory + '/all_mirna_disease_pairs.csv', names=['miRNA', 'disease', 'label'])
    known_associations = all_associations.loc[all_associations['label'] == 1]
    unknown_associations = all_associations.loc[all_associations['label'] == 0]

    unknown_associations_sp_diseases = unknown_associations.loc[unknown_associations['disease'] != disease_number]
    random_negative_sp_disease = unknown_associations_sp_diseases.sample(n=known_associations.shape[0],
                                                                         random_state=random_seed, axis=0)

    test_associations = unknown_associations.loc[unknown_associations['disease'] == disease_number]

    sample_df = known_associations.append(random_negative_sp_disease.append(test_associations))
    sample_df.reset_index(drop=True, inplace=True)

    return sample_df.values, test_associations.values


def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()


def load_data(directory):
    D_SSM1 = np.loadtxt(directory + '/D_SSM1.txt')
    D_SSM2 = np.loadtxt(directory + '/D_SSM2.txt')
    D_GSM = np.loadtxt(directory + '/D_GSM.txt')
    M_FSM = np.loadtxt(directory + '/M_FSM.txt')
    M_GSM = np.loadtxt(directory + '/M_GSM.txt')

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
    return ID, IM


def build_graph(directory, disease_number, random_seed):
    ID, IM = load_data(directory)
    samples, test_samples = sample(directory, disease_number, random_seed)

    g = dgl.DGLGraph()
    g.add_nodes(ID.shape[0] + IM.shape[0])
    node_type = torch.zeros(g.number_of_nodes(), dtype=torch.int64)
    node_type[: ID.shape[0]] = 1
    g.ndata['type'] = node_type

    d_sim = torch.zeros(g.number_of_nodes(), ID.shape[1])
    d_sim[: ID.shape[0], :] = torch.from_numpy(ID.astype('float32'))
    g.ndata['d_sim'] = d_sim

    m_sim = torch.zeros(g.number_of_nodes(), IM.shape[1])
    m_sim[ID.shape[0]: ID.shape[0] + IM.shape[0], :] = torch.from_numpy(IM.astype('float32'))
    g.ndata['m_sim'] = m_sim

    disease_ids = list(range(1, ID.shape[0] + 1))
    mirna_ids = list(range(1, IM.shape[0] + 1))

    disease_ids_invmap = {id_: i for i, id_ in enumerate(disease_ids)}
    mirna_ids_invmap = {id_: i for i, id_ in enumerate(mirna_ids)}

    sample_disease_vertices = [disease_ids_invmap[id_] for id_ in samples[:, 1]]
    sample_mirna_vertices = [mirna_ids_invmap[id_] + ID.shape[0] for id_ in samples[:, 0]]

    g.add_edges(sample_disease_vertices, sample_mirna_vertices,
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})
    g.add_edges(sample_mirna_vertices, sample_disease_vertices,
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})
    g.readonly()

    return g, sample_disease_vertices, sample_mirna_vertices, samples, test_samples, ID, IM


for d_number in disease_number:
    g, disease_vertices, mirna_vertices, samples, test_samples, ID, IM = build_graph(directory, d_number, random_seed)
    g.to(context)

    sample_df = pd.DataFrame(samples, columns=['miRNA', 'disease', 'label'])
    sample_df['train'] = 0
    sample_df['train'][:sample_df.shape[0] - test_samples.shape[0]] = 1
    train_tensor = torch.from_numpy(sample_df['train'].values.astype('int64'))

    edge_data = {'train': train_tensor}

    g.edges[disease_vertices, mirna_vertices].data.update(edge_data)
    g.edges[mirna_vertices, disease_vertices].data.update(edge_data)

    train_eid = g.filter_edges(lambda edges: edges.data['train'])
    g_train = g.edge_subgraph(train_eid, relabel_nodes=False)

    g_train.copy_from_parent()

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
    model.to(context)
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
    model_relearn = model_relearn.to(context)
    optimizer_relearn = torch.optim.Adam(model_relearn.parameters(), lr=lr, weight_decay=wd)

    lambda_reg = 0.001
    calculateloss = CustomLossWithRegularization(lambda_reg)

    last_epochs_pred_matrix = []
    for epoch in range(200):
        model.train()
        with torch.autograd.set_detect_anomaly(True):
            score_train_pre = model(src_train, dst_train, True)
            loss_train_pre = calculateloss(score_train_pre, label_train, model)

            pred_label_tensor = (score_train_pre.view(-1) >= 0.5).long()
            last_epochs_pred_matrix.append(pred_label_tensor.unsqueeze(1))

            optimizer.zero_grad()
            loss_train_pre.backward()
            optimizer.step()

            print('PreTraining... :', epoch + 1, '/200')

    last_epochs_pred_matrix = torch.cat(last_epochs_pred_matrix, dim=1)

    N = last_epochs_pred_matrix.shape[0]
    half = N // 2

    first_half = last_epochs_pred_matrix[:half]
    second_half = last_epochs_pred_matrix[half:]

    unstable_index = detect_change(first_half, second_half)
    unstable_index.sort()

    src_unstable = torch.cat([src_train[unstable_index], dst_train[unstable_index]], dim=0)
    dst_unstable = torch.cat([dst_train[unstable_index], src_train[unstable_index]], dim=0)

    label_relearn_half = torch.from_numpy(sample_df.loc[unstable_index, 'label'].values.astype('int64')).unsqueeze(1)
    label_relearn = torch.cat([label_relearn_half, label_relearn_half], dim=0).float().to(context)

    log_var_pretrain = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))
    log_var_relearn = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))

    for epoch in range(epochs):
        start = time.time()

        model.train()
        with torch.autograd.set_detect_anomaly(True):
            score_train = model(src_train, dst_train, True)
            loss_train = calculateloss(score_train, label_train, model)

            loss_relearn = relearn(model_relearn, optimizer_relearn, src_unstable, dst_unstable, label_relearn,
                                   calculateloss)

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
            loss_val = calculateloss(score_val, label_test, model)

        end = time.time()
        print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss.item(), 'Time: %.2f' % (end - start))

    model.eval()
    with torch.no_grad():
        score_test = model(src_test, dst_test, False)

    score_test_cpu = np.squeeze(score_test.cpu().detach().numpy())
    score_test = score_test_cpu[:test_samples.shape[0]]

    test_samples_df = pd.DataFrame(test_samples, columns=['miRNA', 'disease', 'label'])
    test_samples_df['score'] = score_test

    candidate_number = test_samples_df['miRNA'].values
    candidate_miRNA = []
    for i in candidate_number:
        candidate_miRNA.append(miRNA_name.values[i-1])

    test_samples_df['miRNA_name'] = candidate_miRNA

    results = test_samples_df.sort_values(by='score', ascending=False)
    results.reset_index(drop=True, inplace=True)

    candidate_related_mirnas = results['miRNA_name']
    to_confirmed_dbDEMC = dbDEMC.loc[dbDEMC['disease'] == toverifydisease_dict[d_number]]
    to_confirmed_miR2Disease = miR2Disease.loc[miR2Disease['disease'] == toverifydisease_dict[d_number]]

    evidence = []
    for mirna in candidate_related_mirnas:
        record = []
        if mirna in to_confirmed_dbDEMC['miRNA'].values and mirna in to_confirmed_miR2Disease['miRNA'].values:
            record = ['dbDEMC and miR2Disease']
            evidence.append(record)
        elif mirna in to_confirmed_dbDEMC['miRNA'].values:
            record = ['dbDEMC']
            evidence.append(record)
        elif mirna in to_confirmed_miR2Disease['miRNA'].values:
            record = ['miR2Disease']
            evidence.append(record)
        else:
            record = ['Unconfirmed']
            evidence.append(record)

    results['evidence'] = evidence

    case_study_result = results[['miRNA_name', 'evidence']]

    case_study_result.to_csv(case_study_result_directory + '/%s.csv' % toverifydisease_dict[d_number], index=False)

    for top_num in [10, 20, 30, 40, 50]:
        statistics.loc[index_number, '%d' % top_num] = top_num - case_study_result['evidence'].head(
            top_num).tolist().count(['Unconfirmed'])

    index_number += 1

statistics.to_csv(case_study_result_directory + '/statistics.sv', index=False)

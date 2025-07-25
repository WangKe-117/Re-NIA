import torch as th
import torch.nn as nn
import torch
import torch.nn.functional as F

from Method import method


class PREDICTOR(nn.Module):
    def __init__(self, G_train, hid_dim, n_class, batchnorm, num_diseases, num_mirnas, out_dim, dropout):
        super().__init__()
        self.G_train = G_train
        self.hid_dim = hid_dim
        self.n_class = n_class
        self.num_diseases = num_diseases
        self.num_mirnas = num_mirnas
        self.disease_nodes = G_train.filter_nodes(lambda nodes: nodes.data['type'] == 1)
        self.mirna_nodes = G_train.filter_nodes(lambda nodes: nodes.data['type'] == 0)
        self.m_fc = nn.Linear(G_train.ndata['m_sim'].shape[1], hid_dim, bias=False)
        self.d_fc = nn.Linear(G_train.ndata['d_sim'].shape[1], hid_dim, bias=False)
        self.m_fc1 = nn.Linear(out_dim + n_class, out_dim)
        self.d_fc1 = nn.Linear(out_dim + n_class, out_dim)
        self.mlp = MLP(hid_dim, out_dim, n_class, 0.0, 0.0, batchnorm)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout = dropout
        self.PROCESS_CAGNN = method(64, 64, 64)
        self.predict = nn.Linear(out_dim * 2, 1)
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(64, 64))
        self.act_fn = nn.ReLU()

    def forward(self, diseases, mirnas, training):
        self.G_train.apply_nodes(lambda nodes: {'z': self.dropout1(self.d_fc(nodes.data['d_sim']))}, self.disease_nodes)
        self.G_train.apply_nodes(lambda nodes: {'z': self.dropout1(self.m_fc(nodes.data['m_sim']))}, self.mirna_nodes)
        feats = self.G_train.ndata.pop('z')
        x = feats

        diseases1 = diseases.unsqueeze(0)
        mirnas1 = mirnas.unsqueeze(0)
        edge_index = torch.cat((diseases1, mirnas1), dim=0)

        x = self.PROCESS_CAGNN(x, edge_index)
        x = F.dropout(x, self.dropout, training=training)
        feat0 = th.log_softmax(self.mlp(x), dim=-1)

        h_d = th.cat((feat0[:self.num_diseases], feats[:self.num_diseases]), dim=1)
        h_m = th.cat((feat0[self.num_diseases:], feats[self.num_diseases:]), dim=1)
        h_m = self.dropout1(F.elu(self.m_fc1(h_m)))
        h_d = self.dropout1(F.elu(self.d_fc1(h_d)))
        h = th.cat((h_d, h_m), dim=0)

        h_diseases = h[diseases]

        h_mirnas = h[mirnas]
        h_concat = th.cat((h_diseases, h_mirnas), 1)
        predict_score = th.sigmoid(self.predict(h_concat))

        return predict_score


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, use_bn=False):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.input_dropout = nn.Dropout(input_droprate)
        self.hidden_dropout = nn.Dropout(hidden_droprate)
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn

    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()

    def forward(self, x):
        if self.use_bn:
            x = self.bn1(x)
        x = self.input_dropout(x)
        x = F.relu(self.layer1(x))

        if self.use_bn:
            x = self.bn2(x)
        x = self.hidden_dropout(x)
        x = self.layer2(x)

        return x

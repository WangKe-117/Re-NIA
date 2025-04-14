import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_sparse import SparseTensor, fill_diag
from torch.nn import Module, ModuleList, Linear, LayerNorm


class ONGNN_method(Module):
    def __init__(self, in_channel, hidden_channel, out_channel, dropout):
        super().__init__()
        self.num_layers = 2
        self.chunk_size = 64
        self.dropout = dropout
        self.linear_trans_in = Linear(in_channel, hidden_channel)
        self.linear_trans_out = Linear(hidden_channel, out_channel)
        self.norm_input = LayerNorm(hidden_channel)
        self.convs = ModuleList()

        self.tm_norm = ModuleList()
        self.tm_net = ModuleList()

        # for i in range(self.num_layers - 1):
        #     self.linear_trans_in.append(Linear(hidden_channel, hidden_channel))
        #     self.norm_input.append(LayerNorm(hidden_channel))

        for i in range(self.num_layers):
            self.tm_norm.append(LayerNorm(hidden_channel))
            self.tm_net.append(Linear(2 * hidden_channel, self.chunk_size))
            self.convs.append(ONGNNConv(tm_net=self.tm_net[i], tm_norm=self.tm_norm[i], hidden_channel=hidden_channel,
                                        chunk_size=self.chunk_size))

    def forward(self, x, edge_index):
        # check_signal = []

        # for i in range(len(self.linear_trans_in)):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.linear_trans_in(x))
        x = self.norm_input(x)

        tm_signal = x.new_zeros(self.chunk_size)

        for j in range(len(self.convs)):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x, tm_signal = self.convs[j](x, edge_index, last_tm_signal=tm_signal)
            # check_signal.append(dict(zip(['tm_signal'], [tm_signal])))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear_trans_out(x)

        # encode_values = dict(zip(['x', 'check_signal'], [x, check_signal]))

        return x


class ONGNNConv(MessagePassing):
    def __init__(self, tm_net, tm_norm, hidden_channel, chunk_size):
        super(ONGNNConv, self).__init__('mean')
        self.tm_net = tm_net
        self.tm_norm = tm_norm
        self.hidden_channel = hidden_channel
        self.chunk_size = chunk_size

    def forward(self, x, edge_index, last_tm_signal):
        # if isinstance(edge_index, SparseTensor):
        #     edge_index = fill_diag(edge_index, fill_value=0)
        #     if self.params['add_self_loops'] == True:
        #         edge_index = fill_diag(edge_index, fill_value=1)
        # else:
        #     edge_index, _ = remove_self_loops(edge_index)
        #     if self.params['add_self_loops'] == True:
        #         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        m = self.propagate(edge_index, x=x)
        # if self.params['tm'] == True:
        #     if self.params['simple_gating'] == True:
        #         tm_signal_raw = F.sigmoid(self.tm_net(torch.cat((x, m), dim=1)))
        #     else:
        # x = self.DeProp_Prop(x, edge_index)
        tm_signal_raw = F.softmax(self.tm_net(torch.cat((x, m), dim=1)), dim=-1)
        tm_signal_raw = torch.cumsum(tm_signal_raw, dim=-1)
        # if self.params['diff_or'] == True:
        tm_signal_raw = last_tm_signal + (1 - last_tm_signal) * tm_signal_raw
        tm_signal = tm_signal_raw.repeat_interleave(repeats=int(self.hidden_channel / self.chunk_size), dim=1)
        out = x * tm_signal + m * (1 - tm_signal)
        # else:
        #     out = m
        #     tm_signal_raw = last_tm_signal

        out = self.tm_norm(out)

        return out, tm_signal_raw

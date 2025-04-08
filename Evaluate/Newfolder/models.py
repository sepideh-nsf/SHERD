import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from torch.nn.parameter import Parameter
import torch
from torch.nn import Sequential, Linear, ReLU



class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, lr=0.01, drop=False, weight_decay=5e-4, n_edge=1
                 ,with_relu=True,
                 with_bias=True, device=None):
        super(GCN, self).__init__()
        ###############################
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.dropout = dropout
        self.lr = lr
        self.feat_norm=None
        self.adj_norm_func=None
        self.model_type="torch"

        weight_decay = 0  # set weight_decay as 0

        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.n_edge = n_edge
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.gate = Parameter(torch.rand(1))  # creat a generator between [0,1]
        self.test_value = Parameter(torch.rand(1))
        self.drop_learn_1 = Linear(2, 1)
        self.drop_learn_2 = Linear(2, 1)
        self.drop = drop
        self.bn1 = torch.nn.BatchNorm1d(nhid)
        self.bn2 = torch.nn.BatchNorm1d(nhid)
        nclass = int(nclass)
        ######################################

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):#,RobustnessEpoch):
        x = F.relu(self.gc1(x, adj))
        f1=x
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        f2=x
        return [F.log_softmax(x, dim=1),f1,f2]

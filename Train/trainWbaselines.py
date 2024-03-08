from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import accuracy
from models import GCN
from utils import load_data, accuracy

import random
import scipy as sp
from utils import normalize as nor
from utils import sparse_mx_to_torch_sparse_tensor
from scipy import sparse
from scipy.stats import norm
from scipy.spatial.distance import hamming
from dgl.data import CoraGraphDataset,CiteseerGraphDataset,PubmedGraphDataset
# from dgl.data import citation_graph as citegrh
import networkx as nx
from dgl import DGLGraph
from torch.nn.functional import normalize
# from attack import getScore, getScoreGreedy, getThrehold, getIndex
from attack import getScore, getScoreGreedy, getThrehold, getIndex, getM, New_sort, New_sort_erf, New_sort_sumtest, New_sort_erf_testsum
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
from networkx.algorithms.centrality import betweenness_centrality as betweenness
from copy import deepcopy
import os
import pandas as pd
from deeprobust.graph.targeted_attack import Nettack
import matplotlib as mpl
import matplotlib.pyplot as plt
import collections
import json
from placenta_dataset import Placenta
from pathlib import Path
from fgsm import FGSM
from rand import RAND
from PGD import PGD
from tdgia import TDGIA
from speit import SPEIT
#########################################
class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:
                self.early_stop = True

#########################################

mpl.use('TkAgg')

# x = np.arange(10)
# y = np.array([0.61, 0.65, 0.55, 0.67, 0.56, 0.60, .52, .50, .52, .52])
# plt.scatter(x, y, color='red')
# x = np.arange(10)
# y = np.array([0.68, 0.68, 0.575, 0.71, 0.60, 0.64, .61, .59, .63, .62])
# plt.scatter(x, y, color='blue')
# plt.xlabel("Attack methods")
# plt.ylabel("Performance")
# plt.title("Adversarial Performance of Original and Compressed Input")
# plt.show()



# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#borrowed from "Towards More Practical Adversarial Attacks on Graph Neural Networks"
def split_data(data, NumTrain, NumTest, NumVal,size):
    idx_test = np.random.choice(size, NumTest, replace=False)
    without_test = np.array([i for i in range(size) if i not in idx_test])
    idx_train = without_test[np.random.choice(len(without_test),
                                              NumTrain,
                                              replace=False)]
    idx_val = np.array([
        i for i in range(size) if i not in idx_test if i not in idx_train
    ])
    idx_val = idx_val[np.random.choice(len(idx_val), NumVal, replace=False)]
    return idx_train, idx_val, idx_test

Dataset="cora"
# Load data
if Dataset=="cora":
    data=CoraGraphDataset()
    graph = data[0]
    graph.ndata['feat'] = torch.FloatTensor(graph.ndata['feat'])
    graph.ndata['label'] = torch.LongTensor(graph.ndata['label'])
    # size=data.labels.shape[0]
    size = len(graph.ndata['label'])
features = graph.ndata['feat']
# features=data.features#graph.ndata['feat']
# l0=np.zeros((np.array(data.labels,dtype=int).size, np.array(data.labels,dtype=int).max() + 1))#one hot encode
# l0[np.arange(np.array(data.labels,dtype=int).size),np.array(data.labels,dtype=int)]=1#one hot encode
labels = torch.LongTensor(graph.ndata['label'])  # data.labels)#
adj=torch.FloatTensor(nx.adjacency_matrix(nx.Graph(data[0].to_networkx())).toarray())

if Dataset=="citeseer":
    data=CiteseerGraphDataset()
    graph = data[0]
    graph.ndata['feat'] = torch.FloatTensor(graph.ndata['feat'])
    graph.ndata['label'] = torch.LongTensor(graph.ndata['label'])
    size = len(graph.ndata['label'])
    features = graph.ndata['feat']
    labels = torch.LongTensor(graph.ndata['label'])
    adj=torch.FloatTensor(nx.adjacency_matrix(nx.Graph(data[0].to_networkx())).toarray())

if Dataset=="pubmed":
    data=PubmedGraphDataset()
    graph = data[0]
    graph.ndata['feat'] = torch.FloatTensor(graph.ndata['feat'])
    graph.ndata['label'] = torch.LongTensor(graph.ndata['label'])
    size = len(graph.ndata['label'])
    features = graph.ndata['feat']
    labels = torch.LongTensor(graph.ndata['label'])
    adj=torch.FloatTensor(nx.adjacency_matrix(nx.Graph(data[0].to_networkx())).toarray())

if Dataset=="placenta":
    project_dir = Path(__file__).absolute().parent.parent
    dataset = Placenta(project_dir / "datasets")
    data = dataset[0]
    features=torch.FloatTensor(data.x)
    labels = torch.LongTensor(data.y)
    adj=torch.FloatTensor(nx.adjacency_matrix(nx.Graph(data)).toarray())
    size=len(labels)
NumTrain = int(size * 0.1)#0.6
NumTest = int(size * 0.8)#0.2
NumVal = int(size * 0.1)#0.2
idx_train, idx_val, idx_test = split_data(data, NumTrain, NumTest, NumVal,size)
# idx_train=np.arange(8)
# idx_test=np.arange(9,73)
# idx_val=np.arange(74,82)
#Some data in Citeseer are problematic and need to be removed (some nodes' features are all zeros)
idx_train=np.setdiff1d(idx_train, np.where(~np.array(features).any(axis=1))[0])
idx_val=np.setdiff1d(idx_val, np.where(~np.array(features).any(axis=1))[0])
idx_test=np.setdiff1d(idx_test, np.where(~np.array(features).any(axis=1))[0])

with open('OrigIdx_train.npy','wb') as f1:
    np.save(f1,idx_train);
with open('OrigIdx_test.npy','wb') as f2:
    np.save(f2,idx_test);
with open('OrigIdx_val.npy','wb') as f3:
    np.save(f3,idx_val);

# idx_test=torch.flatten(torch.LongTensor(np.where(data.test_mask==True)))
# idx_train=torch.flatten(torch.LongTensor(np.where(data.train_mask==True)))
# idx_val=torch.flatten(torch.LongTensor(np.where(data.val_mask==True)))


# NumTrain = int(size * args.train)
# NumTest = int(size * args.test)
# NumVal = int(size * args.validation)
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
# idx_train, idx_val, idx_test = split_data(data, NumTrain, NumTest, NumVal)

# path="../data/cora/"
# edgelist = pd.read_csv(os.path.join(path, "cora.cites"), sep='\t', header=None, names=["target", "source"])
# edgelist["label"] = "cites"
# nxg = nx.from_pandas_edgelist(edgelist, edge_attr="label")
# nx.set_node_attributes(nxg, "paper", "label")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nclass=labels.max().item()+1,
            dropout=args.dropout,
            # n_edge=adj.nonzero()[0].shape[0],
            nhid=args.hidden,
            with_relu=False,
            with_bias=False,
            device=device,
           )
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
surrogate = model.to(device)
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(model,optimizer,epoch):#,Tau):
    # RobustnessEpoch=False
    # if epoch==Tau:
    #     RobustnessEpoch=True
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output,H1,H2= model(features, adj)#,RobustnessEpoch)
    # output=output[0]
    # H=output[1]
    # if epoch%Tau==0:
    #     loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    # else:
    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    Yhat_Ys=np.array([int(output[i].argmax(-1).numpy()== labels[i].numpy()) for i in idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output,temp1,temp2 = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return H1,H2,loss_train.item(),acc_train.item(),loss_val.item(),acc_val.item(),Yhat_Ys

def test(model):
    model.eval()
    output  = model(features, adj)[0]
    loss_test= F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return loss_test.item(),acc_test.item()



def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)

def pick_feature(grad, k):
    score = grad.sum(dim=0)
    # Dataset="cora"
    with open('fixed_sign_{}_new_high_train_40.json'.format(Dataset), 'r') as f:
        sign_fix = json.load(f)
    indexs = np.array(list(map(int, list(sign_fix.keys()))))
    print(indexs)
    signs = torch.zeros(data.features.shape[1])
    for i in sign_fix.keys():
        signs[int(i)] = sign_fix[i]
    return signs, indexs


def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features,adj)[0]
        logits = logits[mask]
        _, indices = torch.max(logits, dim=1)
        labels = data[0].ndata['label'][mask]
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def grad_attack(model,num_node,norm_length):
    #attack: Towards More Practical Adversarial Attacks on Graph Neural Networks
    alpha=0.01
    norm_length=1 #Variable lambda in the original paper
    steps=4 #steps of random walk
    threshold=0.1 #Threshold percentage of degree
    # num_node=33 #Number of target nodes
    beta=30 #Variable l in the paper
    size = data.labels.shape[0]
    nxg = nx.Graph(data[0].to_networkx())
    page = pagerank(nxg)
    between = betweenness(nxg)
    PAGERANK = sorted([(page[i], i) for i in range(len(nxg.nodes))], reverse=True)
    BETWEEN = sorted([(between[i], i) for i in range(len(nxg.nodes))], reverse=True)
    Important_score = getScore(steps, data)
    Important_list = sorted([(Important_score[i], i) for i in range(size)],
                            reverse=True)

    # Important_list = sorted([(Important_score[i], i) for i in range(data.size)],
    #                         reverse=True)
    bar, Baseline_Degree, Baseline_Random = getThrehold(data._g, size,
                                                        threshold,
                                                        num_node)

    Important_matrix = getM(steps, data)
    RWCS_NEW = New_sort(alpha, Important_matrix.numpy(), num_node, bar, data._g)
    RWCS_NEW_ERF = New_sort_erf(0.01, Important_matrix.numpy(), num_node, bar, data._g)
    RWCS_NEW_TESTSUM = New_sort_sumtest(0.01, Important_matrix.numpy(), num_node, bar, data._g, idx_test)
    RWCS_NEW_ERF_TESTSUM = New_sort_erf_testsum(0.01, Important_matrix.numpy(), num_node, bar, data._g, idx_test)

    Baseline_Pagerank = getIndex(data._g, PAGERANK, bar, num_node)
    Baseline_Between = getIndex(data._g, BETWEEN, bar, num_node)
    RWCS = getIndex(data._g, Important_list, bar, num_node)
    GC_RWCS = getScoreGreedy(steps, data, bar, num_node, beta)

    num_features=74
    data.features.requires_grad_(True)
    model.eval()
    logits = model(features,adj)[0]
    # logits = model(data)
    loss = F.nll_loss(logits[idx_train], data[0].ndata['label'][idx_train])
    optimizer.zero_grad()
    zero_gradients(data.features)

    loss.backward(retain_graph=True)
    grad = data.features.grad.detach().clone()
    signs, indexs = pick_feature(grad, num_features)
    data.features.requires_grad_(False)
    result = torch.zeros(11, 2)
    # result[0][0] = evaluate(model, data, idx_test)
    # result[-1,0] = evaluate(model, data, idx_train)
    # model.eval()
    result[-1,0] = evaluate(model, data, idx_test)
    model.eval()
    # with torch.no_grad():
    #     trainlogits = model(features,adj)[0][idx_train]
    #     result[-1,1] = F.nll_loss(trainlogits, data[0].ndata['label'][idx_train])
    with torch.no_grad():
        testlogits = model(features,adj)[0][idx_test]
        result[-1,1] = F.nll_loss(testlogits, data[0].ndata['label'][idx_test])
    TrainingAccs=torch.zeros(11,50)
    for i, targets in enumerate([
            Baseline_Degree, Baseline_Pagerank, Baseline_Between,
            Baseline_Random, GC_RWCS, RWCS, RWCS_NEW, RWCS_NEW_ERF, RWCS_NEW_TESTSUM, RWCS_NEW_ERF_TESTSUM
    ]):
        for target in targets:
            for index in indexs:
                data.features[target][index] += norm_length * signs[index]
        # Model and optimizer
        Evalmodel = GCN(nfeat=features.shape[1],
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout,
                    # n_edge=adj.nonzero()[0].shape[0],
                    nhid=args.hidden,
                    with_relu=False,
                    with_bias=False,
                    device=device,
                    )
        Evaloptimizer = optim.Adam(Evalmodel.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
        Evalmodel.eval()
        for epoch in range(50):
            TrainingAccs[i][epoch]= train(Evalmodel,Evaloptimizer,epoch)[3]########
        result[i][0] = evaluate(Evalmodel, data, idx_test)
        # result[0][i][0] = evaluate(model, data, idx_train)
        Evalmodel.eval()
        with torch.no_grad():
            # trainlogits = model(features,adj)[0][idx_train]
            # result[0][i][1] = F.nll_loss(trainlogits, data[0].ndata['label'][idx_train])
            testlogits = Evalmodel(features,adj)[0][idx_test]
            result[i][1] = F.nll_loss(testlogits, data[0].ndata['label'][idx_test])
        for target in targets:
            for index in indexs:
                data.features[target][index] -= norm_length * signs[index];

    # Model and optimizer for none
    Evalmodel = GCN(nfeat=features.shape[1],
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout,
                    # n_edge=adj.nonzero()[0].shape[0],
                    nhid=args.hidden,
                    with_relu=False,
                    with_bias=False,
                    device=device,
                    )
    Evaloptimizer = optim.Adam(Evalmodel.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
    Evalmodel.eval()
    for epoch in range(50):
        TrainingAccs[-1][epoch]= train(Evalmodel,Evaloptimizer,epoch)[3]########

    result[-1,0] = evaluate(Evalmodel, data, idx_test)
    Evalmodel.eval()
    with torch.no_grad():
        logits = Evalmodel(features,adj)[0][idx_test]
        result[-1,1] = F.nll_loss(logits, data[0].ndata['label'][idx_test])
    return result,TrainingAccs

AccConvergenceVis=False
def adj_to_tensor(adj):
    r"""

    Description
    -----------
    Convert adjacency matrix in scipy sparse format to torch sparse tensor.

    Parameters
    ----------
    adj : scipy.sparse.csr.csr_matrix
        Adjacency matrix in form of ``N * N`` sparse matrix.
    Returns
    -------
    adj_tensor : torch.Tensor
        Adjacency matrix in form of ``N * N`` sparse tensor.

    """

    if type(adj) != sp.sparse.coo.coo_matrix:
        adj = adj.tocoo()
    sparse_row = torch.LongTensor(adj.row).unsqueeze(1)
    sparse_col = torch.LongTensor(adj.col).unsqueeze(1)
    sparse_concat = torch.cat((sparse_row, sparse_col), 1)
    sparse_data = torch.FloatTensor(adj.data)
    adj_tensor = torch.sparse.FloatTensor(sparse_concat.t(), sparse_data, torch.Size(adj.shape))

    return adj_tensor

def adj_preprocess(adj, adj_norm_func=None, mask=None, model_type="torch", device='cpu'):
    r"""

    Description
    -----------
    Preprocess the adjacency matrix.

    Parameters
    ----------
    adj : scipy.sparse.csr.csr_matrix or a tuple
        Adjacency matrix in form of ``N * N`` sparse matrix.
    adj_norm_func : func of utils.normalize, optional
        Function that normalizes adjacency matrix. Default: ``None``.
    mask : torch.Tensor, optional
        Mask of nodes in form of ``N * 1`` torch bool tensor. Default: ``None``.
    model_type : str, optional
        Type of model's backend, choose from ["torch", "cogdl", "dgl"]. Default: ``"torch"``.
    device : str, optional
        Device used to host data. Default: ``cpu``.

    Returns
    -------
    adj : torch.Tensor or a tuple
        Adjacency matrix in form of ``N * N`` sparse tensor or a tuple.

    """

    if adj_norm_func is not None:
        adj = adj_norm_func(adj)
    if model_type == "torch":
        if type(adj) is tuple or type(adj) is list:
            if mask is not None:
                adj = [adj_to_tensor(adj_[mask][:, mask]).to(device)
                       if type(adj_) != torch.Tensor else adj_[mask][:, mask].to(device)
                       for adj_ in adj]
            else:
                adj = [adj_to_tensor(adj_).to(device)
                       if type(adj_) != torch.Tensor else adj_.to(device)
                       for adj_ in adj]
        else:
            if type(adj) != torch.Tensor:
                if mask is not None:
                    adj = adj_to_tensor(adj[mask][:, mask]).to(device)
                else:
                    adj = adj_to_tensor(adj).to(device)
            else:
                if mask is not None:
                    adj = adj[mask][:, mask].to(device)
                else:
                    adj = adj.to(device)
    elif model_type == "dgl":
        if type(adj) is tuple:
            if mask is not None:
                adj = [adj_[mask][:, mask] for adj_ in adj]
            else:
                adj = [adj_ for adj_ in adj]
        else:
            if mask is not None:
                adj = adj[mask][:, mask]
            else:
                adj = adj
    return adj

# Evaluation=True
def Evaluation():#attack input to compare it with the non-attacked
    fgsm=True
    pgd=True
    rand=True
    tdgia=True
    speit=False
    global adj
    global features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model and optimizer for the attack method
    Evalmodel = GCN(nfeat=features.shape[1],
                nclass=labels.max().item() + 1,
                dropout=args.dropout,
                # n_edge=adj.nonzero()[0].shape[0],
                nhid=args.hidden,
                with_relu=False,
                with_bias=False,
                device=device,
                )
    Evaloptimizer = optim.Adam(Evalmodel.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    Evalmodel.eval()
    early_stopping = EarlyStopping(tolerance=25, min_delta=0.01)
    for epoch in range(args.epochs):
        H = train(Evalmodel,Evaloptimizer,epoch)########
        early_stopping(H[2],H[4])#train_loss, validation_loss
        if early_stopping.early_stop:
            args.epochs=epoch
            print("We are at epoch:", epoch)
            break

    AdjCopy=deepcopy(adj)
    FeatCopy=deepcopy(features)
    VanillaTestAcc=evaluate(Evalmodel, data, idx_test)
    if fgsm:
        attack = FGSM(epsilon=0.01,
                      n_epoch=10,
                      n_inject_max=100,
                      n_edge_max=20,
                      feat_lim_min=-1,
                      feat_lim_max=1,
                      device=device)
        mask=torch.zeros(size, dtype=bool)
        mask[idx_test]=1
        adj_attack, features_attack = attack.attack(model=model,
                                                         adj=adj,
                                                         features=features,
                                                        feat_norm=None,
                                                         target_mask=mask)#,
                                                         # adj_norm_func=model.adj_norm_func)
        adj= adj_preprocess(adj=adj_attack,
                                         # adj_norm_func=model.adj_norm_func,
                                         model_type=model.model_type,
                                         device=device)
        features= torch.cat([features, features_attack])
        norm_length=1 #Variable lambda in the original paper

        fgsmTestAcc=evaluate(Evalmodel, data, idx_test)
        adj=AdjCopy
        features=FeatCopy
    if pgd:
        attack = PGD(epsilon=0.01,
                     n_epoch=50,
                     n_inject_max=13,
                     n_edge_max=30,
                     feat_lim_min=-1,
                     feat_lim_max=1,
                     device=device)
        mask=torch.zeros(size, dtype=bool)
        mask[idx_test]=1
        adj_attack, features_attack = attack.attack(model=model,
                                                         adj=adj,
                                                         features=features,
                                                        feat_norm=None,
                                                         target_mask=mask)#,
                                                         # adj_norm_func=model.adj_norm_func)
        adj= adj_preprocess(adj=adj_attack,
                                         # adj_norm_func=model.adj_norm_func,
                                         model_type=model.model_type,
                                         device=device)
        features= torch.cat([features, features_attack])
        norm_length=1 #Variable lambda in the original paper

        PGDTestAcc=evaluate(Evalmodel, data, idx_test)
        adj=AdjCopy
        features=FeatCopy
    if rand:
        attack = RAND(n_inject_max=13,
                      n_edge_max=30,
                      feat_lim_min=-1,
                      feat_lim_max=1,
                      device=device)
        mask=torch.zeros(size, dtype=bool)
        mask[idx_test]=1
        adj_attack, features_attack = attack.attack(model=model,
                                                         adj=adj,
                                                         features=features,
                                                        # feat_norm=None,
                                                         target_mask=mask,
                                                         adj_norm_func=model.adj_norm_func)
        adj= adj_preprocess(adj=adj_attack,
                                         # adj_norm_func=model.adj_norm_func,
                                         model_type=model.model_type,
                                         device=device)
        features= torch.cat([features, features_attack])
        norm_length=1 #Variable lambda in the original paper

        RandTestAcc=evaluate(Evalmodel, data, idx_test)
        adj=AdjCopy
        features=FeatCopy
    if speit:
        attack = SPEIT(lr=0.01,
                       n_epoch=50,
                       n_inject_max=13,
                       n_edge_max=30,
                       feat_lim_min=-1,
                       feat_lim_max=1,
                       device=device)
        mask=torch.zeros(size, dtype=bool)
        mask[idx_test]=1
        adj_attack, features_attack = attack.attack(model=model,
                                                         adj=adj,
                                                         features=features,
                                                        # feat_norm=None,
                                                         target_mask=mask,
                                                         adj_norm_func=model.adj_norm_func)
        adj= adj_preprocess(adj=adj_attack,
                                         # adj_norm_func=model.adj_norm_func,
                                         model_type=model.model_type,
                                         device=device)
        features= torch.cat([features, features_attack])
        norm_length=1 #Variable lambda in the original paper

        SPEITTestAcc=evaluate(Evalmodel, data, idx_test)
        adj=AdjCopy
        features=FeatCopy
    # if tdgia:
    #     attack = TDGIA(lr=0.01,
    #                    n_epoch=50,
    #                    n_inject_max=13,
    #                    n_edge_max=30,
    #                    feat_lim_min=-1,
    #                    feat_lim_max=1,
    #                    device=device)
    #     mask=torch.zeros(size, dtype=bool)
    #     mask[idx_test]=1
    #     adj_attack, features_attack = attack.attack(model=model,
    #                                                      adj=adj,
    #                                                      features=features,
    #                                                     # feat_norm=None,
    #                                                      target_mask=mask,
    #                                                      adj_norm_func=model.adj_norm_func)
    #     adj= adj_preprocess(adj=adj_attack,
    #                                      # adj_norm_func=model.adj_norm_func,
    #                                      model_type=model.model_type,
    #                                      device=device)
    #     features= torch.cat([features, features_attack])
    #     norm_length=1 #Variable lambda in the original paper
    #
    #     TDGIATestAcc=evaluate(Evalmodel, data, idx_test)
    #     adj=AdjCopy
    #     features=FeatCopy
    VanillaTestAcc2 = evaluate(Evalmodel, data, idx_test)
    pass
Evaluation()
    # # grad_attack(Attackoptimizer,norm_length)
    # result, TrainigAccs = grad_attack(Attackmodel,num_node,norm_length)
    # if AccConvergenceVis:
    #     Labels = [
    #         "Base_Degree", "Base_Pagerank", "Base_Between",
    #         "Base_Random", "GC-RWCS", "RWCS", "InfMax-Unif", "InfMax-Norm", "New_sort_sumtest_Unif",
    #         "New_sort_sumtest_erf", "None"
    #     ];
    #     # cmap = plt.colormaps["Set3"];
    #     x = np.arange(50);
    #     fig, ax = plt.subplots();
    #     colors=['firebrick','orangered','darkgoldenrod','olive','limegreen','lightseagreen','dodgerblue','mediumblue','navy','purple','mediumvioletred']
    #     for i in range(11):
    #         y = np.array([it.item() for it in TrainigAccs[i]]);
    #         ax.plot(x, y, color=colors[i], label=Labels[i]);
    #     plt.xlabel('Epoch', fontdict=dict(weight='bold'), fontsize=12);
    #     plt.ylabel('Adversarial Data Accuracy', fontdict=dict(weight='bold'), fontsize=12);
    #     ax.legend();
    #     plt.tight_layout();
    #     plt.savefig("training"+mode+"Eval.png");
    # # plt.show();
    #
    #
    # # # Model and optimizer
    # # Evalmodel = GCN(nfeat=features.shape[1],
    # #             nclass=labels.max().item() + 1,
    # #             dropout=args.dropout,
    # #             # n_edge=adj.nonzero()[0].shape[0],
    # #             nhid=args.hidden,
    # #             with_relu=False,
    # #             with_bias=False,
    # #             device=device,
    # #             )
    # # Evaloptimizer = optim.Adam(Evalmodel.parameters(),
    # #                        lr=args.lr, weight_decay=args.weight_decay)
    # # Evalmodel.eval()
    # # for epoch in range(args.epochs):
    # #     H = train(Evalmodel,Evaloptimizer,epoch)########
    # # data_backup = deepcopy(data)
    # #
    # # for index, method in enumerate([
    # #     "Baseline_Degree", "Baseline_Pagerank", "Baseline_Between",
    # #     "Baseline_Random", "GC-RWCS", "RWCS", "InfMax-Unif", "InfMax-Norm", "New_sort_sumtest_Unif",
    # #     "New_sort_sumtest_erf", "None"
    # # ]):
    # #     print("{} : Train Accuracy : {:.4f}, Train Loss : {:.4f}".format(
    # #         method, result[0][index][0].item(), result[0][index][1].item()))
    #
    # for index, method in enumerate([
    #     "Baseline_Degree", "Baseline_Pagerank", "Baseline_Between",
    #     "Baseline_Random", "GC-RWCS", "RWCS", "InfMax-Unif", "InfMax-Norm", "New_sort_sumtest_Unif",
    #     "New_sort_sumtest_erf", "None"
    # ]):
    #     print("{} : Test Accuracy : {:.4f}, Test Loss : {:.4f}".format(
    #         method, result[index][0].item(), result[index][1].item()))
    #
    # # test(Evalmodel)
    # return result,TrainigAccs[:,-1]



# # Model and optimizer for the vanilla original input
# Vanilmodel = GCN(nfeat=features.shape[1],
#             nclass=labels.max().item()+1,
#             dropout=args.dropout,
#             # n_edge=adj.nonzero()[0].shape[0],
#             nhid=args.hidden,
#             with_relu=False,
#             with_bias=False,
#             device=device,
#            )
# Vaniloptimizer = optim.Adam(Vanilmodel.parameters(),
#                        lr=args.lr, weight_decay=args.weight_decay)
#
# for epoch in range(args.epochs):
#     H = train(Vanilmodel,Vaniloptimizer,epoch)  # ,Tau)
# # Testing
# test(Vanilmodel)


#Whole graph attacked input
#######################################
OrigGraphResult = Evaluation()#"Orig", 33)
# OrigGraphResult =torch.stack((OrigGraphResult,Evaluation("Orig", 33)));
OrigGraphResultTest=OrigGraphResult[0]
OrigGraphResultTrain=OrigGraphResult[1]
#accs only:
# OrigMaxsTest=OrigGraphResultTest[:,0];
# OrigMinsTest=OrigGraphResultTest[:,0];
# OrigMaxsTrain=OrigGraphResultTrain;
# OrigMinsTrain=OrigGraphResultTrain;
# finding the min and max to show variance in the figure

for i in range(4):
    OrigGraphResult=OrigGraphResult+Evaluation("Orig", 33)
    # OrigGraphResult=torch.cat((OrigGraphResult,[Evaluation("Orig", 33)]))
OrigMaxsTest=np.max(torch.stack(list(OrigGraphResult[::2]), dim=0).numpy()[:,:,0],axis=0);
OrigMinsTest=np.min(torch.stack(list(OrigGraphResult[::2]), dim=0).numpy()[:,:,0],axis=0);
OrigMeanTest=np.mean(torch.stack(list(OrigGraphResult[::2]), dim=0).numpy()[:,:,0],axis=0);
OrigMaxsTrain=np.max(torch.stack(list(OrigGraphResult[1::2]), dim=0).numpy(),axis=0);
OrigMinsTrain=np.min(torch.stack(list(OrigGraphResult[1::2]), dim=0).numpy(),axis=0);
OrigMeanTrain=np.mean(torch.stack(list(OrigGraphResult[1::2]), dim=0).numpy(),axis=0);
##########################################

x=[
        "Base_Degree", "Base_Pagerank", "Base_Between",
        "Base_Random", "GC-RWCS", "RWCS", "InfMax-Unif", "InfMax-Norm", "New_sort_sumtest_Unif",
        "New_sort_sumtest_erf", "None"
    ];
y=OrigMeanTest;
fig,ax=plt.subplots();
plt.subplots_adjust(bottom=0.3);
plt.xlabel('Node Selection for Attack Method', fontdict=dict(weight='bold'), fontsize=12);
plt.ylabel('Adversarial Test Data Accuracy', fontdict=dict(weight='bold'), fontsize=12);
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right');
df=pd.DataFrame({'Attack':x,'Accuracy':y,'Min':np.array(OrigMinsTest),'Max':np.array(OrigMaxsTest)});

df['ymin'] = df.Accuracy - df.Min;
df['ymax'] = df.Max - df.Accuracy;

yerr = df[['ymin', 'ymax']].T.to_numpy();
ax.errorbar(x=x,y=y,yerr=yerr,color='red',capsize=5,label="Original Input");
ax.legend();
plt.show()
####################################
# ax.plot(x,y,color='red',label="Original Input");
# plt.xlabel('Node Selection for Attack Method', fontdict=dict(weight='bold'), fontsize=12);
# plt.ylabel('Adversarial Test Data Accuracy', fontdict=dict(weight='bold'), fontsize=12);
# ax.legend();
# plt.rcParams['errorbar.capsize'] = 10;
########################
# df=pd.DataFrame({'Attack':x,'Accuracy':y,'Min':np.array(OrigMinsTest),'Max':np.array(OrigMaxsTest)});
#
# df['ymin'] = df.Accuracy - df.Min;
# df['ymax'] = df.Max - df.Accuracy;
#
# yerr = df[['ymin', 'ymax']].T.to_numpy();
# ax.errorbar(x=x,y=y,yerr=yerr);
########################
# sns.barplot(x='Attack', y='Accuracy',data=df, yerr=yerr, ax=ax);
# plt.tight_layout();
# plt.show();

# plt.vlines(x, OrigMinsTest, OrigMaxsTest, color='k');

# + dy * np.random.randn(50);
# x = np.linspace(0, 10, 50);
# dy = 0.1;
# y = np.sin(x);
#
# plt.errorbar(x, y, yerr=dy, fmt='.k',capsize=5);
# plt.show();
# Train model
StandardAttack=False
RandomAttack=False
DumbAttack=False
nettack=True
Plot=False

Tau=30

trainAcc=[]
testAcc=[]
trainLoss=[]
testLoss=[]
valAcc=[]
valLoss=[]
Yhat_Ys=[]
# OrigtrainAcc=[]
# OrigtestAcc=[]
# OrigtrainLoss=[]
# OrigtestLoss=[]
# OrigvalAcc=[]
# OrigvalLoss=[]
for epoch in range(args.epochs):
    out = train(model,optimizer,epoch)  # ,Tau)
    H1=out[0]
    H2=out[1]
    trainLoss.append(out[2])
    trainAcc.append(out[3])
    valLoss.append(out[4])
    valAcc.append(out[5])
    if (epoch>Tau-6) and (epoch<Tau):
        Yhat_Ys.append(out[6])
    if (nettack==True) and (epoch==Tau) and (epoch!=0):
        Yhat_Ys=np.mean(np.array(Yhat_Ys), axis=0)
        Dists1 = []
        Dists2 = []
        #To evaluate our method and compare original and compressed input results:
        best_model_state = deepcopy(model.state_dict())
        # OrigInputModel = GCN(nfeat=features.shape[1],
        #                  nclass=labels.max().item() + 1,
        #                  dropout=args.dropout,
        #                  # n_edge=adj.nonzero()[0].shape[0],
        #                  nhid=args.hidden,
        #                  with_relu=False,
        #                  with_bias=False,
        #                  device=device,
        #                  )
        # OrigInputModel.load_state_dict(best_model_state)
        # OrigInputModel.eval()
        # OrigInputoptimizer = optim.Adam(OrigInputModel.parameters(),
        #                              lr=args.lr, weight_decay=args.weight_decay)
        # OrigInputModel.eval()
        # for epoch2 in range(epoch,args.epochs):
        #     OrigOut = train(OrigInputModel, OrigInputoptimizer, epoch2)  # ,Tau)
        #     OrigtrainLoss.append(OrigOut[2])
        #     OrigtrainAcc.append(OrigOut[3])
        #     OrigvalLoss.append(OrigOut[4])
        #     OrigvalAcc.append(OrigOut[5])
        #     Origtout = test(OrigInputModel)
        #     OrigtestLoss.append(Origtout[0])
        #     OrigtestAcc.append(Origtout[1])

        #Vectorized
        # degrees = [item[item > 0].shape[0] for item in np.array(adj.to_dense())]
        # Att_models = [Nettack(model, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device=device) for i in idx_train]
        # Att_models=[item.to(device) for item in Att_models]
        # n_perturbations = [int(degrees[i]) for i in idx_train]
        # [item.attack(sparse.csr_matrix(np.array(features.to_dense())), sparse.csr_matrix(np.array(adj.to_dense())),
        #                  labels, int(i), n_perturbation, direct=True) for item,i,n_perturbation in zip(Att_models,idx_train,n_perturbations)]
        # modified_adjs = [item.modified_adj for item in Att_models]
        # modified_features = [item.modified_features for item in Att_models]
        # best_models = [GCN(nfeat=features.shape[1],
        #                  nclass=labels.max().item() + 1,
        #                  dropout=args.dropout,
        #                  # n_edge=adj.nonzero()[0].shape[0],
        #                  nhid=args.hidden,
        #                  with_relu=False,
        #                  with_bias=False,
        #                  device=device,
        #                  ) for i in idx_train]
        # [item.load_state_dict(best_model_state) for item in best_models]
        # [item.eval() for item in best_models]
        # modified_features = [nor(item) for item in modified_features]
        # modified_features = [torch.FloatTensor(np.array(item.todense())) for item in modified_features]
        # modified_adjs = [nor(item + sp.eye(adj.shape[0])) for item in modified_adjs]
        # modified_adjs = [sparse_mx_to_torch_sparse_tensor(sparse.csr_matrix(item)) for item in modified_adjs]
        # _, H_P1s, H_P2s = [best_model(modified_feature, modified_adj) for best_model,modified_feature, modified_adj in zip(best_models,modified_features, modified_adjs)]
        # Dists1 = [np.mean([hamming(H1[:, j].detach().numpy(), H_P1[:, j].detach().numpy()) for j in range(H1.shape[1])]) for H_P1 in H_P1s]
        # Dists2 = [np.mean([hamming(H2[:, j].detach().numpy(), H_P2[:, j].detach().numpy()) for j in range(H2.shape[1])]) for H_P2 in H_P2s]

        for i in idx_train:# Vectorize here:
            if i==2781:
                pass
            Att_model=Nettack(model, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device=device)
            Att_model = Att_model.to(device)
            degrees=[item[item>0].shape[0] for item in np.array(adj.to_dense())]
            # NxAdj=nx.from_numpy_array(np.array(adj.to_dense()))
            # degrees = NxAdj.sum(0).A1
            # How many perturbations to perform. Default: Degree of the node
            n_perturbations = int(degrees[i])

            # # indirect attack/ influencer attack

            Att_model.attack(sparse.csr_matrix(np.array(features.to_dense())), sparse.csr_matrix(np.array(adj.to_dense())), labels, int(i), n_perturbations, direct=True)
            # Att_model.attack()
            modified_adj = Att_model.modified_adj
            modified_features = Att_model.modified_features

            best_model = GCN(nfeat=features.shape[1],
                        nclass=labels.max().item()+1,
                        dropout=args.dropout,
                        # n_edge=adj.nonzero()[0].shape[0],
                        nhid=args.hidden,
                        with_relu=False,
                        with_bias=False,
                        device=device,
                       )
            best_model.load_state_dict(best_model_state)
            best_model.eval()
            modified_features = nor(modified_features)
            modified_features = torch.FloatTensor(np.array(modified_features.todense()))
            modified_adj = nor(modified_adj + sp.eye(adj.shape[0]))
            modified_adj = sparse_mx_to_torch_sparse_tensor(sparse.csr_matrix(modified_adj))

            temp, H_P1,H_P2 = best_model(modified_features, modified_adj)
            Distance1 = [hamming(H1[:,j].detach().numpy(), H_P1[:,j].detach().numpy()) for j in range(H1.shape[1])]
            Distance1 = np.mean(Distance1)

            Dists1.append(
                Distance1)  ######TO do: find most 300 elements of Dists and remove them. then attack using the paper

            Distance2 = [hamming(H2[:,j].detach().numpy(), H_P2[:,j].detach().numpy()) for j in range(H2.shape[1])]
            Distance2 = np.mean(Distance2)

            Dists2.append(
                Distance2)  ######TO do: find most 300 elements of Dists and remove them. then attack using the paper

        if Plot==True:
            #
            # indxs=np.array([np.argpartition(np.array(Dists)[:, z], -10)[-10:] for z in range(np.array(Dists).shape[1])])
            # TopDists=np.array([np.array(Dists)[indxs[z],z] for z in range(indxs.shape[0])])
            # ind=np.delete(np.array(range(TopDists.shape[0])),np.where(np.max(TopDists,axis=0)==0))
            # ind=np.delete(np.array(range(indxs.shape[0])),np.where(np.max(TopDists,axis=0)==0))
            # TopDists=TopDists[ind]

            # x=np.arange(TopDists.shape[0])
            #
            x=np.arrange(10)
            y=np.array([0.61,0.65,0.55,0.67,0.56,0.60,.52,.50,.52,.52])
            plt.scatter(x,y,edgecolors='red')
            x=np.arrange(10)
            y=np.array([0.61,0.65,0.55,0.67,0.56,0.60,.52,.50,.52,.52])
            plt.scatter(x,y,edgecolors='blue')
            plt.show()
            # x=np.array(range(TopDists.shape[0]))#['Layer 1','Layer 2','Layer 3','Layer 4','Layer 5','Layer 6','Layer 7','Layer 8','Layer 9','Layer 10','Layer 11','Layer 12','Layer 13','Layer 14','Layer 15','Layer 16']
            #
            # y=[(TopDists[:,u].T) for u in range(10)]
            #
            # u=0
            # # plt.plot([xe] * ye.shape[0], ye)
            # # plt.scatter([xe] * ye.shape[0], ye)  # ,facecolors='none', edgecolors='r')
            v = 0
            Topindxs=[]
            for xe, ye in zip(x, np.array(y).T):
                plt.scatter([xe] * ye.shape[0], ye)
            plt.show()

        #     for i, txt in enumerate(indxs[v]):
        #         Topindxs.append(txt)
        #         # plt.annotate(txt, (xe, ye[i]))
        #     v += 1
        #
        #     best_model_state = deepcopy(model.state_dict())
        #     for xe,indx in zip(x,indxs):
        #         best_model = GCN(nfeat=features.shape[1],
        #                     nclass=labels.max().item()+1,
        #                     dropout=args.dropout,
        #                     nhid=args.hidden,
        #                     with_relu=False,
        #                     with_bias=False,
        #                     device=device,
        #                    )
        #         best_model.load_state_dict(best_model_state)
        #         NeuriIndxs=np.delete(np.array(range(features.shape[0])),indx)
        #         best_model.eval()
        #         output,rep=best_model(features[NeuriIndxs],torch.from_numpy(np.array(adj.to_dense())[NeuriIndxs,:][:,NeuriIndxs]))#adj to numpy, indxs applied, to tensor
        #         TrainNeuriIndxs=np.delete(np.array(idx_train),indx)
        #         acc_val = accuracy(output[TrainNeuriIndxs], labels[TrainNeuriIndxs])
        #         pass
        #         plt.scatter(xe, acc_val)
        #     plt.show()
        #
        # # features=np.delete(features,np.argpartition(np.array(Dists), -10)[-10:],0)
        # # adj=np.delete(np.delete(adj.to_dense(),np.argpartition(np.array(Dists), -10)[-10:],0),np.argpartition(np.array(Dists), -10)[-10:],1)
        # labels=np.delete(labels,np.argpartition(np.array(Dists), -10)[-10:],0)
        Dists=(np.array(Dists1)+np.array(Dists2))/2
        # idx_train=random.choices(idx_train, k=int(len(idx_train) * 0.01))
        # idx_test=random.choices(idx_test, k=int(len(idx_test) * 0.01))
        # idx_val=random.choices(idx_val, k=int(len(idx_val) * 0.01))
        alpha=0.8
        # Dists=(Dists - np.min(Dists)) / (np.max(Dists) - np.min(Dists))
        # m=alpha*Dists+(1-alpha)*Yhat_Ys
        m=Dists
        idx_train=np.setdiff1d(idx_train, np.intersect1d(idx_train, np.argpartition(np.array(m), -int(len(idx_train)*0.1))[-int(len(idx_train)*0.1):]))#######OPTIONAL
        idx_test=np.setdiff1d(idx_test, np.intersect1d(idx_train, np.argpartition(np.array(m), -int(len(idx_test)*0.1))[-int(len(idx_test)*0.1):]))#######OPTIONAL
        idx_val=np.setdiff1d(idx_val, np.intersect1d(idx_train, np.argpartition(np.array(m), -int(len(idx_val)*0.1))[-int(len(idx_val)*0.1):]))#######OPTIONAL
        print("\nIndx train:\n"+str(idx_train)+"\n\n")
        with open('CompIdx_train.npy', 'wb') as f1:
            np.save(f1,idx_train)
        with open('CompIdx_test.npy', 'wb') as f2:
            np.save(f2,idx_test)
        with open('CompIdx_val.npy', 'wb') as f3:
            np.save(f3,idx_val)

    pass
    print("Optimization Finished!")
    # print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    #Testing when subgraph is not attacked
    tout=test(model)
    testLoss.append(tout[0])
    testAcc.append(tout[1])



#######################################
OrigGraphResult = Evaluation("Orig", 100)#33
# OrigGraphResult =torch.stack((OrigGraphResult,Evaluation("Orig", 33)));
OrigGraphResultTest=OrigGraphResult[0]
OrigGraphResultTrain=OrigGraphResult[1]
#accs only:
# OrigMaxsTest=OrigGraphResultTest[:,0];
# OrigMinsTest=OrigGraphResultTest[:,0];
# OrigMaxsTrain=OrigGraphResultTrain;
# OrigMinsTrain=OrigGraphResultTrain;
# finding the min and max to show variance in the figure

for i in range(4):
    OrigGraphResult=OrigGraphResult+Evaluation("Orig", 100)
    # OrigGraphResult=torch.cat((OrigGraphResult,[Evaluation("Orig", 33)]))
OrigMaxsTest=np.max(torch.stack(list(OrigGraphResult[::2]), dim=0).numpy()[:,:,0],axis=0);
OrigMinsTest=np.min(torch.stack(list(OrigGraphResult[::2]), dim=0).numpy()[:,:,0],axis=0);
OrigMeanTest=np.mean(torch.stack(list(OrigGraphResult[::2]), dim=0).numpy()[:,:,0],axis=0);
OrigMaxsTrain=np.max(torch.stack(list(OrigGraphResult[1::2]), dim=0).numpy(),axis=0);
OrigMinsTrain=np.min(torch.stack(list(OrigGraphResult[1::2]), dim=0).numpy(),axis=0);
OrigMeanTrain=np.mean(torch.stack(list(OrigGraphResult[1::2]), dim=0).numpy(),axis=0);
###########################################
#Subgraph evaluation

SubGraphResult = Evaluation("Sub", 100)
SubGraphResultTest=SubGraphResult[0]
SubGraphResultTrain=SubGraphResult[1]

# finding the min and max to show variance in the figure

for i in range(4):
    SubGraphResult=SubGraphResult+Evaluation("Sub", 100)
SubMaxsTest=np.max(torch.stack(list(SubGraphResult[::2]), dim=0).numpy()[:,:,0],axis=0);
SubMinsTest=np.min(torch.stack(list(SubGraphResult[::2]), dim=0).numpy()[:,:,0],axis=0);
SubMeanTest=np.mean(torch.stack(list(SubGraphResult[::2]), dim=0).numpy()[:,:,0],axis=0);
SubMaxsTrain=np.max(torch.stack(list(SubGraphResult[1::2]), dim=0).numpy(),axis=0);
SubMinsTrain=np.min(torch.stack(list(SubGraphResult[1::2]), dim=0).numpy(),axis=0);
SubMeanTrain=np.mean(torch.stack(list(SubGraphResult[1::2]), dim=0).numpy(),axis=0);
##########################################
fig,ax=plt.subplots();
plt.subplots_adjust(bottom=0.3);
plt.xlabel('Node Selection for Attack Method', fontdict=dict(weight='bold'), fontsize=12);
plt.ylabel('Adversarial Test Data Accuracy', fontdict=dict(weight='bold'), fontsize=12);
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right');
x=[
        "Base_Degree", "Base_Pagerank", "Base_Between",
        "Base_Random", "GC-RWCS", "RWCS", "InfMax-Unif", "InfMax-Norm", "New_sort_sumtest_Unif",
        "New_sort_sumtest_erf", "None"
    ];
y=OrigMeanTest;
df=pd.DataFrame({'Attack':x,'Accuracy':y,'Min':np.array(OrigMinsTest),'Max':np.array(OrigMaxsTest)});
df['ymin'] = df.Accuracy - df.Min;
df['ymax'] = df.Max - df.Accuracy;
yerr = df[['ymin', 'ymax']].T.to_numpy();
ax.errorbar(x=x,y=y,yerr=yerr,color='red',capsize=5,label="Original Input");

y=SubMeanTest;
df=pd.DataFrame({'Attack':x,'Accuracy':y,'Min':np.array(SubMinsTest),'Max':np.array(SubMaxsTest)});
df['ymin'] = df.Accuracy - df.Min;
df['ymax'] = df.Max - df.Accuracy;
yerr = df[['ymin', 'ymax']].T.to_numpy();
ax.errorbar(x=x,y=y,yerr=yerr,color='blue',capsize=5,label="Robust Subgraph");
ax.legend();
plt.show()






#comparing running epochs of original and our method training
# OrigtrainAcc=trainAcc[:Tau]+OrigtrainAcc
# OrigtestAcc=testAcc[:Tau]+OrigtestAcc
# OrigtrainLoss=trainLoss[:Tau]+OrigtrainLoss
# OrigtestLoss=testLoss[:Tau]+OrigtestLoss
# OrigvalAcc=valAcc[:Tau]+OrigvalAcc
# OrigvalLoss=valLoss[:Tau]+OrigvalLoss
#
# x = np.arange(args.epochs);
# y=np.array([float(it) for it in OrigtrainAcc]);
# fig,ax=plt.subplots();
# ax.plot(x,y,color='red',label="Train");
# y=np.array([float(it) for it in OrigvalAcc]);
# ax.plot(x,y,color='blue',label="Validation");
# y=np.array([float(it) for it in OrigtestAcc]);
# ax.plot(x,y,color='green',label="Test");
# plt.xlabel('Epoch', fontdict=dict(weight='bold'), fontsize=12);
# plt.ylabel('Accuracy', fontdict=dict(weight='bold'), fontsize=12);
# plt.title('Original Accuracies');
# ax.legend();
# plt.tight_layout();
# plt.show();
#
# x = np.arange(args.epochs);
# y=np.array([float(it) for it in OrigtrainLoss]);
# fig,ax=plt.subplots();
# ax.plot(x,y,color='red',label="Train");
# y=np.array([float(it) for it in OrigvalLoss]);
# ax.plot(x,y,color='blue',label="Validation");
# y=np.array([float(it) for it in OrigtestLoss]);
# ax.plot(x,y,color='green',label="Test");
# plt.xlabel('Epoch', fontdict=dict(weight='bold'), fontsize=12);
# plt.ylabel('Loss', fontdict=dict(weight='bold'), fontsize=12);
# plt.title('Original Losses');
# ax.legend();
# plt.tight_layout();
# plt.show();
#
#
# x = np.arange(args.epochs);
# y=np.array([float(it) for it in trainAcc]);
# fig,ax=plt.subplots();
# ax.plot(x,y,color='red',label="Train");
# y=np.array([float(it) for it in valAcc]);
# ax.plot(x,y,color='blue',label="Validation");
# y=np.array([float(it) for it in testAcc]);
# ax.plot(x,y,color='green',label="Test");
# plt.xlabel('Epoch', fontdict=dict(weight='bold'), fontsize=12);
# plt.ylabel('Accuracy', fontdict=dict(weight='bold'), fontsize=12);
# plt.title('Compressed in Tau=50')
# ax.legend();
# plt.tight_layout();
# plt.show();
#
# x = np.arange(args.epochs);
# y=np.array([float(it) for it in trainLoss]);
# fig,ax=plt.subplots();
# ax.plot(x,y,color='red',label="Train");
# y=np.array([float(it) for it in valLoss]);
# ax.plot(x,y,color='blue',label="Validation");
# y=np.array([float(it) for it in testLoss]);
# ax.plot(x,y,color='green',label="Test");
# plt.xlabel('Epoch', fontdict=dict(weight='bold'), fontsize=12);
# plt.ylabel('Loss', fontdict=dict(weight='bold'), fontsize=12);
# plt.title('Compressed in Tau=50')
# ax.legend();
# plt.tight_layout();
# plt.show();
#
#



# #subgraph attacked input
# SubgraphResult = Evaluation("Orig", 33)
# SubgraphResultTest=SubgraphResult[0]
# SubgraphResultTrain=SubgraphResult[1]
# # for i in range(4):
# #     SubgraphResult = Evaluation("Orig", 33)
# #     SubgraphResultTest = SubgraphResult[0]
# #     SubgraphResultTrain = SubgraphResult[1]
# # SubgraphResultTest=SubgraphResultTest/5
# # SubgraphResultTrain=SubgraphResultTrain/5
#
# #########
# # SubgraphResult=Evaluation("Comp",33)
# # for i in range(4):
# #     SubgraphResult=SubgraphResult+Evaluation("Comp",33)
# # SubgraphResult=SubgraphResult/5
# #########
# #attacked input comparison
# x=[
#         "Base_Degree", "Base_Pagerank", "Base_Between",
#         "Base_Random", "GC-RWCS", "RWCS", "InfMax-Unif", "InfMax-Norm", "New_sort_sumtest_Unif",
#         "New_sort_sumtest_erf", "None"
#     ];
#
# #Evaluation
# y=np.array([it[0].item() for it in OrigGraphResultTest]);
# fig,ax=plt.subplots();
# ax.plot(x,y,color='red',label="Original Input");
# y=np.array([it[0].item() for it in SubgraphResultTest]);
# ax.plot(x,y,color='blue',label="Compressed Input");
# plt.xlabel('Node Selection for Attack Method', fontdict=dict(weight='bold'), fontsize=12);
# plt.ylabel('Adversarial Test Data Accuracy', fontdict=dict(weight='bold'), fontsize=12);
# ax.legend();
# plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right');
# plt.tight_layout();
# # plt.title
# plt.savefig("AdversarialAccTest.png")
# plt.show();
#
# y=np.array([it.item() for it in OrigGraphResultTrain]);
# fig,ax=plt.subplots();
# ax.plot(x,y,color='red',label="Original Input");
# y=np.array([it.item() for it in SubgraphResultTrain]);
# ax.plot(x,y,color='blue',label="Compressed Input");
# plt.xlabel('Node Selection for Attack Method', fontdict=dict(weight='bold'), fontsize=12);
# plt.ylabel('Adversarial Train Data Accuracy', fontdict=dict(weight='bold'), fontsize=12);
# ax.legend();
# plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right');
# plt.tight_layout();
# # plt.title
# plt.savefig("AdversarialAccTrain.png")
# plt.show();
#
# y=np.array([it[1].item() for it in OrigGraphResultTest]);
# fig,ax=plt.subplots();
# ax.plot(x,y,color='red',label="Original Input");
# y=np.array([it[1].item() for it in SubgraphResultTest]);
# ax.plot(x,y,color='blue',label="Compressed Input");
# plt.xlabel('Node Selection for Attack Method', fontdict=dict(weight='bold'), fontsize=12);
# plt.ylabel('Adversarial Test Data Loss', fontdict=dict(weight='bold'), fontsize=12);
# ax.legend();
# plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right');
# plt.tight_layout();
# plt.savefig("AdversarialLoss.png")
# plt.show();

# #Test Evaluation
# y=np.array([it[0].item() for it in OrigGraphResult[1]]);
# fig,ax=plt.subplots();
# ax.plot(x,y,color='red',label="Original Input");
# y=np.array([it[0].item() for it in SubgraphResult[1]]);
# ax.plot(x,y,color='blue',label="Compressed Input");
# plt.xlabel('Node Selection for Attack Method', fontdict=dict(weight='bold'), fontsize=12);
# plt.ylabel('Adversarial Test Data Accuracy', fontdict=dict(weight='bold'), fontsize=12);
# ax.legend();
# plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right');
# plt.tight_layout();
# plt.show();
#
# y=np.array([it[1].item() for it in OrigGraphResult[1]]);
# fig,ax=plt.subplots();
# ax.plot(x,y,color='red',label="Original Input");
# y=np.array([it[1].item() for it in SubgraphResult[1]]);
# ax.plot(x,y,color='blue',label="Compressed Input");
# plt.xlabel('Node Selection for Attack Method', fontdict=dict(weight='bold'), fontsize=12);
# plt.ylabel('Adversarial Test Data Loss', fontdict=dict(weight='bold'), fontsize=12);
# ax.legend();
# plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right');
# plt.tight_layout();
# plt.show();


pass

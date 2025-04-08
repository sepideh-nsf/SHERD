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
import random
# from Evaluation import adj_preprocess
from utils import adj_preprocess

from PGD import PGD

from netrd.distance.deltacon import DeltaCon
from netrd.distance.hamming import Hamming

from netrd.distance.hamming_ipsen_mikhailov import HammingIpsenMikhailov
from netrd.distance.jaccard_distance import JaccardDistance
from netrd.distance.laplacian_spectral_method import LaplacianSpectral
from netrd.distance.netsimile import NetSimile
from netrd.distance.resistance_perturbation import ResistancePerturbation
from sklearn.cluster import KMeans
import math
import tracemalloc

#mpl.use('TkAgg')


# tracemalloc.start()


# fig,ax=plt.subplots(figsize=(15,4.5));
# plt.subplots_adjust(bottom=0.3);
# plt.xlabel('Node Selection for Attack Method', fontdict=dict(weight='bold'), fontsize=12);
# plt.ylabel('Adversarial Test Data Accuracy', fontdict=dict(weight='bold'), fontsize=12);
# plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right');
# x=np.arange(1000)
# for i in range(7):
#     ax.plot(x, i * x, label=f'y={i}xhfskjhgsdfjghish hi i ihi ih ih iijskgfslg why is this happeing here?')
# # ax.plot(x,y)
# ax.legend(bbox_to_anchor=(1, 1),loc='upper left',bbox_transform=ax.transAxes);
#
# x, y = ax.xaxis.get_label().get_position() # position of xlabel
# h, w = ax.bbox.height, ax.bbox.width # height and width of the Axes
#
# leg_pos = [x +  w, y + h] # this needs to be adjusted according to your needs
# # fig.legend(loc="lower center", bbox_to_anchor=leg_pos, bbox_transform=ax.transAxes)
#
# plt.tight_layout()
# # plt.savefig("AdversarialAccTest.png")
# plt.show()

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
parser.add_argument('--hidden', type=int, default=16,
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
EvalFlag=False
Dataset="cora"
# Load data
if Dataset=="cora":
    data=CoraGraphDataset()
    graph = data[0]
    graph.ndata['feat'] = torch.FloatTensor(graph.ndata['feat'])
    graph.ndata['label'] = torch.LongTensor(graph.ndata['label'])
    size = len(graph.ndata['label'])
    features = graph.ndata['feat']

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

NumTrain = int(size * 0.6)#0.6
NumTest = int(size * 0.2)#0.2
NumVal = int(size * 0.2)#0.2
CompPercentage=0.5#Compression percentage 10%
idx_train, idx_val, idx_test = split_data(data, NumTrain, NumTest, NumVal,size)

#Some data in Citeseer are problematic and need to be removed (some nodes' features are all zeros)
idx_train=np.setdiff1d(idx_train, np.where(~np.array(features).any(axis=1))[0])
idx_val=np.setdiff1d(idx_val, np.where(~np.array(features).any(axis=1))[0])
idx_test=np.setdiff1d(idx_test, np.where(~np.array(features).any(axis=1))[0])

NotEvaluation=False
if NotEvaluation:
    with open('OrigIdx_train.npy','wb') as f1:
        np.save(f1,idx_train);
    with open('OrigIdx_test.npy','wb') as f2:
        np.save(f2,idx_test);
    with open('OrigIdx_val.npy','wb') as f3:
        np.save(f3,idx_val);


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nclass=labels.max().item()+1,
            dropout=args.dropout,
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


def train(model,optimizer,epoch,idx_train,idx_val):#,Tau):

    t = time.time()
    model.train()
    optimizer.zero_grad()
    output,H1,H2= model(features, adj)#,RobustnessEpoch)

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
    # print('Epoch: {:04d}'.format(epoch+1),
    #       'loss_train: {:.4f}'.format(loss_train.item()),
    #       'acc_train: {:.4f}'.format(acc_train.item()),
    #       'loss_val: {:.4f}'.format(loss_val.item()),
    #       'acc_val: {:.4f}'.format(acc_val.item()),
    #       'time: {:.4f}s'.format(time.time() - t))
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







AccConvergenceVis=False
# Evaluation=True


def Distance(vanilla,attacked,delconFlag,hammingFlag,jaccardFlag,laplSpectFlag):
    Dist=[]
    if delconFlag:
        delcon = DeltaCon()
        DeltaconDistance = delcon.dist(vanilla, attacked, exact=True, g=None)
        Dist.append(DeltaconDistance)
    if hammingFlag:
        ham = Hamming()
        HammingDistance = ham.dist(vanilla, attacked)
        Dist.append(HammingDistance)
    # if hamIpsMikFlag:#time-consuming
    #     hamIM=HammingIpsenMikhailov()
    #     HammingIMDistance=hamIM.dist(vanilla, attacked, combination_factor=1)
    if jaccardFlag:
        jacc = JaccardDistance()
        JaccDistance = jacc.dist(vanilla, attacked)
        Dist.append(JaccDistance)
    if laplSpectFlag:
        lapSpec = LaplacianSpectral()
        laplSpectDistance = lapSpec.dist(vanilla, attacked, normed=True, kernel='normal', hwhm=0.011775,
                                         measure='jensen-shannon', k=None, which='LM')
        Dist.append(laplSpectDistance)
    return Dist
    pass

Tau=15
def main():
    global features
    global adj
    global idx_train
    global idx_test
    global idx_val
    StandardAttack = False
    RandomAttack = False
    DumbAttack = False
    nettack = False
    Plot = False

    trainAcc=[]
    testAcc=[]
    trainLoss=[]
    testLoss=[]
    valAcc=[]
    valLoss=[]
    Yhat_Ys=[]

    delconFlag=True
    hammingFlag=True
    hamIpsMikFlag=True
    jaccardFlag=True
    laplSpectFlag=True
    netsmileFlag=True
    resistPertFlag=True
    Multi_level=True
    idx_trains = []
    idx_tests = []
    idx_vals = []
    BunchsCount=100
    OurMethodTime=0
    Dists1 = []
    Dists2 = []
    performanceControl=True
    for epoch in range(args.epochs):
        out = train(model,optimizer,epoch,idx_train,idx_val)  # ,Tau)
        H1=out[0]
        H2=out[1]
        trainLoss.append(out[2])
        trainAcc.append(out[3])
        valLoss.append(out[4])
        valAcc.append(out[5])
        if (epoch>Tau-6) and (epoch<Tau):
            Yhat_Ys.append(out[6])
        if (epoch==Tau) and (epoch!=0):
            start = time.time()
            best_model_state = deepcopy(model.state_dict())

            if Multi_level:
                pass
                attack = PGD(epsilon=0.01,
                             n_epoch=50,
                             # n_inject_max=13,
                             # n_edge_max=30,
                             n_node_mod=40,
                             n_edge_mod=80,
                             feat_lim_min=-1,
                             feat_lim_max=1,
                             device=device)

                # Kmean = KMeans(n_clusters=BunchsCount)
                # Kmean.fit(features.detach().numpy())
                index_targets=[]
                idx_backup=np.arange(adj.shape[0])
                for i in np.arange(BunchsCount):
                    # mask = torch.zeros(size, dtype=bool)
                    # mask[i*(int(size/BunchsCount)):(i+1)*(int(size/BunchsCount))] = 1
                    # index_target=np.arange(i*(int(size/BunchsCount)),(i+1)*(int(size/BunchsCount)))
                    # index_target=np.where(Kmean.labels_==i)[0]
                    index_targets.append(random.sample(sorted(idx_backup), int(adj.shape[0] / BunchsCount)) if (len(idx_backup)>int(adj.shape[0] / BunchsCount)) else idx_backup)
                    idx_backup=np.setdiff1d(idx_backup, index_targets[i])
                    adj_attack, modified_features = attack.attack(model=model,
                                                                adj=adj,
                                                                features=features,
                                                                feat_norm=None,
                                                                index_target=np.array(index_targets[i]))  # ,
                    # adj_norm_func=model.adj_norm_func)
                    modified_adj = adj_preprocess(adj=adj_attack,
                                         # adj_norm_func=model.adj_norm_func,
                                         model_type=model.model_type,
                                         device=device)
                    # features = torch.cat([features, features_attack])
                    norm_length = 1  # Variable lambda in the original paper

                    # PGDTestAcc = evaluate(Attackmodel, data, idx_test)
                    # result[11][0] = PGDTestAcc

                    # adj = AdjCopy
                    # features = FeatCopy

                    best_model = GCN(nfeat=features.shape[1],
                                     nclass=labels.max().item() + 1,
                                     dropout=args.dropout,
                                     nhid=args.hidden,
                                     with_relu=False,
                                     with_bias=False,
                                     device=device,
                                     )
                    best_model.load_state_dict(best_model_state)
                    best_model.eval()
                    modified_features = nor(modified_features)
                    modified_features = torch.FloatTensor(modified_features)
                    modified_adj = nor(np.array(modified_adj.to_dense()) + np.eye(adj.shape[0]))
                    modified_adj = sparse_mx_to_torch_sparse_tensor(sparse.csr_matrix(modified_adj))

                    temp, H_P1, H_P2 = best_model(modified_features, modified_adj)
                    attacked = nx.from_numpy_array(np.array(modified_adj.to_dense()))
                    attacked.nodes = H_P1
                    vanilla = nx.from_numpy_array(np.array(adj.to_dense()))
                    vanilla.nodes = H1
                    # Dists1.append(math.sqrt(1/(math.exp(len(index_target)/len(features))))*np.array(Distance(vanilla,attacked,delconFlag,hammingFlag,jaccardFlag,laplSpectFlag)))
                    Dists1.append(np.array(Distance(vanilla,attacked,delconFlag,hammingFlag,jaccardFlag,laplSpectFlag)))
                    if performanceControl:
                        delconFlag = False
                        hammingFlag = False
                        # laplSpectFlag=False
                        AdjCopy = deepcopy(adj)
                        FeatCopy = deepcopy(features)
                        SubIndxs = np.setdiff1d(np.arange(adj.shape[0]), index_targets[i])
                        adj=adj[torch.from_numpy(SubIndxs).long()].T[torch.from_numpy(SubIndxs).long()].T
                        features=features[SubIndxs]
                        temp, H_P1, H_P2 = best_model(features, adj)
                        SubRep = nx.from_numpy_array(np.array(adj.to_dense()))#Subrepresentation
                        SubRep.nodes=H_P1
                        Dists2.append(np.array(Distance(vanilla, SubRep, delconFlag, hammingFlag, jaccardFlag, laplSpectFlag)))
                        # Dists2.append(math.sqrt(1/(math.exp(len(index_target)/len(features))))*np.array(Distance(vanilla, SubRep, delconFlag, hammingFlag, jaccardFlag, laplSpectFlag)))
                        adj = AdjCopy
                        features = FeatCopy
                        delconFlag = True
                        hammingFlag = True
                        # laplSpectFlag = True

                Dists1 = np.array(Dists1).T
                if performanceControl:
                    DistMeas1 = ["DeltaCon", "Hamming", "JaccardDistance", "LaplacianSpectral"]
                    DistMeas2 = ["JaccardDistance","LaplacianSpectral"]

                    Dists2 = np.array(Dists2).T

                    idx_trainBackup=deepcopy(idx_train)
                    idx_valBackup=deepcopy(idx_val)
                    idx_testBackup=deepcopy(idx_test)

                    for item2,PerformDist in zip(Dists2,DistMeas2):
                        for item1,RobustDist in zip(Dists1,DistMeas1):
                            item=(item1-item2)/2
                            # np.where(item==np.argsort(item)[-(int(size/4)):])
                            OrderedIndx=np.array([])
                            # Susceptibles=np.array([])
                            for Bunch in np.argsort(item):
                                OrderedIndx=np.concatenate([OrderedIndx,index_targets[Bunch]])
                                # OrderedIndx=np.concatenate([OrderedIndx,np.where(Kmean.labels_==Bunch)[0]])
                                # Susceptibles=np.concatenate([Susceptibles,np.where(Kmean.labels_==Bunch)[0]])#np.arange(Bunch*(int(size/BunchsCount)),(Bunch+1)*(int(size/BunchsCount)))])
                            idx_train=OrderedIndx[np.where(np.in1d(OrderedIndx, idx_train))[0]][-int(len(idx_train)*CompPercentage):].astype(int)
                            idx_test=OrderedIndx[np.where(np.in1d(OrderedIndx, idx_test))[0]][-int(len(idx_test)*CompPercentage):].astype(int)
                            idx_val=OrderedIndx[np.where(np.in1d(OrderedIndx, idx_val))[0]][-int(len(idx_val)*CompPercentage):].astype(int)



                            with open('SubIdx_train-' + RobustDist + '-' + PerformDist + '.npy',
                                      'wb') as fsub1:
                                np.save(fsub1, idx_train);
                            with open('SubIdx_test-' + RobustDist + '-' + PerformDist + '.npy',
                                      'wb') as fsub2:
                                np.save(fsub2, idx_test);
                            with open('SubIdx_val-' + RobustDist + '-' + PerformDist + '.npy',
                                      'wb') as fsub3:
                                np.save(fsub3, idx_val);
                            idx_train=idx_trainBackup
                            idx_test=idx_testBackup
                            idx_val=idx_valBackup

                            # [np.concatenate([Susceptibles,np.arange(Bunch*(int(size/BunchsCount)),(Bunch+1)*(int(size/BunchsCount)))]) for Bunch in np.argsort(item)[-(int(BunchsCount/4)):]]
                            # idx_trains.append(np.setdiff1d(idx_train,Susceptibles ))
                            # idx_tests.append(np.setdiff1d(idx_test,Susceptibles ))
                            # idx_vals.append(np.setdiff1d(idx_val,Susceptibles ))
                else:
                    for item1 in Dists1:
                        # np.where(item==np.argsort(item)[-(int(size/4)):])
                        Susceptibles = np.array([])
                        for Bunch in np.argsort(item1)[-(int(BunchsCount * CompPercentage)):]:
                            Susceptibles = np.concatenate([Susceptibles, np.arange(Bunch * (int(size / BunchsCount)),
                                                                                   (Bunch + 1) * (
                                                                                       int(size / BunchsCount)))])
                        # [np.concatenate([Susceptibles,np.arange(Bunch*(int(size/BunchsCount)),(Bunch+1)*(int(size/BunchsCount)))]) for Bunch in np.argsort(item)[-(int(BunchsCount/4)):]]
                        idx_trains.append(np.setdiff1d(idx_train, Susceptibles))
                        idx_tests.append(np.setdiff1d(idx_test, Susceptibles))
                        idx_vals.append(np.setdiff1d(idx_val, Susceptibles))
            if nettack==True:
                Yhat_Ys=np.mean(np.array(Yhat_Ys), axis=0)
                Dists1 = []
                Dists2 = []
                #To evaluate our method and compare original and compressed input results:

                for i in np.arange(size):# Vectorize here:
                    Dist=[]

                #Not vectorized
                    if i==2781:
                        pass
                    Att_model=Nettack(model, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device=device)
                    Att_model = Att_model.to(device)
                    degrees=[item[item>0].shape[0] for item in np.array(adj.to_dense())]
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
                    attacked = nx.from_numpy_array(np.array(modified_adj.to_dense()))
                    attacked.nodes = H_P1
                    vanilla = nx.from_numpy_array(np.array(adj.to_dense()))
                    vanilla.nodes = H1

                    Dists1.append(Distance(vanilla,attacked))

                Dists1=np.array(Dists1).T
                itera=0
                for item in Dists1:
                    itera+=1
                    Dists=np.array(item)#+np.array(Dists2))/2
                    alpha=0.8
                    m=Dists
                    with open('sensitivities.npy','wb') as f:
                        np.save(f,np.array(m))
                    mTrain=m[idx_train]
                    mTest=m[idx_test]
                    mVal=m[idx_val]
                    # idx_trains.append(np.setdiff1d(idx_train, idx_train[np.argpartition(np.array(mTrain), -int(len(idx_train)*0.1))[-int(len(idx_train)*0.1):]]))#######OPTIONAL
                    idx_tests.append(np.setdiff1d(idx_test, idx_test[np.argpartition(np.array(mTest), -int(len(idx_test)*0.1))[-int(len(idx_test)*0.1):]]))#######OPTIONAL
                    # idx_vals.append(np.setdiff1d(idx_val, idx_val[np.argpartition(np.array(mVal), -int(len(idx_val)*0.1))[-int(len(idx_val)*0.1):]]))#######OPTIONAL

                    # print("\nIndx train:\n"+str(idx_trains[itera])+"\n\n")
                    # with open('CompIdx_train'+itera+'.npy', 'wb') as f1:
                    #     np.save(f1,idx_trains[itera])
                    # with open('CompIdx_test'+itera+'.npy', 'wb') as f2:
                    #     np.save(f2,idx_tests[itera])
                    # with open('CompIdx_val'+itera+'.npy', 'wb') as f3:
                    #     np.save(f3,idx_vals[itera])
            end = time.time()
            OurMethodTime=end-start
            # break
        pass
        print("Optimization Finished!")

        #Testing when subgraph is not attacked
        # tout=test(model)
        # testLoss.append(tout[0])
        # testAcc.append(tout[1])

if __name__ == '__main__':
    main()
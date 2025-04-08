print("before torch")
import torch
import torch.nn.functional as F
import torch.optim as optim
print("After torch")
from models import GCN
print("after models")

from fgsm import FGSM
from rand import RAND
from PGD import PGD
from dice import DICE
from flip import FLIP
from fga import FGA
from nea import NEA
from stack import STACK
print("After attacks")
import networkx as nx
from attack import getScore, getScoreGreedy, getThrehold, getIndex, getM, New_sort, New_sort_erf, New_sort_sumtest, New_sort_erf_testsum
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
from networkx.algorithms.centrality import betweenness_centrality as betweenness
import collections
import numpy as np
from train import train,args,device,Dataset,labels,json,EarlyStopping,size,CompPercentage,data,features,adj
import matplotlib.pyplot as plt
import random
import pandas as pd
from utils import adj_preprocess
import time
import tracemalloc
print("AFter imports")

import scipy as sp
from copy import deepcopy

def evaluate(model,features,adj, data, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features,adj)[0]
        logits = logits[mask]
        _, indices = torch.max(logits, dim=1)
        labels = data[0].ndata['label'][mask]
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

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



def grad_attack(model,optimizer, num_node, norm_length,idx_test):
    global data
    global features
    global adj
    evasion = True
    poisoning = False
    # attack: Towards More Practical Adversarial Attacks on Graph Neural Networks
    alpha = 0.01
    norm_length = 1  # Variable lambda in the original paper
    steps = 4  # steps of random walk
    threshold = 0.1  # Threshold percentage of degree
    # num_node=33 #Number of target nodes
    beta = 30  # Variable l in the paper
    graph=data[0]
    size=len(graph.ndata['label'])
    # size = data.labels.shape[0]
    nxg = nx.Graph(graph.to_networkx())
    # nxg = nx.Graph(data[0].to_networkx())
    page = pagerank(nxg)
    between = betweenness(nxg)
    PAGERANK = sorted([(page[i], i) for i in range(len(nxg.nodes))], reverse=True)
    BETWEEN = sorted([(between[i], i) for i in range(len(nxg.nodes))], reverse=True)
    Important_score = getScore(steps, data)
    Important_list = sorted([(Important_score[i], i) for i in range(size)],
                            reverse=True)

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

    num_features = 74
    data.features.requires_grad_(True)
    model.eval()
    logits = model(features, adj)[0]
    loss = F.nll_loss(logits[idx_train], data[0].ndata['label'][idx_train])
    optimizer.zero_grad()
    zero_gradients(data.features)

    loss.backward(retain_graph=True)
    grad = data.features.grad.detach().clone()
    signs, indexs = pick_feature(grad, num_features)
    data.features.requires_grad_(False)
    result = torch.zeros(14, 2)  # (11, 2)
    testMemories = torch.zeros(14)
    testtimes = torch.zeros(14)

    # time and memory monitoring
    ############################
    teststart=time.time()
    tracemalloc.start()

    # Vanilla test acc
    result[-1, 0] = evaluate(model,features,adj, data, idx_test)

    # displaying the memory
    testMemories[-1]=tracemalloc.get_traced_memory()[1]

    # stopping the library
    tracemalloc.stop()
    testend=time.time()
    testtimes[-1]=testend-teststart
    #############################
    # result[-1,0] = evaluate(model, data, idx_test)
    model.eval()
    with torch.no_grad():
        testlogits = model(features, adj)[0][idx_test]
        result[-1, 1] = F.nll_loss(testlogits, data[0].ndata['label'][idx_test])
    TrainingAccs = torch.zeros(11, 50)
    for i, targets in enumerate([
        Baseline_Degree, Baseline_Pagerank, Baseline_Between,
        Baseline_Random, GC_RWCS, RWCS, RWCS_NEW, RWCS_NEW_ERF, RWCS_NEW_TESTSUM, RWCS_NEW_ERF_TESTSUM
    ]):
        pass
        targ = targets
        for target in targets:
            temp = data.features[target]

            for index in indexs:
                with torch.no_grad():
                    features[target][index] += norm_length * signs[index]
                # data.features[target][index] += norm_length * signs[index]
        if evasion:
            teststart=time.time()
            tracemalloc.start() 
            result[i, 0] = evaluate(model,features,adj, data, idx_test)
            # displaying the memory
            testMemories[i]=tracemalloc.get_traced_memory()[1]

            # stopping the library
            tracemalloc.stop()
            testend=time.time()
            testtimes[i]=testend-teststart

            model.eval()
            # with torch.no_grad():
            #     trainlogits = model(features,adj)[0][idx_train]
            #     result[-1,1] = F.nll_loss(trainlogits, data[0].ndata['label'][idx_train])
            with torch.no_grad():
                testlogits = model(features, adj)[0][idx_test]
                result[i, 1] = F.nll_loss(testlogits, data[0].ndata['label'][idx_test])

        if poisoning:
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
                TrainingAccs[i][epoch] = train(Evalmodel, Evaloptimizer, epoch)[3]  ########
            result[i][0] = evaluate(Evalmodel,features,adj, data, idx_test)
            # result[0][i][0] = evaluate(model, data, idx_train)
            Evalmodel.eval()
            with torch.no_grad():
                # trainlogits = model(features,adj)[0][idx_train]
                # result[0][i][1] = F.nll_loss(trainlogits, data[0].ndata['label'][idx_train])
                testlogits = Evalmodel(features, adj)[0][idx_test]
                result[i][1] = F.nll_loss(testlogits, data[0].ndata['label'][idx_test])
        for target in targets:
            for index in indexs:
                with torch.no_grad():
                    features[target][index] -= norm_length * signs[index]
                    # data.features[target][index] -= norm_length * signs[index];


    return result, TrainingAccs  ,testMemories,testtimes



def Evaluation(num_node,idx_train,idx_val,idx_test):#mode,Attackmodel,Attackoptimizer,num_node,idx_test):#attack input to compare it with the non-attacked
    # tracemalloc.start()

    global adj
    global features
    norm_length=1 #Variable lambda in the original paper

    # Model and optimizer for the attack method
    Attackmodel = GCN(nfeat=features.shape[1],
                      nclass=labels.max().item() + 1,
                      dropout=args.dropout,
                      # n_edge=adj.nonzero()[0].shape[0],
                      nhid=args.hidden,
                      with_relu=False,
                      with_bias=False,
                      device=device,
                      )
    Attackoptimizer = optim.Adam(Attackmodel.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    Attackmodel.eval()
    early_stopping = EarlyStopping(tolerance=10, min_delta=0.001)
    trainstart=time.time()
    tracemalloc.start()
    for epoch in range(args.epochs):
        H = train(Attackmodel, Attackoptimizer, epoch,idx_train,idx_val)  ########
        early_stopping(H[2], H[4])  # train_loss, validation_loss
        #if early_stopping.early_stop:
        #    args.epochs = epoch
        #    print("We are at epoch:", epoch)
        #    break
    # displaying the memory
    trainMemory=tracemalloc.get_traced_memory()[1]

    # stopping the library
    tracemalloc.stop()
    trainend=time.time()
    traintime=trainend-trainstart
    # data_backup = deepcopy(data)
    AttModelBackup=deepcopy(Attackmodel)
    AttOptimizerBackup=deepcopy(Attackoptimizer)
    #result, TrainigAccs = grad_attack(AttModelBackup,AttOptimizerBackup,num_node,norm_length,idx_test)
    #result, TrainigAccs,testMemories,testtimes = grad_attack(AttModelBackup,AttOptimizerBackup,num_node,norm_length,idx_test)
    result = torch.zeros(9, 2)  # (11, 2)
    testMemories = torch.zeros(9)
    testtimes = torch.zeros(9)

    # time and memory monitoring
    ############################
    teststart = time.time()
    tracemalloc.start()

    # Vanilla test acc
    result[-1, 0] = evaluate(Attackmodel, features, adj, data, idx_test)

    # displaying the memory
    testMemories[-1] = tracemalloc.get_traced_memory()[1]

    # stopping the library
    tracemalloc.stop()
    testend = time.time()
    testtimes[-1] = testend - teststart

    #Injection attacks

    fgsm = True
    pgd = True
    rand = True#
    dice = True
    flip = True
    fga = True#
    nea = True
    stack = True
    AdjCopy=deepcopy(adj)
    FeatCopy=deepcopy(features)
    # tracemalloc.start()

    if dice:
        n_edge_test = len(idx_test);
        n_mod_ratio = 0.6;
        n_edge_mod = int(n_edge_test * n_mod_ratio);
        ratio_delete = 0.6;
        attack = DICE(n_edge_mod, ratio_delete);
        adj_attack = attack.attack(adj, idx_test, labels);
        adj = adj_preprocess(adj=adj_attack,
                             # adj_norm_func=model.adj_norm_func,
                             model_type=Attackmodel.model_type,
                             device=device)
        trainstart = time.time()
        tracemalloc.start()

        DiceTestAcc = evaluate(Attackmodel, features, adj, data, idx_test)

        testMemories[0] = tracemalloc.get_traced_memory()[1]

        # stopping the library
        tracemalloc.stop()
        trainend = time.time()
        testtimes[0] = trainend - trainstart

        result[0][0] = DiceTestAcc

        adj = deepcopy(AdjCopy)
        features = deepcopy(FeatCopy)
    if fga:
        n_edge_test = len(idx_test);
        n_mod_ratio = 0.6;
        n_edge_mod = int(n_edge_test * n_mod_ratio);
        attack = FGA(n_edge_mod, device=device)
        adj_attack = attack.attack(Attackmodel, adj, features, idx_test);
        adj = adj_preprocess(adj=adj_attack,
                             # adj_norm_func=model.adj_norm_func,
                             model_type=Attackmodel.model_type,
                             device=device)
                             
        trainstart = time.time()
        tracemalloc.start()
        
        FgaTestAcc = evaluate(Attackmodel, features, adj, data, idx_test)
        
        testMemories[1] = tracemalloc.get_traced_memory()[1]

        # stopping the library
        tracemalloc.stop()
        trainend = time.time()
        testtimes[1] = trainend - trainstart

        result[1][0]=FgaTestAcc

        adj = deepcopy(AdjCopy)
        features = deepcopy(FeatCopy)
    if nea:
        n_edge_test = len(idx_test);
        n_mod_ratio = 0.6;
        n_edge_mod = int(n_edge_test * n_mod_ratio);
        attack = NEA(n_edge_mod)
        adj_attack = attack.attack(adj, idx_test);
        adj = adj_preprocess(adj=adj_attack,
                             # adj_norm_func=model.adj_norm_func,
                             model_type=Attackmodel.model_type,
                             device=device)
        trainstart = time.time()
        tracemalloc.start()

        NeaTestAcc = evaluate(Attackmodel, features, adj, data, idx_test)

        testMemories[2] = tracemalloc.get_traced_memory()[1]

        # stopping the library
        tracemalloc.stop()
        trainend = time.time()
        testtimes[2] = trainend - trainstart

        result[2][0] = NeaTestAcc

        adj = deepcopy(AdjCopy)
        features = deepcopy(FeatCopy)
    if stack:
        n_edge_test = len(idx_test);
        n_mod_ratio = 0.6;
        n_edge_mod = int(n_edge_test * n_mod_ratio);
        attack = STACK(n_edge_mod);
        adj_attack = attack.attack(adj, idx_test);
        adj = adj_preprocess(adj=adj_attack,
                             # adj_norm_func=model.adj_norm_func,
                             model_type=Attackmodel.model_type,
                             device=device)
        trainstart = time.time()
        tracemalloc.start()

        StackTestAcc = evaluate(Attackmodel, features, adj, data, idx_test)

        testMemories[3] = tracemalloc.get_traced_memory()[1]

        # stopping the library
        tracemalloc.stop()
        trainend = time.time()
        testtimes[3] = trainend - trainstart

        result[3][0] = StackTestAcc

        adj = deepcopy(AdjCopy)
        features = deepcopy(FeatCopy)
    if flip:
        n_edge_test = len(idx_test);
        n_mod_ratio = 0.5;
        n_edge_mod = int(n_edge_test * n_mod_ratio);
        # betweenness flipping
        attack = FLIP(n_edge_mod, flip_type="bet", mode="ascend", device=device)
        adj_attack = attack.attack(adj, idx_test);
        adj = adj_preprocess(adj=adj_attack,
                             # adj_norm_func=model.adj_norm_func,
                             model_type=Attackmodel.model_type,
                             device=device)
        trainstart = time.time()
        tracemalloc.start()
        FlipTestAcc = evaluate(Attackmodel, features, adj, data, idx_test)
        
        testMemories[4] = tracemalloc.get_traced_memory()[1]

        # stopping the library
        tracemalloc.stop()
        trainend = time.time()
        testtimes[4] = trainend - trainstart

        
        result[4][0]=FlipTestAcc

        adj = deepcopy(AdjCopy)
        features = deepcopy(FeatCopy)

    if fgsm:
        attack = FGSM(epsilon=0.01,
                      n_epoch=10,
                      n_inject_max=100,
                      n_edge_max=200,
                      feat_lim_min=-1,
                      feat_lim_max=1,
                      device=device)
        mask = torch.zeros(size, dtype=bool)
        mask[idx_test] = 1
        adj_attack, features_attack = attack.attack(model=Attackmodel,
                                                    adj=adj,
                                                    features=features,
                                                    feat_norm=None,
                                                    target_mask=mask)  # ,
        # adj_norm_func=model.adj_norm_func)
        adj = adj_preprocess(adj=adj_attack,
                             # adj_norm_func=model.adj_norm_func,
                             model_type=Attackmodel.model_type,
                             device=device)
        features = torch.cat([features, features_attack])
        norm_length = 1  # Variable lambda in the original paper
        
        trainstart = time.time()
        tracemalloc.start()

        fgsmTestAcc = evaluate(Attackmodel, features, adj, data, idx_test)
        
        testMemories[5] = tracemalloc.get_traced_memory()[1]

        # stopping the library
        tracemalloc.stop()
        trainend = time.time()
        testtimes[5] = trainend - trainstart

        result[5][0]=fgsmTestAcc
        adj = deepcopy(AdjCopy)
        features = deepcopy(FeatCopy)
    if pgd:
        attack = PGD(epsilon=0.01,
                     n_epoch=50,
                     n_node_mod=100,
                     n_edge_mod=200,
                     feat_lim_min=-1,
                     feat_lim_max=1,
                     device=device)
        # mask=torch.zeros(size, dtype=bool)
        # mask[idx_test]=1
        adj_attack, features = attack.attack(model=Attackmodel,
                                             adj=adj,
                                             features=features,
                                             feat_norm=None,
                                             index_target=idx_test)  # ,
        # adj_norm_func=model.adj_norm_func)
        adj = adj_preprocess(adj=adj_attack,
                             # adj_norm_func=model.adj_norm_func,
                             model_type=Attackmodel.model_type,
                             device=device)

        # features= torch.cat([features, features_attack])
        norm_length = 1  # Variable lambda in the original paper

        trainstart = time.time()
        tracemalloc.start()

        PGDTestAcc = evaluate(Attackmodel, features, adj, data, idx_test)
        
        testMemories[6] = tracemalloc.get_traced_memory()[1]

        # stopping the library
        tracemalloc.stop()
        trainend = time.time()
        testtimes[6] = trainend - trainstart

        result[6][0]=PGDTestAcc

        adj = deepcopy(AdjCopy)
        features = deepcopy(FeatCopy)
    if rand:
        attack = RAND(n_inject_max=100,
                      n_edge_max=200,
                      feat_lim_min=-1,
                      feat_lim_max=1,
                      device=device)
        mask = torch.zeros(size, dtype=bool)
        mask[idx_test] = 1
        adj_attack, features_attack = attack.attack(model=Attackmodel,
                                                    adj=adj,
                                                    features=features,
                                                    # feat_norm=None,
                                                    target_mask=mask,
                                                    adj_norm_func=Attackmodel.adj_norm_func)
        adj = adj_preprocess(adj=adj_attack,
                             # adj_norm_func=model.adj_norm_func,
                             model_type=Attackmodel.model_type,
                             device=device)
        features = torch.cat([features, features_attack])
        norm_length = 1  # Variable lambda in the original paper
        
        trainstart = time.time()
        tracemalloc.start()
        
        RandTestAcc = evaluate(Attackmodel, features, adj, data, idx_test)
        
        testMemories[7] = tracemalloc.get_traced_memory()[1]

        # stopping the library
        tracemalloc.stop()
        trainend = time.time()
        testtimes[7] = trainend - trainstart
        
        result[7][0]=RandTestAcc

        adj = deepcopy(AdjCopy)
        features = deepcopy(FeatCopy)

    # trainMemory=tracemalloc.get_traced_memory()[1]

    # stopping the library
    # tracemalloc.stop()

    #for index, method in enumerate([
    #    "Base_Degree", "Base_Pagerank", "Base_Between",
    #    "Base_Random", "GC-RWCS", "RWCS", "InfMax-Unif", "InfMax-Norm", "New_sort_sumtest_Unif",
    #    "New_sort_sumtest_erf", "DICE", "NEA", "STACK", "None"  # ,"FGSM","PGD","RAND","DICE","NEA","STACK", "None"
    #]):
    for index, method in enumerate([
        "DICE","FGA","NEA","STACK","FLIP","FGSM","PGD","RAND", "Vanilla"
    ]):
        print("{} : Test Accuracy : {:.4f}, Test Loss : {:.4f}".format(
            method, result[index][0].item(), result[index][1].item()))
    # # displaying the memory
    # trainMemory=tracemalloc.get_traced_memory()[1]
    #
    # # stopping the library
    # tracemalloc.stop()
    # trainMemory=0
    return result.numpy(),trainMemory,traintime,epoch,testMemories,testtimes#,TrainigAccs[:,-1] only reporting the maxmimum memory used(not the current one)


###########################################################################################################################################comment here


idx_train=np.load('../OrigIdx_train.npy')
idx_test=np.load('../OrigIdx_test.npy')
idx_val=np.load('../OrigIdx_val.npy')

#OrigFlag=True
SplitFlag=SplitVar
if SplitFlag==1:

    print("\n Original Evaluation!\n")
    
    
    #######################################
if (SplitFlag>=2) and (SplitFlag<7):
    # Random node drop
    
    idx_train=np.load('../RandomIndexTrain_CompVar_SplitVar.npy')
    idx_test=np.load('../RandomIndexTest_CompVar_SplitVar.npy')
    idx_val=np.load('../RandomIndexVal_CompVar_SplitVar.npy')

    print("\n Random CompVar_SplitVar Evaluation!\n")
    


if SplitFlag>=7:
    if SplitFlag==7:
        DistMeas1=["JaccardDistance1"]#,"Hamming","JaccardDistance","LaplacianSpectral"]
        DistMeas2=["JaccardDistance1"]#,"LaplacianSpectral"]
    if SplitFlag==8:
        DistMeas1=["JaccardDistance1"]#,"Hamming","JaccardDistance","LaplacianSpectral"]
        DistMeas2=["Hamming"]#,"LaplacianSpectral"]
    if SplitFlag==9:
        DistMeas1=["JaccardDistance1"]#,"Hamming","JaccardDistance","LaplacianSpectral"]
        DistMeas2=["JaccardDistance"]#,"LaplacianSpectral"]
    if SplitFlag==10:
        DistMeas1=["JaccardDistance1"]#,"Hamming","JaccardDistance","LaplacianSpectral"]
        DistMeas2=["Semi_PearsonCorr"]#,"LaplacianSpectral"]
    if SplitFlag==11:
        DistMeas1=["JaccardDistance1"]#,"Hamming","JaccardDistance","LaplacianSpectral"]
        DistMeas2=["PearsonCorr"]#,"LaplacianSpectral"]
    if SplitFlag==12:
        DistMeas1=["JaccardDistance1"]#,"Hamming","JaccardDistance","LaplacianSpectral"]
        DistMeas2=["linearCKA"]#,"LaplacianSpectral"]
    if SplitFlag==13:
        DistMeas1=["JaccardDistance1"]#,"Hamming","JaccardDistance","LaplacianSpectral"]
        DistMeas2=["kernel_CKA"]#,"LaplacianSpectral"]

    if SplitFlag==14:
        DistMeas1=["Hamming"]#,"Hamming","JaccardDistance","LaplacianSpectral"]
        DistMeas2=["JaccardDistance1"]#,"LaplacianSpectral"]
    if SplitFlag==15:
        DistMeas1=["Hamming"]#,"Hamming","JaccardDistance","LaplacianSpectral"]
        DistMeas2=["Hamming"]#,"LaplacianSpectral"]
    if SplitFlag==16:
        DistMeas1=["Hamming"]#,"Hamming","JaccardDistance","LaplacianSpectral"]
        DistMeas2=["JaccardDistance"]#,"LaplacianSpectral"]
    if SplitFlag==17:
        DistMeas1=["Hamming"]
        DistMeas2=["Semi_PearsonCorr"]
    if SplitFlag==18:
        DistMeas1=["Hamming"]
        DistMeas2=["PearsonCorr"]
    if SplitFlag==19:
        DistMeas1=["Hamming"]
        DistMeas2=["linearCKA"]
    if SplitFlag==20:
        DistMeas1=["Hamming"]
        DistMeas2=["kernel_CKA"]

    if SplitFlag==21:
        DistMeas1=["JaccardDistance"]
        DistMeas2=["JaccardDistance1"]
    if SplitFlag==22:
        DistMeas1=["JaccardDistance"]
        DistMeas2=["Hamming"]
    if SplitFlag==23:
        DistMeas1=["JaccardDistance"]
        DistMeas2=["JaccardDistance"]
    if SplitFlag==24:
        DistMeas1=["JaccardDistance"]
        DistMeas2=["Semi_PearsonCorr"]
    if SplitFlag==25:
        DistMeas1=["JaccardDistance"]
        DistMeas2=["PearsonCorr"]
    if SplitFlag==26:
        DistMeas1=["JaccardDistance"]
        DistMeas2=["linearCKA"]
    if SplitFlag==27:
        DistMeas1=["JaccardDistance"]
        DistMeas2=["kernel_CKA"]
        
    if SplitFlag==28:
        DistMeas1=["Semi_PearsonCorr"]
        DistMeas2=["JaccardDistance1"]
    if SplitFlag==29:
        DistMeas1=["Semi_PearsonCorr"]
        DistMeas2=["Hamming"]
    if SplitFlag==30:
        DistMeas1=["Semi_PearsonCorr"]
        DistMeas2=["JaccardDistance"]
    if SplitFlag==31:
        DistMeas1=["Semi_PearsonCorr"]
        DistMeas2=["Semi_PearsonCorr"]
    if SplitFlag==32:
        DistMeas1=["Semi_PearsonCorr"]
        DistMeas2=["PearsonCorr"]
    if SplitFlag==33:
        DistMeas1=["Semi_PearsonCorr"]
        DistMeas2=["linearCKA"]
    if SplitFlag==34:
        DistMeas1=["Semi_PearsonCorr"]
        DistMeas2=["kernel_CKA"]

    if SplitFlag==35:
        DistMeas1=["PearsonCorr"]
        DistMeas2=["JaccardDistance1"]
    if SplitFlag==36:
        DistMeas1=["PearsonCorr"]
        DistMeas2=["Hamming"]
    if SplitFlag==37:
        DistMeas1=["PearsonCorr"]
        DistMeas2=["JaccardDistance"]
    if SplitFlag==38:
        DistMeas1=["PearsonCorr"]
        DistMeas2=["Semi_PearsonCorr"]
    if SplitFlag==39:
        DistMeas1=["PearsonCorr"]
        DistMeas2=["PearsonCorr"]
    if SplitFlag==40:
        DistMeas1=["PearsonCorr"]
        DistMeas2=["linearCKA"]
    if SplitFlag==41:
        DistMeas1=["PearsonCorr"]
        DistMeas2=["kernel_CKA"]

    if SplitFlag==42:
        DistMeas1=["linearCKA"]
        DistMeas2=["JaccardDistance1"]
    if SplitFlag==43:
        DistMeas1=["linearCKA"]
        DistMeas2=["Hamming"]
    if SplitFlag==44:
        DistMeas1=["linearCKA"]
        DistMeas2=["JaccardDistance"]
    if SplitFlag==45:
        DistMeas1=["linearCKA"]
        DistMeas2=["Semi_PearsonCorr"]
    if SplitFlag==46:
        DistMeas1=["linearCKA"]
        DistMeas2=["PearsonCorr"]
    if SplitFlag==47:
        DistMeas1=["linearCKA"]
        DistMeas2=["linearCKA"]
    if SplitFlag==48:
        DistMeas1=["linearCKA"]
        DistMeas2=["kernel_CKA"]

    if SplitFlag==49:
        DistMeas1=["kernel_CKA"]
        DistMeas2=["JaccardDistance1"]
    if SplitFlag==50:
        DistMeas1=["kernel_CKA"]
        DistMeas2=["Hamming"]
    if SplitFlag==51:
        DistMeas1=["kernel_CKA"]
        DistMeas2=["JaccardDistance"]
    if SplitFlag==52:
        DistMeas1=["kernel_CKA"]
        DistMeas2=["Semi_PearsonCorr"]
    if SplitFlag==53:
        DistMeas1=["kernel_CKA"]
        DistMeas2=["PearsonCorr"]
    if SplitFlag==54:
        DistMeas1=["kernel_CKA"]
        DistMeas2=["linearCKA"]
    if SplitFlag==55:
        DistMeas1=["kernel_CKA"]
        DistMeas2=["kernel_CKA"]

    #Subgraph evaluation


    idx_train = np.load('../SubIdx_train-'+DistMeas1[0]+'-'+DistMeas2[0]+'_CompVar.npy')###############
    idx_test = np.load('../SubIdx_test-'+DistMeas1[0]+'-'+DistMeas2[0]+'_CompVar.npy')
    idx_val = np.load('../SubIdx_val-'+DistMeas1[0]+'-'+DistMeas2[0]+'_CompVar.npy')

    print("\n Our Method "+DistMeas1[0]+"-"+DistMeas2[0]+" Evaluation!\n")


#GraphResult= Evaluation( 100,idx_train,idx_val,idx_test)  # 33  Acc,TrainMemory,TrainTime,TestMemory,TestTime 
for EvalIter in range(5):
    GraphResult=Evaluation( 100,idx_train,idx_val,idx_test)  # 33  Acc,TrainMemory,TrainTime,TestMemory,TestTime 
    with open('Accs'+str(EvalIter)+'.npy','wb') as f1:
        np.save(f1,np.array(GraphResult[0]));
    with open('TrainResources'+str(EvalIter)+'.npy','wb') as f2:
        np.save(f2,np.array(GraphResult[1:4]));
    with open('TestMemory'+str(EvalIter)+'.npy','wb') as f3:
        np.save(f3,np.array(GraphResult[4]));
    with open('TestTimes'+str(EvalIter)+'.npy','wb') as f4:
        np.save(f4,np.array(GraphResult[5]));
                                                  


#with open('Accs.npy','wb') as f1:
#    np.save(f1,np.array(GraphResult[0]));
#with open('TrainResources.npy','wb') as f2:
#    np.save(f2,np.array(GraphResult[1:4]));
#with open('TestMemory.npy','wb') as f3:
#    np.save(f3,np.array(GraphResult[4]));
#with open('TestTimes.npy','wb') as f4:
#    np.save(f4,np.array(GraphResult[5]));



import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd

fig,ax=plt.subplots(figsize=(5,4));
#plt.subplots_adjust(bottom=0.3);
plt.xlabel('Method', fontdict=dict(weight='bold'), fontsize=12);
plt.ylabel('Train+Average Test Time', fontdict=dict(weight='bold'), fontsize=12);
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right');
#ax.set_ylim([0, 1])

x=["Rand","Degree","Feature","Jaccard","SVD","Orig","SHERD"]

cmap = plt.colormaps["Dark2"]#hsv"]

#mpl.use('TkAgg')
#Cora
labels=["Cora","Citeseer","Mini-Pubmed","Mini-Placenta"]
datasets=["./Cora-10-25-AttackAllBunchPGD/","./Citeseer-10-25-AttackAllBunchPGD/",
            "./MiniPubmed-10-30-AttackAllBunchPGD/","./Placenta5000-11-3-AttackAllBunchPGD/"]
hp0=["tb50_200_2/Evaluate/Train1","tb70_200_2/Evaluate/Train1","tb50_100_2/Evaluate/Train1","tb30_100_2/Evaluate/Train1"]
hparameters=["/z10_0.4/","/z10_0.5/","/z32_0.5/","/z10_0.5/"]
CompRatio=["0.4/","0.5/","0.5/","0.5/"]
markers=["o","^","s","*"]
for iter in range(4):

    file=datasets[iter]+hp0[iter]+hparameters[iter]
    Times=[]
    for i in range(5):
        #Times.append(np.load(file+"TrainResources"+str(i)+".npy")[1])
        Times.append(np.mean(np.load(file+"TestTimes"+str(i)+".npy"))+np.load(file+"TrainResources"+str(i)+".npy")[1])
    SHERDtime=np.mean(np.array(Times))
    SHERDtimeSTD=np.std(np.array(Times))
    
    Times=[]
    for j in range(2,7):
        file=datasets[iter]+hp0[iter]+"/z"+str(j)+"_"+CompRatio[iter]
        for i in range(5):
            #Times.append(np.load(file+"TrainResources"+str(i)+".npy")[1])
            Times.append(np.mean(np.load(file+"TestTimes"+str(i)+".npy"))+np.load(file+"TrainResources"+str(i)+".npy")[1])
    Randtime=np.mean(np.array(Times))
    RandtimeSTD=np.std(np.array(Times))
    
    file=datasets[iter]+hp0[iter]+"/z1_0.1/"
    Times=[]
    for i in range(5):
        #Times.append(np.load(file+"TrainResources"+str(i)+".npy")[1])
        Times.append(np.mean(np.load(file+"TestTimes"+str(i)+".npy"))+np.load(file+"TrainResources"+str(i)+".npy")[1])
    Origtime=np.mean(np.array(Times))
    OrigtimeSTD=np.std(np.array(Times))
    
    file=datasets[iter]+"/GCNJaccardBaseline/"
    Times=[]
    for i in range(5):
        #Times.append(np.load(file+"JaccardTrainResources"+str(i)+".npy")[1])
        Times.append(np.mean(np.load(file+"JaccardTestTimes"+str(i)+".npy"))+np.load(file+"JaccardTrainResources"+str(i)+".npy")[1])
    Jaccardtime=np.mean(np.array(Times))
    JaccardtimeSTD=np.std(np.array(Times))
    
    file=datasets[iter]+"/GCNSVDBaseline/"
    Times=[]
    for i in range(5):
        #Times.append(np.load(file+"JaccardTrainResources"+str(i)+".npy")[1])
        Times.append(np.mean(np.load(file+"JaccardTestTimes"+str(i)+".npy"))+np.load(file+"JaccardTrainResources"+str(i)+".npy")[1])
    SVDtime=np.mean(np.array(Times))
    SVDtimeSTD=np.std(np.array(Times))
    
    
    file=datasets[iter]+"/FeaturesMean/backup/Evaluate/copy_Train/"+CompRatio[iter]
    Times=[]
    for i in range(5):
        #Times.append(np.load(file+"JaccardTrainResources"+str(i)+".npy")[1])
        Times.append(np.mean(np.load(file+"TestTimes"+str(i)+".npy"))+np.load(file+"TrainResources"+str(i)+".npy")[1])
    Feattime=np.mean(np.array(Times))
    FeattimeSTD=np.std(np.array(Times))
    
    
    file=datasets[iter]+"/NodeDegreeBaseline/backup/Evaluate/copy_Train/"+CompRatio[iter]
    Times=[]
    for i in range(5):
        #Times.append(np.load(file+"JaccardTrainResources"+str(i)+".npy")[1])
        Times.append(np.mean(np.load(file+"TestTimes"+str(i)+".npy"))+np.load(file+"TrainResources"+str(i)+".npy")[1])
    Degtime=np.mean(np.array(Times))
    DegtimeSTD=np.std(np.array(Times))
    
    y=np.array([Randtime,Degtime,Feattime,Jaccardtime,SVDtime,Origtime,SHERDtime])
    ySTD=np.array([RandtimeSTD,DegtimeSTD,FeattimeSTD,JaccardtimeSTD,SVDtimeSTD,OrigtimeSTD,SHERDtimeSTD])
    ax.errorbar(x=x, y=y, color=cmap(iter), label=labels[iter],marker=markers[iter]);
    ax.fill_between(x,y-ySTD,y+ySTD,color=cmap(iter),alpha=0.4)
    ax.legend(loc='upper left');
    #ax.legend(bbox_to_anchor=(1.05, 1),loc='upper left');
plt.tight_layout()
#plt.title("Train and Test Time")
plt.savefig("Time.png")

#Memory

fig,ax=plt.subplots(figsize=(5,4));
#plt.subplots_adjust(bottom=0.3);
plt.xlabel('Method', fontdict=dict(weight='bold'), fontsize=12);
plt.ylabel('Train+Average Test Memory', fontdict=dict(weight='bold'), fontsize=12);
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right');
#ax.set_ylim([0, 1])

x=["Rand","Degree","Feature","Jaccard","SVD","Orig","SHERD"]

cmap = plt.colormaps["Dark2"]#hsv"]

for iter in range(4):

    file=datasets[iter]+hp0[iter]+hparameters[iter]
    Times=[]
    for i in range(5):
        #Times.append(np.load(file+"TrainResources"+str(i)+".npy")[1])
        Times.append(np.mean(np.load(file+"TestMemory"+str(i)+".npy"))+np.load(file+"TrainResources"+str(i)+".npy")[0])
    SHERDtime=np.mean(np.array(Times))
    SHERDtimeSTD=np.std(np.array(Times))
    
    Times=[]
    for j in range(2,7):
        file=datasets[iter]+hp0[iter]+"/z"+str(j)+"_"+CompRatio[iter]
        for i in range(5):
            #Times.append(np.load(file+"TrainResources"+str(i)+".npy")[1])
            Times.append(np.mean(np.load(file+"TestMemory"+str(i)+".npy"))+np.load(file+"TrainResources"+str(i)+".npy")[0])
    Randtime=np.mean(np.array(Times))
    RandtimeSTD=np.std(np.array(Times))
    
    file=datasets[iter]+hp0[iter]+"/z1_0.1/"
    Times=[]
    for i in range(5):
        #Times.append(np.load(file+"TrainResources"+str(i)+".npy")[1])
        Times.append(np.mean(np.load(file+"TestMemory"+str(i)+".npy"))+np.load(file+"TrainResources"+str(i)+".npy")[0])
    Origtime=np.mean(np.array(Times))
    OrigtimeSTD=np.std(np.array(Times))
    
    file=datasets[iter]+"/GCNJaccardBaseline/"
    Times=[]
    for i in range(5):
        #Times.append(np.load(file+"JaccardTrainResources"+str(i)+".npy")[1])
        Times.append(np.mean(np.load(file+"JaccardTestMemory"+str(i)+".npy"))+np.load(file+"JaccardTrainResources"+str(i)+".npy")[0])
    Jaccardtime=np.mean(np.array(Times))
    JaccardtimeSTD=np.std(np.array(Times))
    
    file=datasets[iter]+"/GCNSVDBaseline/"
    Times=[]
    for i in range(5):
        #Times.append(np.load(file+"JaccardTrainResources"+str(i)+".npy")[1])
        Times.append(np.mean(np.load(file+"JaccardTestMemory"+str(i)+".npy"))+np.load(file+"JaccardTrainResources"+str(i)+".npy")[0])
    SVDtime=np.mean(np.array(Times))
    SVDtimeSTD=np.std(np.array(Times))
    
    
    file=datasets[iter]+"/FeaturesMean/backup/Evaluate/copy_Train/"+CompRatio[iter]
    Times=[]
    for i in range(5):
        #Times.append(np.load(file+"JaccardTrainResources"+str(i)+".npy")[1])
        Times.append(np.mean(np.load(file+"TestMemory"+str(i)+".npy"))+np.load(file+"TrainResources"+str(i)+".npy")[0])
    Feattime=np.mean(np.array(Times))
    FeattimeSTD=np.std(np.array(Times))
    
    
    file=datasets[iter]+"/NodeDegreeBaseline/backup/Evaluate/copy_Train/"+CompRatio[iter]
    Times=[]
    for i in range(5):
        #Times.append(np.load(file+"JaccardTrainResources"+str(i)+".npy")[1])
        Times.append(np.mean(np.load(file+"TestMemory"+str(i)+".npy"))+np.load(file+"TrainResources"+str(i)+".npy")[0])
    Degtime=np.mean(np.array(Times))
    DegtimeSTD=np.std(np.array(Times))
    
    y=np.array([Randtime,Degtime,Feattime,Jaccardtime,SVDtime,Origtime,SHERDtime])
    ySTD=np.array([RandtimeSTD,DegtimeSTD,FeattimeSTD,JaccardtimeSTD,SVDtimeSTD,OrigtimeSTD,SHERDtimeSTD])
    ax.errorbar(x=x, y=y, color=cmap(iter+4), label=labels[iter],marker=markers[iter]);
    ax.fill_between(x,y-ySTD,y+ySTD,color=cmap(iter+4),alpha=0.4)
    ax.legend(loc='upper left');
    #ax.legend(bbox_to_anchor=(1.05, 1),loc='upper left');
plt.tight_layout()
#plt.title("Train and Test Time")
plt.savefig("Memory.png")

#Training Epochs

fig,ax=plt.subplots(figsize=(5,4));
#plt.subplots_adjust(bottom=0.3);
plt.xlabel('Method', fontdict=dict(weight='bold'), fontsize=12);
plt.ylabel('Training Epochs', fontdict=dict(weight='bold'), fontsize=12);
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right');
#ax.set_ylim([0, 1])

x=["Rand","Degree","Feature","Jaccard","SVD","Orig","SHERD"]

cmap = plt.colormaps["tab10"]#hsv"]

for iter in range(4):

    file=datasets[iter]+hp0[iter]+hparameters[iter]
    Times=[]
    for i in range(5):
        #Times.append(np.load(file+"TrainResources"+str(i)+".npy")[1])
        Times.append(np.load(file+"TrainResources"+str(i)+".npy")[2])
    SHERDtime=np.mean(np.array(Times))
    SHERDtimeSTD=np.std(np.array(Times))
    
    Times=[]
    for j in range(2,7):
        file=datasets[iter]+hp0[iter]+"/z"+str(j)+"_"+CompRatio[iter]
        for i in range(5):
            #Times.append(np.load(file+"TrainResources"+str(i)+".npy")[1])
            Times.append(np.load(file+"TrainResources"+str(i)+".npy")[2])
    Randtime=np.mean(np.array(Times))
    RandtimeSTD=np.std(np.array(Times))
    
    file=datasets[iter]+hp0[iter]+"/z1_0.1/"
    Times=[]
    for i in range(5):
        #Times.append(np.load(file+"TrainResources"+str(i)+".npy")[1])
        Times.append(np.load(file+"TrainResources"+str(i)+".npy")[2])
    Origtime=np.mean(np.array(Times))
    OrigtimeSTD=np.std(np.array(Times))
    
    file=datasets[iter]+"/GCNJaccardBaseline/"
    Times=[]
    for i in range(5):
        #Times.append(np.load(file+"JaccardTrainResources"+str(i)+".npy")[1])
        Times.append(np.load(file+"JaccardTrainResources"+str(i)+".npy")[2])
    Jaccardtime=np.mean(np.array(Times))
    JaccardtimeSTD=np.std(np.array(Times))
    
    file=datasets[iter]+"/GCNSVDBaseline/"
    Times=[]
    for i in range(5):
        #Times.append(np.load(file+"JaccardTrainResources"+str(i)+".npy")[1])
        Times.append(np.load(file+"JaccardTrainResources"+str(i)+".npy")[2])
    SVDtime=np.mean(np.array(Times))
    SVDtimeSTD=np.std(np.array(Times))
    
    
    file=datasets[iter]+"/FeaturesMean/backup/Evaluate/copy_Train/"+CompRatio[iter]
    Times=[]
    for i in range(5):
        #Times.append(np.load(file+"JaccardTrainResources"+str(i)+".npy")[1])
        Times.append(np.load(file+"TrainResources"+str(i)+".npy")[2])
    Feattime=np.mean(np.array(Times))
    FeattimeSTD=np.std(np.array(Times))
    
    
    file=datasets[iter]+"/NodeDegreeBaseline/backup/Evaluate/copy_Train/"+CompRatio[iter]
    Times=[]
    for i in range(5):
        #Times.append(np.load(file+"JaccardTrainResources"+str(i)+".npy")[1])
        Times.append(np.load(file+"TrainResources"+str(i)+".npy")[2])
    Degtime=np.mean(np.array(Times))
    DegtimeSTD=np.std(np.array(Times))
    
    y=np.array([Randtime,Degtime,Feattime,Jaccardtime,SVDtime,Origtime,SHERDtime])
    ySTD=np.array([RandtimeSTD,DegtimeSTD,FeattimeSTD,JaccardtimeSTD,SVDtimeSTD,OrigtimeSTD,SHERDtimeSTD])
    ax.errorbar(x=x, y=y, color=cmap(iter+4), label=labels[iter],marker=markers[iter]);
    ax.fill_between(x,y-ySTD,y+ySTD,color=cmap(iter+4),alpha=0.4)
    ax.legend(loc='upper left');
    #ax.legend(bbox_to_anchor=(1.05, 1),loc='upper left');
plt.tight_layout()
#plt.title("Train and Test Time")
plt.savefig("Epochs.png")


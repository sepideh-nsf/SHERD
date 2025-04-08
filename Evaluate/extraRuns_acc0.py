import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd

#mpl.use('TkAgg')
for compRatio in [0.1, 0.2, 0.3, 0.4, 0.5,0.7,0.9]:
    fig,ax=plt.subplots(figsize=(15,6));
    plt.subplots_adjust(bottom=0.3);
    plt.xlabel('Node Selection for Attack Method', fontdict=dict(weight='bold'), fontsize=12);
    plt.ylabel('Adversarial Test Data Accuracy', fontdict=dict(weight='bold'), fontsize=12);
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right');
    ax.set_ylim([0, 1])

    x=[
            "Base_Degree", "Base_Pagerank", "Base_Between",
                "Base_Random", "GC-RWCS", "RWCS", "InfMax-Unif", "InfMax-Norm", "New_sort_sumtest_Unif",
                "New_sort_sumtest_erf","DICE","NEA","STACK", "None"#"FGSM","PGD","RAND","DICE","NEA","STACK", "None"
        ];
    #Result = np.load('./Train1/z1_'+str(compRatio)+'/Accs.npy',
    #                 allow_pickle=True)
    Accs=[]
    for i in range(5):
        # for itera2 in range(1,6):
        Result = np.load('./Train1/z1_'+str(compRatio)+'/Accs'+str(i)+'.npy',allow_pickle=True)
        Accs.append(Result)

    # Accs.append(Result[0])
    # SubMeanTest = np.mean(np.array(Accs)[:, :, 0], axis=0)
    # SubMinsTest = np.min(np.array(Accs)[:, :, 0], axis=0)
    # SubMaxsTest = np.max(np.array(Accs)[:, :, 0], axis=0)
    # y = SubMeanTest;
    #y=Result[:,0]
    y=np.mean(np.array(Accs)[:, :, 0], axis=0)
    # df = pd.DataFrame({'Attack': x, 'Accuracy': y, 'Min': np.array(SubMinsTest), 'Max': np.array(SubMaxsTest)});
    # df['ymin'] = df.Accuracy - df.Min;
    # df['ymax'] = df.Max - df.Accuracy;
    # yerr = df[['ymin', 'ymax']].T.to_numpy();
    # itera = itera1 - 7
    ax.errorbar(x=x, y=y,color='red',label="Original Graph")#, yerr=yerr, color=cmap(itera * 35), capsize=5,
                #label=DistMeas1[itera % 4] + "-attack Robust Subgraph" + DistMeas2[int(itera / 4)]);

    #Random Subgraph
    Accs=[]
    for j in range(2,7):
        for i in range(5):
            Result = np.load('./Train1/z'+str(j)+'_'+str(compRatio)+'/Accs'+str(i)+'.npy',allow_pickle=True)
            Accs.append(Result)
    SubMeanTest = np.mean(np.array(Accs)[:, :, 0], axis=0)
    SubMinsTest = np.min(np.array(Accs)[:, :, 0], axis=0)
    SubMaxsTest = np.max(np.array(Accs)[:, :, 0], axis=0)
    y = SubMeanTest;
    df = pd.DataFrame({'Attack': x, 'Accuracy': y, 'Min': np.array(SubMinsTest), 'Max': np.array(SubMaxsTest)});
    df['ymin'] = df.Accuracy - df.Min;
    df['ymax'] = df.Max - df.Accuracy;
    yerr = df[['ymin', 'ymax']].T.to_numpy();
    ax.errorbar(x=x, y=y, yerr=yerr, color='green', capsize=5,label='Random Subgraph');

    #plotting part 1

    cmap = plt.colormaps["tab20"]
    DistMeas1=["JaccardDistance1","Hamming","JaccardDistance","Semi_PearsonCorr"]#["DeltaCon","DeltaCon","Hamming","JaccardDistance",
    DistMeas2=["JaccardDistance1","Hamming","JaccardDistance","Semi_PearsonCorr"]

    itera1=7
    # itera2=1
    for PerformDist in DistMeas2:
        for RobustDist in DistMeas1:
            Accs = []
            trainMemories = []
            traintime = []
            testMemory = []
            testtime = []
            for i in range(5):
                # for itera2 in range(1,6):
                Result = np.load('./Train1/z'+str(itera1)+'_'+str(compRatio)+'/Accs'+str(i)+'.npy',allow_pickle=True)
                Accs.append(Result)
            SubMeanTest=np.mean(np.array(Accs)[:, :, 0], axis=0)
            SubMinsTest=np.min(np.array(Accs)[:, :, 0], axis=0)
            SubMaxsTest=np.max(np.array(Accs)[:, :, 0], axis=0)
            y = SubMeanTest;
            df = pd.DataFrame({'Attack': x, 'Accuracy': y, 'Min': np.array(SubMinsTest), 'Max': np.array(SubMaxsTest)});
            df['ymin'] = df.Accuracy - df.Min;
            df['ymax'] = df.Max - df.Accuracy;
            yerr = df[['ymin', 'ymax']].T.to_numpy();
            itera=itera1-7
            # if (DistMeas1[itera%4]=="LaplacianSpectral") and (DistMeas2[int(itera/4)]=="JaccardDistance"):
            ax.errorbar(x=x, y=y, yerr=yerr, color=cmap(itera), capsize=5, label=DistMeas1[int(itera/4)]+"-attack Robust Subgraph"+DistMeas2[int(itera%4)]);
            itera1+=1

    ax.legend(bbox_to_anchor=(1.05, 1),loc='upper left');
    plt.tight_layout()
    plt.title("Compression Ratio: "+str(compRatio))
    plt.savefig("AdversarialAccTest"+str(compRatio)+".png")
    # plt.show()
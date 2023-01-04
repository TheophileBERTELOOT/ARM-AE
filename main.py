import copy

from BenchmarkManager import *
from ARMAE import *
import time
import numpy as np
import torch
print(torch.cuda.is_available())


datasetName = 'nursery'
isPolypharmacie = False
isLoadedModel = False
nbEpoch = 100
modelPath = 'models/'
data = pd.read_csv('../data/'+datasetName+'.csv',index_col=0,dtype=float,header=0)
times = []
# columns = ['FpExecTime','FpTransformRuleTime','FpTotalTime','FpSupportAvg','FpConfAvg','nbNotNull','nnAvgSupp',
#            'nnAvgConf','nnFindInFp','FpFindInnn','timeTraining','timeCreatingRule','timeComputingMeasure','iter','goalLoss','nbEpoch']
#------------------AJOUTER LES COLONNES A LA FIN DNAS LE DF
columns = ['FpExecTime','FpTransformRuleTime','FpTotalTime','FpSupportAvg','FpConfAvg','nbNotNull','nnAvgSupp',
           'nnAvgConf','nnFindInFp','FpFindInnn','timeTraining','timeCreatingRule','timeComputingMeasure','iter','goalLoss','nbEpoch']
scores = []
goalLosses = [1,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
goalLoss = 0.1
lastEpoch = 0
nbRun = 10

dataSize = len(data.loc[0])
bm = BenchmarkManager('Results/', dataSize)
# for goalLoss in goalLosses:
for i in range(nbRun):

    NN = NeuralNetwork(dataSize,maxEpoch=nbEpoch,batchSize=128,learningRate=10e-3, likeness=0.5, maxLossDifference=0.01,goalLoss=goalLoss)

    dataLoader = NN.dataPretraitement(data)
    t1 = time.time()
    if goalLoss!=1:
        lastEpoch = NN.train(dataLoader,modelPath)
    else:
        lastEpoch = 0
    t2 = time.time()

    timeCreatingRule, timeComputingMeasure = NN.generateRules(data, numberOfRules=2, nbAntecedent=2,
                                                     path='Results/NN/' + datasetName + str(i)+'.csv')
    timeTraining = t2-t1
    times.append([timeTraining,timeCreatingRule,timeComputingMeasure])

    firstScore = bm.CompareToExhaustive(data, 'Results/Exhaustive/' + datasetName + str(i) + '.csv', 0.01, 0.01,
                                            'Results/NN/' + datasetName + str(i) + '.csv', i, threshold=0,
                                            nbAntecedent=2, dataset=datasetName)

    scoresIter = copy.deepcopy(firstScore)
    scoresIter += [timeCreatingRule,timeComputingMeasure,timeTraining,i,goalLoss,lastEpoch]
    scores.append(scoresIter)

df = pd.DataFrame(scores,columns=columns)
df.to_csv('Results/Final/'+datasetName+'.csv')




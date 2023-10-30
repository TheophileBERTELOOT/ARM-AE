import copy

from BenchmarkManager import *
from ARMAE import *
import time
import numpy as np
import torch


print(torch.cuda.is_available())

nbRun = 2
nbEpoch = 2
batchSize = 128
learningRate=10e-3
#the proportion of similar items in rule with the same consequents
likeness = 0.5
#The number of rule per consequent
numberOfRules = 2
#The maximum number of antecedents
nbAntecedents = 2
#For Fp-Growth
minSupp = 0.01
minConf = 0.01


datasetName = 'nursery'


ARMAEResultsPath= 'Results/NN/'
exhaustiveResultsPath = 'Results/Exhaustive/'
overallResultsPath = 'Results/Final/'
dataPath ='data/'+datasetName+'.csv'



isLoadedModel = False

#the path to save the models
modelPath = 'models/'

#if you want to use a pretrained model those are the path to update
encoderPath = 'models/1encoder.pt'
decoderPath = 'models/1decoder.pt'

data = pd.read_csv(dataPath,index_col=0,dtype=float,header=0)
times = []

columns = ['FpExecTime','FpTransformRuleTime','FpTotalTime','FpSupportAvg','FpConfAvg','nbNotNull','nnAvgSupp',
           'nnAvgConf','nnFindInFp','FpFindInnn','timeTraining','timeCreatingRule','timeComputingMeasure','iter']
scores = []





dataSize = len(data.loc[0])
bm = BenchmarkManager('Results/', dataSize)
# for goalLoss in goalLosses:
for i in range(nbRun):

    NN = ARMAE(dataSize,maxEpoch=nbEpoch,batchSize=batchSize,learningRate=learningRate, likeness=likeness)

    dataLoader = NN.dataPretraitement(data)
    t1 = time.time()
    if not isLoadedModel:
        NN.train(dataLoader,modelPath)
    else:
        NN.load(encoderPath,decoderPath)
    t2 = time.time()

    timeCreatingRule, timeComputingMeasure = NN.generateRules(data, numberOfRules=numberOfRules, nbAntecedent=nbAntecedents,
                                                     path=ARMAEResultsPath + datasetName + str(i)+'.csv')
    timeTraining = t2-t1
    times.append([timeTraining,timeCreatingRule,timeComputingMeasure])

    firstScore = bm.CompareToExhaustive(data, exhaustiveResultsPath + datasetName + str(i) + '.csv', minSupp, minConf,
                                            ARMAEResultsPath + datasetName + str(i) + '.csv', i,
                                            nbAntecedent=nbAntecedents, dataset=datasetName)

    scoresIter = copy.deepcopy(firstScore)
    scoresIter += [timeCreatingRule,timeComputingMeasure,timeTraining,i]
    scores.append(scoresIter)

df = pd.DataFrame(scores,columns=columns)
df.to_csv(overallResultsPath+datasetName+'.csv')




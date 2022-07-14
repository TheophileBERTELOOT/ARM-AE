import pandas as pd

from src.NeuralNetwork import *
from src.BenchmarkManager import *
import time

datasetName = 'connect'

# data = pd.read_csv('data/iris.csv',dtype=float,header=0,index_col=0)
data = pd.read_csv('data/'+datasetName+'.csv',index_col=0,dtype=float,header=0)
print(data)
dataSize = len(data.loc[0])
# print('datasize : '+str(dataSize))
# NN = NeuralNetwork(dataSize,True,False,False,False,False,False,numEpoch=100,output='tanh')
print('avant')
NN = NeuralNetwork(dataSize,False,False,False,False,False,False,True,numEpoch=20,output='tanh',learningRate=10e-4,likeness=0)
print('apres')
# # NN = NeuralNetwork(dataSize,False,True,False,False,False,False,numEpoch=100,output='tanh')
# # NN = NeuralNetwork(dataSize,False,False,True,False,False,False,numEpoch=500,output='tanh',dropout=0)
# # NN = NeuralNetwork(dataSize,False,False,False,True,False,False,numEpoch=100,output='relu')
# # NN = NeuralNetwork(dataSize,False,False,False,False,True,False,numEpoch=100,output='tanh',batchSize=dataSize)
# # NN = NeuralNetwork(dataSize,False,False,False,False,False,True,numEpoch=100,output='relu',batchSize=dataSize)
dataLoader = NN.dataPretraitement(data)
t1 = time.time()
NN.train(dataLoader)
t2 = time.time()
averageResult,timeCreatingRule,timeComputingMeasure = NN.generateRules(data,numberOfRules=2,nbAntecedent=1)
print(averageResult)
file = open('Results/ExecutionTime/'+datasetName+'.txt', "w")
file.write('Entrainement Model et creation de lois : '+str((t2-t1)+timeCreatingRule)+'\n'
           +'Calcul mesures : '+str(timeComputingMeasure)+'\n'
           +'Total : '+str((t2-t1)+timeCreatingRule+timeComputingMeasure)+'\n')
file.close()

NN.generateReport('Results/'+datasetName+'.txt')

bm = BenchmarkManager('Results/',dataSize)
# # nursery
# bm.CompareToExhaustive(data,'Results/Exhaustive/'+datasetName+'.csv',0.005,0.01,'Results/'+datasetName+'.txt',threshold=0,nbAntecedent=1,dataset=datasetName)
# # chess
bm.CompareToExhaustive(data,'Results/Exhaustive/'+datasetName+'.csv',0.05,0.05,'Results/'+datasetName+'.txt',threshold=0,nbAntecedent=1,dataset=datasetName)
# # bm.CompareToExhaustive(data,'Results/Exhaustive/iris.csv',0.1,0.9,'Results/LSTMTanH.txt',threshold=0.1)
# # bm.ExhaustiveSearch(data,'Results/Exhaustive/iris.csv',0.1)
bm.computeScores()
# bm.printHeatMap()


#
# bm = BenchmarkManager('Results/',dataSize)
# bm.RechercheGrilleLearningRateEpochs(data)



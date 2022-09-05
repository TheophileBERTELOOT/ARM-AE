import pandas as pd


# from src.BenchmarkManager import *
from src.NeuralNetwork import *
import time
import sqlite3
import random
import torch

datasetName = 'INSPQ'
isPolypharmacie = True
isLoadedModel = True
nbEpoch = 1
batchSize = 100000
modelPath = 'models/'
outcomeIndex = 473
nbHospit = 0
torch.set_num_threads(4)
# data = pd.read_csv('data/iris.csv',dtype=float,header=0,index_col=0)
# data = pd.read_csv('data/'+datasetName+'.csv',index_col=0,dtype=float,header=0)
con = sqlite3.connect('/home/travail/datapmsm_with_POLY.db')
nbPolyRows = pd.read_sql_query("SELECT MAX(ROWID) from POLY", con)
nbPolyRows = nbPolyRows['MAX(ROWID)'][0]
# nbPolyRows = 50000
print('nbPolyRows : ' + str(nbPolyRows))
data = pd.read_sql_query("SELECT * from POLY LIMIT 1", con)
data = data.drop(['AGE_AT_DEATH','AGE_AT_HOSPIT','AGE_AT_START','AGE_AT_END','ID_INDV8','DECES','ROWID'],axis=1)
print(data)
dataSize = len(data.loc[0])
columns = data.columns

# print('datasize : '+str(dataSize))
# NN = NeuralNetwork(dataSize,True,False,False,False,False,False,numEpoch=100,output='tanh')
print('avant')
NN = NeuralNetwork(dataSize, False, False, False, False, False, False, True, numEpoch=1, output='tanh',
                   learningRate=10e-4, likeness=0.6, isPolypharmacy=isPolypharmacie,con=con,columns=columns)

for i in range(1):
    # NN = NeuralNetwork(dataSize,False,False,False,False,False,False,True,numEpoch=10,output='tanh',
    #                    learningRate=10e-4,likeness=0.4,isPolypharmacy = isPolypharmacie)
    if not isLoadedModel:
        t1 = time.time()
        data = pd.read_sql_query('SELECT * from POLY  WHERE "HOSPIT" = ' + "'1'", con)
        data = data.drop(['AGE_AT_DEATH', 'AGE_AT_HOSPIT', 'AGE_AT_START', 'AGE_AT_END', 'ID_INDV8', 'DECES', 'ROWID'],
                         axis=1)
        print(data)
        for k in range(nbEpoch):
            print('epoch numero  : '+str(k))
            dataSize = len(data.loc[0])
            dataLoader = NN.dataPretraitement(data)
            print('apres')
        # # NN = NeuralNetwork(dataSize,False,True,False,False,False,False,numEpoch=100,output='tanh')
        # # NN = NeuralNetwork(dataSize,False,False,True,False,False,False,numEpoch=500,output='tanh',dropout=0)
        # # NN = NeuralNetwork(dataSize,False,False,False,True,False,False,numEpoch=100,output='relu')
        # # NN = NeuralNetwork(dataSize,False,False,False,False,True,False,numEpoch=100,output='tanh',batchSize=dataSize)
        # # NN = NeuralNetwork(dataSize,False,False,False,False,False,True,numEpoch=100,output='relu',batchSize=dataSize)


            if isLoadedModel:
                NN.load(modelPath)
            else:
                NN.train(dataLoader)
                try:
                    NN.save(modelPath)
                except:
                    print('dommage bien tente')
    if(isLoadedModel):
        NN.load(modelPath)
    t2 = time.time()
    timeCreatingRule,timeComputingMeasure = NN.generateRules(data,numberOfRules=10,nbAntecedent=5,outcomeIndex=outcomeIndex,path = 'Results/'+datasetName+'.txt')

    # file = open('Results/ExecutionTime/'+datasetName+'.txt', "a")
    # file.write('Entrainement Model et creation de lois : '+str((t2-t1)+timeCreatingRule)+'\n'
    #            +'Calcul mesures : '+str(timeComputingMeasure)+'\n'
    #            +'Total : '+str((t2-t1)+timeCreatingRule+timeComputingMeasure)+'\n')
    # file.close()

    # NN.generateReport('Results/'+datasetName+'.txt')

    # bm = BenchmarkManager('Results/',dataSize)
    # # # chess
    # bm.CompareToExhaustive(data,'Results/Exhaustive/'+datasetName+'.csv',0.005,0.01,'Results/'+datasetName+'.txt',i,threshold=0,nbAntecedent=2,dataset=datasetName)
    # # # nursery
    # # bm.CompareToExhaustive(data,'Results/Exhaustive/'+datasetName+'.csv',0.01,0.01,'Results/'+datasetName+'.txt',i,threshold=0,nbAntecedent=2,dataset=datasetName)
    # # # connect
    # # bm.CompareToExhaustive(data,'Results/Exhaustive/'+datasetName+'.csv',0.005,0.01,'Results/'+datasetName+'.txt',threshold=0,nbAntecedent=2,dataset=datasetName)
    # file = open('Results/ExecutionTime/'+datasetName+'.txt', "a")
    # file.write('______________________________________________________________ \n')
    # file.close()



# # bm.CompareToExhaustive(data,'Results/Exhaustive/iris.csv',0.1,0.9,'Results/LSTMTanH.txt',threshold=0.1)
# # bm.ExhaustiveSearch(data,'Results/Exhaustive/iris.csv',0.1)
# bm.computeScores()
# bm.printHeatMap()


#
# bm = BenchmarkManager('Results/',dataSize)
# bm.RechercheGrilleLearningRateEpochs(data)



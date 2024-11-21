import copy
import click
import pandas as pd
from arm_ae.BenchmarkManager import *
from arm_ae.ARMAE import *

import time
import numpy as np
import torch


print(torch.cuda.is_available())

@click.option(
    '--input-path', '-ip', 
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default='../data/nursery.csv', 
    help='the path of the input data',
    required = True
)

@click.option(
    '--ARMAE-results-path', '-arp', 
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default='../Results/NN/', 
    help='the path of the results of ARM-AE algorithms',
    required = False
)

@click.option(
    '--exhaustive-path', '-ep', 
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default='../Results/Exhaustive/', 
    help='the path of the results of exhaustive algorithms',
    required = False
)

@click.option(
    '--overall-results-path', '-orp', 
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default='../Results/Final/', 
    help='the path of the results combine of all the experiments',
    required = False
)

@click.option(
    '--nbRun', '-nr', 
    type=int,
    default=2, 
    help='the number of run for the experiment',
    required = False
)

@click.option(
    '--nbEpoch', '-ne', 
    type=int,
    default=2, 
    help='the number of epoch of training  for the ARM-AE',
    required = False
)

@click.option(
    '--batchSize', '-bs', 
    type=int,
    default=128, 
    help='the batchsize for the  training of ARM-AE',
    required = False
)

@click.option(
    '--learningRate', '-lr', 
    type=float,
    default=10e-3, 
    help='the learning rate for the  training of ARM-AE',
    required = False
)

@click.option(
    '--likeness', '-lk', 
    type=float,
    default=0.5, 
    help='the proportion of similar items in rule with the same consequents',
    required = False
)

@click.option(
    '--numberOfRules', '-nbor', 
    type=int,
    default=2, 
    help='The number of rule per consequent',
    required = False
)

@click.option(
    '--nbAntecedents', '-nba', 
    type=int,
    default=2, 
    help='The maximum number of antecedents',
    required = False
)

@click.option(
    '--minSupp', '-ms', 
    type=float,
    default=0.01, 
    help='min support for fpgrowth',
    required = False
)

@click.option(
    '--minConf', '-mc', 
    type=float,
    default=0.01, 
    help='min confidence for fpgrowth',
    required = False
)

@click.option(
    '--is-Loaded-Model', '-ilm', 
    help='do you want to train the ARM-AE model again',
    required = False
)

@click.option(
    '--model-path', '-mp', 
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default='models/', 
    help='the path to save the models',
    required = False
)

@click.option(
    '--encoder-path', '-ep', 
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default='models/1encoder.pt', 
    help='if you want to use a pretrained model those are the path to update',
    required = False
)

@click.option(
    '--decoder-path', '-dp', 
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default='models/1decoder.pt', 
    help='if you want to use a pretrained model those are the path to update',
    required = False
)

# nbRun = 2
# nbEpoch = 2
# batchSize = 128
# learningRate=10e-3
# #the proportion of similar items in rule with the same consequents
# likeness = 0.5
# #The number of rule per consequent
# numberOfRules = 2
# #The maximum number of antecedents
# nbAntecedents = 2
# #For Fp-Growth
# minSupp = 0.01
# minConf = 0.01


# datasetName = 'nursery'


# ARMAEResultsPath= 'Results/NN/'
# exhaustiveResultsPath = 'Results/Exhaustive/'
# overallResultsPath = 'Results/Final/'
# dataPath ='data/'+datasetName+'.csv'



# isLoadedModel = False

# #the path to save the models
# modelPath = 'models/'

# #if you want to use a pretrained model those are the path to update
# encoderPath = 'models/1encoder.pt'
# decoderPath = 'models/1decoder.pt'



def cli(input_path,exhaustive_path,ARMAE_results_path,
        overall_results_path,
        nbRun,nbEpoch,batchSize,
        learningRate,likeness,numberOfRules,
        nbAntecedents,minSupp,minConf,
        is_Loaded_Model,model_path,encoder_path,
        decoder_path):
    
    datasetName = input_path.split('/')[-1].split('.')[0]
    data = pd.read_csv(input_path,index_col=0,dtype=float,header=0)
    times = []

    columns = ['FpExecTime','FpTransformRuleTime','FpTotalTime','FpSupportAvg','FpConfAvg','nbNotNull','nnAvgSupp',
            'nnAvgConf','nnFindInFp','FpFindInnn','timeTraining','timeCreatingRule','timeComputingMeasure','iter']
    scores = []





    dataSize = len(data.loc[0])
    bm = BenchmarkManager('Results/', dataSize)
    for i in range(nbRun):

        NN = ARMAE(dataSize,maxEpoch=nbEpoch,batchSize=batchSize,learningRate=learningRate, likeness=likeness)

        dataLoader = NN.dataPretraitement(data)
        t1 = time.time()
        if not is_Loaded_Model:
            NN.train(dataLoader,model_path)
        else:
            NN.load(encoder_path,decoder_path)
        t2 = time.time()

        timeCreatingRule, timeComputingMeasure = NN.generateRules(data, numberOfRules=numberOfRules, nbAntecedent=nbAntecedents,
                                                        path=ARMAE_results_path + datasetName + str(i)+'.csv')
        timeTraining = t2-t1
        times.append([timeTraining,timeCreatingRule,timeComputingMeasure])

        firstScore = bm.CompareToExhaustive(data, exhaustive_path + datasetName + str(i) + '.csv', minSupp, minConf,
                                                ARMAE_results_path + datasetName + str(i) + '.csv', i,
                                                nbAntecedent=nbAntecedents, dataset=datasetName)

        scoresIter = copy.deepcopy(firstScore)
        scoresIter += [timeCreatingRule,timeComputingMeasure,timeTraining,i]
        scores.append(scoresIter)

    df = pd.DataFrame(scores,columns=columns)
    df.to_csv(overall_results_path+datasetName+'.csv')

if __name__ == '__main__':
    cli()




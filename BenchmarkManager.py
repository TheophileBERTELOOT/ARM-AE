from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import json


from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.frequent_patterns import association_rules
import time

from ARMAE import ARMAE


class BenchmarkManager():
    def __init__(self,path,dataSize,IM=['support','confidence']):
        self.folderPath = path
        self.dataSize = dataSize
        self.algos = []
        self.IM=IM
        self.globalDiffAvgs = []
        self.exhauRules = []

    def findFiles(self):
        fichiers = [f for f in listdir(self.folderPath) if isfile(join(self.folderPath, f))]
        return fichiers

    def ExhaustiveSearch(self,df, path_laws, min_supp,min_conf,nbAntecedent=1,dataset='Mushroom',nbRules=2):
        t1 = time.time()
        print(nbAntecedent)
        frequent_itemsets = fpgrowth(df, min_support=min_supp,max_len=nbAntecedent+1)
        print('fini avec les itemsets')

        self.exhauRules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
        print(self.exhauRules)
        nbLoisInit = str(len(self.exhauRules))
        print(nbLoisInit)
        t2 = time.time()
        self.exhauRules = self.exhauRules[self.exhauRules.consequents.apply(lambda x: len(x) == 1)]
        self.exhauRules = self.exhauRules.reset_index()
        self.exhauRules = self.exhauRules.sort_values(by=['support'], ignore_index=True, ascending=False)

        temp = []
        nbRulesFind =  [[0 for _ in range(len(df.loc[0]))] for _ in range(nbRules)]
        for ruleIndex in range(len(self.exhauRules)):
            rule = self.exhauRules.loc[ruleIndex]

            if len(rule['consequents'])==1 and  nbRulesFind[len(list(rule['antecedents']))-1][list(rule['consequents'])[0]]<nbRules:
                temp.append({'antecedent':sorted(list(rule['antecedents'])),
                             'consequent':list(rule['consequents']),
                             'support':round(rule['support'],3),
                             'confiance':round(rule['confidence'],3)})
                nbRulesFind[len(list(rule['antecedents']))-1][list(rule['consequents'])[0]]+=1

        self.exhauRules = pd.DataFrame(temp)
        print(self.exhauRules)
        t3 = time.time()
        ttNbRules = sum([sum(x) for x in nbRulesFind])
        execTime = t2-t1
        transformRulesTime = t3-t2
        totalTime = t3-t1
        supportAvg = np.mean(self.exhauRules['support'])
        confAvg = np.mean(self.exhauRules['confiance'])
        resPropsData = [execTime,transformRulesTime,totalTime,supportAvg,confAvg]
        print(resPropsData)
        resProps = pd.DataFrame([resPropsData],columns=['execTime','transformRulesTime','totalTime','supportAvg','confAvg'])
        resProps.to_csv('Results/ExecutionTime/'+dataset+'_exhau.csv')
        print('support average')
        print(supportAvg)
        print('confiance average')
        print(confAvg)
        self.exhauRules.to_csv(path_laws)
        print(nbRulesFind)
        return [execTime,transformRulesTime,totalTime,supportAvg,confAvg]


    def ruleIsInRulesDf(self,antecedent,consequent,rulesDf):


        for ruleIndex in range(len(rulesDf)):
            testRule = rulesDf.loc[ruleIndex]
            if str(antecedent) == str(testRule['antecedent']) and str(consequent) == str(testRule['consequent']):
                return True
        return False


    def CompareToExhaustive(self,df,pathRules,min_supp,min_conf,resultPath,iteration,threshold=0.1,nbAntecedent=1,dataset='Mushroom',):
        nnR = pd.read_csv(resultPath)
        nbNotNull = len(nnR[nnR['support']>0])
        nnAvgSupp =  np.mean(nnR['support'])
        nnAvgConf = np.mean(nnR['confidence'])
        percentageNotNull = round(nbNotNull/len(nnR),2)
        print(nnR)
        print('nbNotNull')
        print(nbNotNull)
        print('support average ARM-AE')
        print(nnAvgSupp)
        print('confiance average ARM-AE')
        print(nnAvgConf)
        print('there is {percent} % of rules with a support greater than {threshold}'.format(percent=percentageNotNull,threshold={0}))
        results = self.ExhaustiveSearch(df,pathRules,min_supp,min_conf,nbAntecedent=nbAntecedent,dataset=dataset)
        nbSearch = len(self.exhauRules)
        print('nb lois a trouver :'+str(nbSearch))
        nbFoundRecall = 0
        for ruleIndex in range(len(self.exhauRules)):
            rule = self.exhauRules.loc[ruleIndex]
            antecedent =rule['antecedent']
            consequent = rule['consequent']
            isPresentRecall = self.ruleIsInRulesDf(antecedent,consequent,nnR)
            if isPresentRecall:
                nbFoundRecall+=1

        nbFoundAccu = 0
        for ruleIndex in range(len(nnR)):
            rule = nnR.loc[ruleIndex]
            antecedent = rule['antecedent']
            consequent = rule['consequent']
            isPresentAccu = self.ruleIsInRulesDf(antecedent,consequent,self.exhauRules)

            if isPresentAccu:
                nbFoundAccu += 1

        nnFindInFp = round(nbFoundRecall/nbSearch,2)
        FpFindInnn = round(nbFoundAccu / len(nnR), 2)
        print('with min_supp = {min_supp} and min_conf = {min_conf} and {nbRules} proposed'.format(min_supp=min_supp,min_conf=min_conf,nbRules=nbSearch))
        print('Percentage of the best rules found with NN : {}'.format(round(nbFoundRecall/nbSearch,2)))
        print('Percentage of the rules found with NN in the top rules : {}'.format(round(nbFoundAccu / len(nnR), 2)))
        results+=[nbNotNull,nnAvgSupp,nnAvgConf,nnFindInFp,FpFindInnn]
        return results

    def RechercheGrilleLearningRateEpochs(self,data):
        dataSize = len(data.loc[0])
        scores = []
        arrayScore = []
        lrRange = [1* 10**-lrExpo for lrExpo in range(1,6)]
        nbEpochRange = [nbEpoch for nbEpoch in range(1,110,10)]
        for lr  in lrRange:


            print(lr)
            for nbEpoch in nbEpochRange:
                row = []
                NN = ARMAE(dataSize, False, False, False, False, False, False, True, numEpoch=nbEpoch, output='tanh',
                                   learningRate=lr)
                dataLoader = NN.dataPretraitement(data)
                NN.train(dataLoader)
                averageResult = NN.generateRules(data, numberOfRules=2, nbAntecedent=5)
                scores.append([lr,nbEpoch,averageResult])
                row.append(averageResult)
            arrayScore.append(row)
        df = pd.DataFrame(scores,columns=['lr','nbEpoch','score'])
        print(df)
        result = df.pivot(index='lr', columns='nbEpoch', values='score')
        print(result)
        p = sns.heatmap(result, annot=True, fmt="g",cmap='viridis')
        p.set_title("average score function of lr and nbEpoch")
        plt.show()




















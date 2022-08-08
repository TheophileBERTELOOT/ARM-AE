from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.frequent_patterns import association_rules
import time

from src.NeuralNetwork import NeuralNetwork


class BenchmarkManager():
    def __init__(self,path,dataSize):
        self.folderPath = path
        self.dataSize = dataSize
        self.algos = []
        self.globalDiffAvgs = []
        self.exhauRules = []

    def findFiles(self):
        fichiers = [f for f in listdir(self.folderPath) if isfile(join(self.folderPath, f))]
        return fichiers

    def computeScores(self):
        fichiers = self.findFiles()
        nbAlgo = len(fichiers)-1
        scores = []
        for fichier in fichiers:
            if fichier != 'report.txt':
                scoreInstance = []
                f = open(self.folderPath+fichier, 'r')
                results = f.readlines()
                f.close()
                for line in results:
                    scoreInstance.append(json.loads(line))
                scores.append(scoreInstance)
                self.algos.append(fichier[:-4])

        scoreAverage = []
        for consequent in range(self.dataSize):
            scoresAverageInstance = []
            for algoIndex in range(nbAlgo):
                avg = [0,0,0]
                nbRuleConsequent = 0
                for rule in scores[algoIndex]:
                    if rule['consequent'] == consequent:
                        avg[0]+= rule['support']
                        avg[1]+= rule['confidence']
                        avg[2]+= rule['cosine']
                        nbRuleConsequent+=1
                if nbRuleConsequent == 0:
                    nbRuleConsequent =1
                avg= [round(a/nbRuleConsequent,2) for a in avg]
                scoresAverageInstance.append(avg)
            scoreAverage.append(scoresAverageInstance)

        differenceAvgs = np.zeros((nbAlgo,nbAlgo))
        for algo1Index in range(nbAlgo):
            for algo2Index in range(nbAlgo):
                if algo1Index != algo2Index:
                    globalDiffAvg = 0
                    for consequent in range(self.dataSize):
                        algo1Score = scoreAverage[consequent][algo1Index]
                        algo2Score = scoreAverage[consequent][algo2Index]
                        suppDiff = algo1Score[0]-algo2Score[0]
                        confDiff = algo1Score[1] - algo2Score[1]
                        cosDiff = algo1Score[2] - algo2Score[2]
                        diffAvg = round((suppDiff+confDiff+cosDiff)/3,2)
                        globalDiffAvg+=diffAvg
                    globalDiffAvg/=self.dataSize
                    differenceAvgs[algo1Index][algo2Index] = globalDiffAvg
        self.globalDiffAvgs = differenceAvgs
        print(self.globalDiffAvgs)
        f = open(self.folderPath+'report.txt','w')
        f.write(str(self.globalDiffAvgs))
        f.close()

    def printHeatMap(self):
        p = sns.heatmap(self.globalDiffAvgs, annot=True,xticklabels=self.algos, yticklabels=self.algos)
        p.set_title("Average Difference between average score for each rule")
        plt.show()

    def ExhaustiveSearch(self,df, path_laws, min_supp,min_conf,nbAntecedent=1,dataset='Mushroom',nbRules=2):
        # df = pd.read_csv('Data/synth_transformed.csv')
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

            if len(rule['consequents'])==1 and nbRulesFind[len(list(rule['antecedents']))-1][list(rule['consequents'])[0]]<nbRules:
                temp.append({'antecedent':sorted(list(rule['antecedents'])),
                             'consequent':list(rule['consequents']),
                             'support':round(rule['support'],2),
                             'confiance':round(rule['confidence'],2)})
                nbRulesFind[len(list(rule['antecedents']))-1][list(rule['consequents'])[0]]+=1
        self.exhauRules = pd.DataFrame(temp)
        print(self.exhauRules)
        t3 = time.time()
        ttNbRules = sum([sum(x) for x in nbRulesFind])
        file = open('Results/ExecutionTime/'+dataset+'.txt', "a")
        file.write('Recherche exhaustive exec time : ' + str(t2 - t1) + '\n'
                    +'Transformation des lois exec time: ' + str(t3 - t2) + '\n'
                   + 'Total  exec time: ' + str(t3 - t1) + '\n'
                   +'Nombre de lois trouvees : '+nbLoisInit+'\n'
                   +'Support FP : '+str(np.mean(self.exhauRules['support'])) +'\n'
                   + 'confiance FP : '+str(np.mean(self.exhauRules['confiance']))+'\n'
                   +'Nombre de lois regulieres trouvees : ' + str(ttNbRules) + '\n')
        file.close()
        print('support average')
        print(np.mean(self.exhauRules['support']))
        print('confiance average')
        print(np.mean(self.exhauRules['confiance']))
        self.exhauRules.to_csv(path_laws)
        print(nbRulesFind)


    def ruleIsInRulesDf(self,antecedent,consequent,rulesDf):
        if '[' in consequent:
            consequent = int(consequent.replace('[', '').replace(']', ''))
        if '[' in antecedent:
            antecedent = antecedent.replace('[', '').replace(']', '')
            antecedent = antecedent.split(',')
            antecedent = [ int(x) for x in antecedent]
        for ruleIndex in range(len(rulesDf)):
            testRule = rulesDf.loc[ruleIndex]

            if antecedent == testRule['antecedent'] and consequent == testRule['consequent']:
                return True
        return False


    def CompareToExhaustive(self,df,pathRules,min_supp,min_conf,resultPath,iteration,threshold=0.1,nbAntecedent=1,dataset='Mushroom',):
        f = open(resultPath, 'r')
        res = f.readlines()
        f.close()
        score = []
        for line in res:
            score.append(json.loads(line))
        nnR = pd.DataFrame(score)
        nbNotNull = len(nnR[nnR['support']>0])
        file = open('Results/ExecutionTime/' + dataset + '.txt', "a")
        file.write('nbNotNull : ' + str(nbNotNull) + '\n'
                   + 'support average ARM-AE: ' + str(np.mean(nnR['support'])) + '\n'
                   + 'confiance average ARM-AE: ' + str(np.mean(nnR['confidence'])) + '\n'
                   + 'there is  rules with a support greater than : ' + str(round(nbNotNull/len(nnR),2)) + '\n')
        file.close()
        print('nbNotNull')
        print(nbNotNull)
        print('support average ARM-AE')
        print(np.mean(nnR['support']))
        print('confiance average ARM-AE')
        print(np.mean(nnR['confidence']))
        print('there is {percent} % of rules with a support greater than {threshold}'.format(percent=round(nbNotNull/len(nnR),2),threshold={0}))
        if iteration == 0:
            self.ExhaustiveSearch(df,pathRules,min_supp,min_conf,nbAntecedent=nbAntecedent,dataset=dataset)
        else:
            self.exhauRules = pd.read_csv(pathRules,index_col=0)
            print(self.exhauRules)
        f = open(resultPath,'r')
        res = f.readlines()
        f.close()
        score = []
        for line in res:
            score.append(json.loads(line))
        nnR = pd.DataFrame(score)
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
            consequent = [rule['consequent']]
            isPresentAccu = self.ruleIsInRulesDf(antecedent,consequent,self.exhauRules)

            if isPresentAccu:
                nbFoundAccu += 1


        file = open('Results/ExecutionTime/' + dataset + '.txt', "a")
        file.write('with min_supp = {min_supp} and min_conf = {min_conf} and {nbRules} proposed'.format(min_supp=min_supp,min_conf=min_conf,nbRules=nbSearch)+ '\n'
                   + 'Percentage of the 100 best rules found with NN : {}'.format(round(nbFoundRecall/nbSearch,2)) + '\n'
                   + 'Percentage of the rules found with NN in the top 500 : {}'.format(round(nbFoundAccu / len(nnR), 2))+ '\n'
                   )
        file.close()
        print('with min_supp = {min_supp} and min_conf = {min_conf} and {nbRules} proposed'.format(min_supp=min_supp,min_conf=min_conf,nbRules=nbSearch))
        print('Percentage of the 100 best rules found with NN : {}'.format(round(nbFoundRecall/nbSearch,2)))
        print('Percentage of the rules found with NN in the top 500 : {}'.format(round(nbFoundAccu / len(nnR), 2)))

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
                NN = NeuralNetwork(dataSize, False, False, False, False, False, False, True, numEpoch=nbEpoch, output='tanh',
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




















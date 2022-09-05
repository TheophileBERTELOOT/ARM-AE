import numpy as np
import pandas as pd
import torch
from src.AutoEncoder import *

import copy
import time

class NeuralNetwork:
    def __init__(self,dataSize,baseline,lstm,gru,vae,conv,conv2D,recursiveModel,learningRate=1e-3,numEpoch=10,batchSize=128,dropout=0.5,hiddenSize='dataSize',output='relu',likeness=0.4,isPolypharmacy = False,con=None,columns=[]):
        self.baseline = baseline
        self.lstm = lstm
        self.gru = gru
        self.vae = vae
        self.conv = conv
        self.conv2D = conv2D
        self.con = con
        self.recursiveModel = recursiveModel
        self.dataSize = dataSize
        self.learningRate = learningRate
        self.likeness = likeness
        self.hiddenSize = hiddenSize
        self.dropout = dropout
        self.columns = columns
        self.output = output
        self.isPolypharmacy = isPolypharmacy
        if self.hiddenSize == 'dataSize':
            self.hiddenSize = self.dataSize
        if self.recursiveModel:
            self.baseline = True
        self.numEpoch = numEpoch
        self.batchSize = batchSize
        self.model = AutoEncoder(self.dataSize,self.baseline,self.lstm,self.gru,self.vae,self.conv,self.conv2D,self.dropout,self.hiddenSize,self.output).cuda()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learningRate)
        self.results = []


    def dataPretraitement(self,d):
        self.columns = d.columns
        trainTensor = torch.tensor(d.values)
        dataLoader = DataLoader(trainTensor.float(), batch_size=self.batchSize, shuffle=True)
        # cv2.imshow('dataset', d.values)
        # cv2.waitKey(0)
        x = torch.tensor([float('nan'), float('inf'), -float('inf'), 3.14])
        torch.nan_to_num(x, nan=0.0, posinf=0.0)
        return dataLoader

    def save(self,p):
        self.model.save(p)
    def load(self,p):
        self.model.load(p)
    def train(self,dataLoader):
        showImage = True
        for epoch in range(self.numEpoch):
            for data in dataLoader:
                if len(data) == self.batchSize:
                    d = Variable(data).cuda()
                    if self.conv2D:
                        if showImage:
                            cv2.imshow('batch', np.array(data))
                            showImage = False
                        d = d.reshape((1,1,self.dataSize,self.batchSize))
                    # ===================forward=====================
                    output = self.model.forward(d)

                    loss = self.criterion(output[0], d)

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # ===================log========================
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, self.numEpoch, loss.data))

    def computeMeasures(self,antecedent,consequent):
        datasize = 298167971
        # datasize = 50000
        data = pd.read_sql_query("SELECT * from POLY LIMIT 1", self.con)
        data = data.drop(['AGE_AT_DEATH', 'AGE_AT_HOSPIT', 'AGE_AT_START', 'AGE_AT_END', 'ID_INDV8', 'DECES', 'ROWID'],
                         axis=1)
        searchColumns = [consequent] + antecedent
        searchColumnsNames = data.columns[searchColumns]
        requestLineSupp = ' '.join(['"'+searchColumnsNames[k] + '"'+" = '1' AND " for k in range(len(searchColumns))])
        requestLineSupp = requestLineSupp[:-4]
        antecedentNames = data.columns[antecedent]
        requestLineAnt = ' '.join(['"'+antecedentNames[k] + '"'+" = '1' AND " for k in range(len(antecedent))])
        requestLineAnt = requestLineAnt[:-4]
        consequentNames = data.columns[consequent]
        requestLinePNonANonC = ' '.join(['"'+searchColumnsNames[k] + '"'+" = '0' AND " for k in range(len(searchColumns))])
        requestLinePNonANonC = requestLinePNonANonC[:-4]
        requestLinePANonC = ' '.join(['"' + antecedentNames[k] + '"' + " = '1' AND " for k in range(len(antecedent))])
        requestLinePANonC += '"HOSPIT" = '+"'0'"
        requestLinePNonAC = ' '.join(['"' + antecedentNames[k] + '"' + " = '0' AND " for k in range(len(antecedent))])
        requestLinePNonAC += '"HOSPIT"= '+"'1'"

        print('rules : ')
        print('antecedent : '+str(list(self.columns[antecedent])))
        print('consequent : '+str(self.columns[consequent]))

        PC = 8695061



        print(requestLineSupp)
        PAC = pd.read_sql_query('SELECT COUNT(*) from POLY  WHERE '+requestLineSupp,self.con)
        PAC = int(PAC.loc[0])
        print(PAC)

        print(requestLineAnt)
        PA = pd.read_sql_query('SELECT COUNT(*) from POLY  WHERE ' + requestLineAnt, self.con)
        PA = int(PA.loc[0])
        PNonA = datasize-PA
        print(PA)

        PNonANonC = pd.read_sql_query('SELECT COUNT(*) from POLY  WHERE ' + requestLinePNonANonC, self.con)
        PNonANonC = int(PNonANonC.loc[0])

        PANonC = pd.read_sql_query('SELECT COUNT(*) from POLY  WHERE ' + requestLinePANonC, self.con)
        PANonC = int(PANonC.loc[0])

        PNonAC = pd.read_sql_query('SELECT COUNT(*) from POLY  WHERE ' + requestLinePNonAC, self.con)
        PNonAC = int(PNonAC.loc[0])

        print('divided by datasize : ')
        PA/=datasize
        PNonA/=datasize
        print(PA)
        PAC/=datasize
        print(PAC)
        PC/=datasize
        print(PC)
        PNonANonC /= datasize
        print(PNonANonC)
        PANonC /= datasize
        print(PANonC)
        PNonAC /= datasize
        print(PNonAC)

        if PA == 0:
            conf = 0
        else:
            conf = PAC / PA

        if PA == 0 or PC==0:
            cos = 0
        else:
            cos = PAC / np.sqrt(PC * PA)

        yule = (PAC*PNonANonC - PANonC*PNonAC)/(PAC*PNonANonC + PANonC*PNonAC)

        relativeRisk = (PAC/PA)/(PNonAC/PNonA)

        return PAC,conf,cos,yule,relativeRisk



    def findMostPresentItem(self,data):
        items = [0 for _ in range(len(data.columns))]
        for i in data.index:
            row = data.loc[i]
            for j in range(len(items)):
                if row[j] == 1:
                    items[j]+=1
        items = pd.Series(items)
        mostPresent = items.nlargest(int(0.02 * len(items)))
        mostPresentIndex = mostPresent.index.values.tolist()
        print(items)
        print(mostPresent)
        print(mostPresentIndex)
        return mostPresentIndex


    def computeSimilarity(self, allAntecedents, antecedentsArray,nbantecedent):
        onlySameSize = [x for x in allAntecedents if len(x) >= len(antecedentsArray)]
        maxSimilarity = 0
        print('allAntecedents')
        print(allAntecedents)
        for antecedentIndex in range(len(onlySameSize)):
            antecedents = onlySameSize[antecedentIndex]
            similarity = 0
            print('antecedents')
            print(antecedents)
            print('antecedentsArray')
            print(antecedentsArray)
            for item in antecedents:
                if item in antecedentsArray:
                    similarity+=1
            similarity /= nbantecedent
            if similarity > maxSimilarity:
                maxSimilarity = similarity
        print(maxSimilarity)
        return maxSimilarity

    def generateRules(self,data,numberOfRules = 2,nbAntecedent=2,outcomeIndex=473,path = ''):
        print('begin rules generation')
        timeCreatingRule = 0
        timeComputingMeasure = 0
        nbConsequent = self.dataSize
        file = open(path, "a")
        if self.isPolypharmacy:
            nbConsequent = 1
            
        for consequent in range(nbConsequent):
            if self.isPolypharmacy:
                consequent = outcomeIndex
            if consequent%10==0 :
                print('progress : '+str(round(consequent/self.dataSize,2))+' %')
            consequentArray = np.zeros(self.dataSize)

            if self.isPolypharmacy:
                consequentArray[outcomeIndex] = 1
            else:
                consequentArray[consequent] = 1
            consequentArray = torch.tensor(consequentArray).cuda()
            consequentArray = consequentArray.unsqueeze(0)
            output = self.model(consequentArray.float())
            output = output.cpu()
            output = np.array(output.detach().numpy())
            output = pd.DataFrame(output.reshape(self.dataSize, -1))
            output = pd.DataFrame(output)

            t1 = time.time()
            if self.recursiveModel:
                allAntecedents = []
                for j in range(numberOfRules):
                    if j % 10 == 0 and self.isPolypharmacy:
                        print('progress : ' + str(round(j / numberOfRules, 2)) + ' %')
                    antecedentsArray = []
                    for i in range(nbAntecedent):
                        consequentArray = np.zeros(self.dataSize)
                        consequentArray[consequent] = 1
                        consequentArray[antecedentsArray] = 1
                        consequentArray = torch.tensor(consequentArray).cuda()
                        consequentArray = consequentArray.unsqueeze(0)
                        output = self.model(consequentArray.float())
                        output = output.cpu()
                        output = np.array(output.detach().numpy())
                        output = pd.DataFrame(output.reshape(self.dataSize, -1))
                        potentialAntecedentsArray = output[0].nlargest(len(data.loc[0]))
                        for antecedent in potentialAntecedentsArray.keys():
                            potentialAntecedents = copy.deepcopy(antecedentsArray)
                            potentialAntecedents.append(antecedent)
                            potentialAntecedents = sorted(potentialAntecedents)
                            if antecedent != consequent and antecedent not in antecedentsArray and self.computeSimilarity(allAntecedents,potentialAntecedents,nbAntecedent) <= self.likeness:
                                    antecedentsArray.append(antecedent)
                                    break

                        t3 = time.time()

                        support, confidence, cosine, yule ,RR = self.computeMeasures( copy.deepcopy(antecedentsArray), consequent)
                        t4 = time.time()
                        timeComputingMeasure += t4-t3
                        if np.isnan(support):
                            support = 0
                        if np.isnan(confidence):
                            confidence = 0
                        if np.isnan(cosine):
                            cosine = 0
                        if np.isnan(yule):
                            yule = 0
                        if np.isnan(RR):
                            RR = 0
                        if self.isPolypharmacy:
                            self.results.append({
                                'antecedent': list(self.columns[sorted(copy.deepcopy(antecedentsArray))]),
                                'consequent': self.columns[consequent],
                                'support': support,
                                'confidence': confidence,
                                'cosine': cosine,
                                'yule':yule,
                                'RR':RR
                            })
                            lineStr = str({
                                'antecedent': list(self.columns[sorted(copy.deepcopy(antecedentsArray))]),
                                'consequent': self.columns[consequent],
                                'support': support,
                                'confidence': confidence,
                                'cosine': cosine,
                                'yule': yule,
                                'RR':RR
                            })
                            print(lineStr)
                            json_acceptable_string = lineStr.replace("'", "\"")
                            file.write(json_acceptable_string)
                            file.write('\n')

                            allAntecedents.append(sorted(copy.deepcopy(antecedentsArray)),)
            t2 = time.time()
            timeCreatingRule += t2 - t1
        timeCreatingRule -= timeComputingMeasure
        file.close()
        return timeCreatingRule,timeComputingMeasure

    def generateReport(self,path):
        file = open(path, "w")
        for line in self.results:
            lineStr = str(line)
            json_acceptable_string = lineStr.replace("'", "\"")
            file.write(json_acceptable_string)
            file.write('\n')
        file.close()






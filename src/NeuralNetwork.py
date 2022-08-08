import numpy as np
import pandas as pd
import torch
from src.AutoEncoder import *
import cv2
import copy
import time

class NeuralNetwork:
    def __init__(self,dataSize,baseline,lstm,gru,vae,conv,conv2D,recursiveModel,learningRate=1e-3,numEpoch=10,batchSize=128,dropout=0.5,hiddenSize='dataSize',output='relu',likeness=0.4,isPolypharmacy = False):
        self.baseline = baseline
        self.lstm = lstm
        self.gru = gru
        self.vae = vae
        self.conv = conv
        self.conv2D = conv2D
        self.recursiveModel = recursiveModel
        self.dataSize = dataSize
        self.learningRate = learningRate
        self.likeness = likeness
        self.hiddenSize = hiddenSize
        self.dropout = dropout
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
            self.model.parameters(), lr=self.learningRate, weight_decay=1e-5)
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

    def computeMeasures(self,data,antecedent,consequent):
        searchColumns = [consequent]+antecedent
        antD = data[data.columns[antecedent]]
        antD = sum([sum(antD.loc[x]) == len(antecedent) for x in range(len(antD))]) / len(antD)
        suppD = data[data.columns[searchColumns]]
        suppD = sum([sum(suppD.loc[x]) == len(searchColumns) for x in range(len(suppD))]) / len(suppD)

        if antD == 0:
            confD = 0
        else:
            confD = suppD / antD
        cons = (data[data.columns[[consequent]]].sum()[0] / len(data))
        if antD == 0 or cons==0:
            cosD = 0
        else:
            cosD = suppD / np.sqrt(cons * antD)
        return suppD,confD,cosD



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


    def computeSimilarity(self, allAntecedents, antecedentsArray):
        onlySameSize = [x for x in allAntecedents if len(x) >= len(antecedentsArray)]
        maxSimilarity = 0
        for antecedentIndex in range(len(onlySameSize)):
            antecedents = onlySameSize[antecedentIndex]
            similarity = 0
            for item in antecedents:
                if item in antecedentsArray:
                    similarity+=1
            similarity /= len(antecedentsArray)
            if similarity > maxSimilarity:
                maxSimilarity = similarity
        return maxSimilarity

    def generateRules(self,data,numberOfRules = 2,nbAntecedent=2,outcomeIndex=473):
        print('begin rules generation')
        timeCreatingRule = 0
        timeComputingMeasure = 0
        nbConsequent = self.dataSize
        if self.isPolypharmacy:
            nbConsequent = 1
        for consequent in range(nbConsequent):

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
                            if antecedent != consequent and antecedent not in antecedentsArray and self.computeSimilarity(allAntecedents,potentialAntecedents) <= self.likeness:
                                    antecedentsArray.append(antecedent)
                                    break

                        t3 = time.time()
                        if not self.isPolypharmacy or (self.isPolypharmacy and len(antecedentsArray ==nbAntecedent)):
                            support, confidence, cosine = self.computeMeasures(data, copy.deepcopy(antecedentsArray), consequent)
                            t4 = time.time()
                            timeComputingMeasure += t4-t3
                            if np.isnan(support):
                                support = 0
                            if np.isnan(confidence):
                                confidence = 0
                            if np.isnan(cosine):
                                cosine = 0
                            if self.isPolypharmacy:
                                self.results.append({
                                    'antecedent': self.columns[sorted(copy.deepcopy(antecedentsArray))],
                                    'consequent': self.columns[consequent],
                                    'support': support,
                                    'confidence': confidence,
                                    'cosine': round(cosine, 2)
                                })
                            else:
                                self.results.append({
                                    'antecedent': sorted(copy.deepcopy(antecedentsArray)),
                                    'consequent': consequent,
                                    'support': support,
                                    'confidence': confidence,
                                    'cosine': round(cosine, 2)
                                })
                            allAntecedents.append(sorted(copy.deepcopy(antecedentsArray)),)
            t2 = time.time()
            timeCreatingRule += t2 - t1
        timeCreatingRule -= timeComputingMeasure
        return timeCreatingRule,timeComputingMeasure

    def generateReport(self,path):
        file = open(path, "w")
        for line in self.results:
            lineStr = str(line)
            json_acceptable_string = lineStr.replace("'", "\"")
            file.write(json_acceptable_string)
            file.write('\n')
        file.close()






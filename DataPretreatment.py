import pandas as pd
import numpy as np
import random as rd
from sklearn.manifold import TSNE


#AS an example :
#d = Data('./Data/Raw/chess.data',separator=',')
#d.TransformToHorizontalBinary()
#d.Save('./Data/Transform/chess.csv')

#For some other datasets :
# TransformRowsDSToHorizontalBinary('./Data/Raw/plants.data','./Data/Transform/plants.csv')

class Data:
    def __init__(self,path='',header=None,indexCol=None,nbSample = 5,artificial=False,nbRow=2000,nbItem=50,separator=','):

        self.artificial = artificial
        self.nbRow=nbRow
        self.nbItem = nbItem
        self.nbSample = nbSample
        self.labels = []
        if self.artificial:
            self.GenerateArtificialData()
        else:
            self.data = pd.read_csv(path, header=header, index_col=indexCol,sep=separator)
            print(self.data)
            print('nbColonnes :'+str(len(self.data.columns)))
            print('nbLignes :' + str(len(self.data)))


    def GenerateArtificialData(self):
        data = []
        for i in range(self.nbRow):
            row = []
            for j in range(self.nbItem):
                row.append(rd.randint(0,1))
            data.append(row)
        self.data = pd.DataFrame(np.array(data))

    def isListFullOfDigit(self,l):
        if(l.dtype == np.str_ or l.dtype==np.object_):
            for i in range(len(l)):
                if not l[i].lstrip('-').replace('.','',1).isdigit() and l[i] != '?' and l[i] != '-':
                    return False
        return True

    def RemoveRowWithMissingValue(self):
        indexWithMissingValues = self.data[(self.data == '?').any(axis=1)].index
        self.data = self.data.drop(indexWithMissingValues)

    def TransformToHorizontalBinary(self):
        self.RemoveRowWithMissingValue()
        transformed = []
        for col in self.data.columns:
            possibleValues = self.data[col].unique()
            if len(possibleValues)>20 and  self.isListFullOfDigit(possibleValues) :
                possibleValues = self.Sampling(col)
                self.labels+=list(possibleValues)
            else:
                self.labels+=list(possibleValues)
            binaryCols = [[] for i in range(len(possibleValues))]
            for index,row in self.data.iterrows():
                value = np.where( possibleValues == row[col])[0][0]
                for i in range(len(binaryCols)):
                    if i  == value:
                        binaryCols[i].append(1)
                    else:
                        binaryCols[i].append(0)
            transformed+=binaryCols

        transformed = np.array(transformed,dtype=int).T
        self.data = pd.DataFrame(transformed,columns=self.labels)

    def TransformRowsDSToHorizontalBinary(self,p, fp):
        with open(p) as f:
            lines = f.readlines()
            splitLines = []
            columns = ['name']
            ds = []
            for line in lines:
                splitLine = line.split(',')
                attributes = splitLine[1:]
                for attribute in attributes:
                    cleanAttribute = attribute.strip()
                    if not cleanAttribute in columns:
                        columns.append(cleanAttribute)
                splitLines.append(splitLine)
            for line in splitLines:
                row = [0 for _ in range(len(columns))]
                row[0] = line[0]
                for attribute in line[1:]:
                    cleanAttribute = attribute.strip()
                    attributeIndex = columns.index(cleanAttribute)
                    row[attributeIndex] = 1
                ds.append(row)
            df = pd.DataFrame(ds, columns=columns)
            df.drop('name', inplace=True, axis=1)
            print(df)
            df.to_csv(fp)

    def Sampling(self,col):
        self.data[col]=self.data[col].astype(float)
        ma = self.data[col].max()
        mi = self.data[col].min()
        r = ma-mi
        p = (r/self.nbSample)+0.00001
        for index in self.data.index:
            for j in range(1,self.nbSample+1):
                if self.data[col][index]<=mi+j*p:
                    self.data[col][index] = j-1
                    break
        print(self.data[col].unique())
        possibleValues = np.arange(self.nbSample)
        return possibleValues

    def GetSplitATCLvl(self,nbChar):
        uniqueColumns = self.data.columns()
        columns = [col[:nbChar] for col in uniqueColumns]
        columns = list(dict.fromkeys(columns))
        dictColumn = {i:[] for i in range(len(columns))}
        for col in uniqueColumns:
            index = columns.index(col[:nbChar])
            dictColumn[index].append(uniqueColumns.index(col))
        return dictColumn


    def tSneDataView(self):
        embedded = TSNE(n_components=2, learning_rate='auto',
                        init='random').fit_transform(np.asarray(self.data,dtype='float64'))
        return embedded

    def Save(self,path):
        self.data.to_csv(path)

    def ToNumpy(self):
        self.data = self.data.to_numpy(dtype=int)



import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('Results/Final/chess3.csv')
data.sort_index(axis=1)
print(data)

# print('FpFindInnn')
# print(data['FpFindInnn'].mean())
# print('nnAvgSupp')
# print(data['nnAvgSupp'].mean())
# print('nnAvgConf')
# print(data['nnAvgConf'].mean())
# print('FpFindInNSGAII')
# print(data['FpFindInNSGAII'].mean())
# print('avgSuppNSGAII')
# print(data['avgSuppNSGAII'].mean())
# print('avgConfNSGAII')
# print(data['avgConfNSGAII'].mean())
# print('FpSupportAvg')
# print(data['FpSupportAvg'].mean())
# print('FpConfAvg')
# print(data['FpConfAvg'].mean())

print('nnTime')
print(data['timeCreatingRule'].mean() + data['timeComputingMeasure'].mean() +data['timeTraining'].mean())
print('nbNotNull')
print(data['nbNotNull'].mean() )
print('timeNSGAII')
print(data['timeNSGAII'].mean() )
print('NSGAIINbNotNull')
print(data['NSGAIINbNotNull'].mean() )
print('FpTotalTime')
print(data['FpTotalTime'].mean())








data = pd.read_csv('Results/Final/plants_goalLosses.csv')
plot = sns.lineplot(data=data, x="goalLoss", y="nnAvgSupp",ci=None)
data = pd.read_csv('Results/Final/chess_goalLosses.csv')
plot = sns.lineplot(data=data, x="goalLoss", y="nnAvgSupp",ci=None)
data = pd.read_csv('Results/Final/nursery_goalLosses.csv')
plot = sns.lineplot(data=data, x="goalLoss", y="nnAvgSupp",ci=None)
plt.legend(labels=["plants","chess","nursery"])
plt.xlabel('goal loss threshold')
plt.ylabel('average support of rule set')
plot.set(xscale='log')
plt.show()

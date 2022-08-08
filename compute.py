import numpy as np



f = open('./Results/ExecutionTime/plants.txt','r')

lines = f.readlines()
print(lines)
data = []
for line in lines :
   if 'Percentage of the 100 best rules found' in line :
       lineSplit = line.split(':')
       cell = lineSplit[1]
       data.append(round(float(cell),2))
f.close()
print(data)
print(np.average(data))
import pandas as pd

df1 = pd.read_csv('Results/NSGAII/nursery_final.csv')
df2 = pd.read_csv('Results/Final/nursery.csv')

print(df1)
print(df2)
result = pd.concat([df2,df1],axis=1)
print(result)
result.to_csv('Results/Final/nursery3.csv')





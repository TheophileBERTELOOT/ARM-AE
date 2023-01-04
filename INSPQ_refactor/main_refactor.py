import json
import pandas as pd
f = open('data.txt','r')
lines = f.readlines()
f.close()
data = []
columns = []
for line in lines:
    dict = json.loads(line)
    print(dict.keys())
    print(dict.values())
    data.append(dict.values())
    columns = dict.keys()

df = pd.DataFrame(data,columns=columns)
print(df.columns)
print(df[['support','confidence']])

df[['support','confidence']].to_csv('data.csv',sep=';')


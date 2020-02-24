import pandas as pd

file = pd.read_csv('datasets.csv') 


x =  [1,4,6]

y = file.iloc[:, x].values
for i in y :
    print(i)

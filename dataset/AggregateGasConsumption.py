import pandas as pd

data1 = pd.read_csv("2021_to_2023_GC_PastInfo.csv",sep=";")

data2 = pd.read_csv("2018_to_2021_GC.csv")

data2['2022']=data1['2022']

data2.to_csv("../2018_to_2022_GC.csv",index=False)
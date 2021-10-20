from utilities.VGChartz import get_games_data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('data\VGChartz.csv')
df1=pd.read_csv('data\VGChartz1.csv')
df2=pd.read_csv('data\VGChartz2.csv')
df3=pd.read_csv('data\VGChartz3.csv')
df4=pd.read_csv('data\VGChartz4.csv')
df5=pd.read_csv('data\VGChartz5.csv')

dfc=pd.concat([df,df1,df2,df3,df4,df5]).reset_index()
dfc.head(2)
dfc=dfc.drop(['index','Unnamed: 0'], axis=1)
dfc.head()
dfc.dtypes
dfc.info()
#dfc[["a", "b"]] = df[["a", "b"]].apply(pd.to_numeric)


def split_m(name:str):
    a=[]
    for i in range(len(dfc[name])):
        if type(dfc[name][i])== str:
            a.append(float(dfc[name][i].split('m')[0])*10**6)
        else:
            a.append(0)
    dfa = pd.DataFrame(a,columns=[name])
    dfc[name]=dfa[name]



dfc['Release Date']


dfc.head(20).sort_values('Critic Score',ascending=False)

fig = sns.pairplot(dfc)

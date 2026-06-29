
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform, pdist
#dataset = pd.read_excel(r"C:\Users\ASHWIK.ACHARYA\Brillio\Mohana Datla - CRED\python script\KPI Mapping.xlsx",sheet_name= "Report_KPI Mapping")

#Filtering data for Fact tables
if 'Table Type' in dataset.columns:
    dataset = dataset[dataset["Table Type"]!="Dimension Table"]

dataset = dataset[["Domain","Report Name","KPI/Used Columns"]].dropna().reset_index()
dataset = dataset[dataset["Domain"].isin(["Customer Support","Human Resources","Wallet","Product"])]
dataset["Domain report name"] = dataset["Domain"] +" \_/ "+ dataset["Report Name"]
dataset = dataset[["Domain report name","KPI/Used Columns"]].dropna().reset_index()
dataset.drop(["index"],axis=1,inplace=True)

# matrix = pd.get_dummies(dataset['KPI/Used Columns'],sparse = True ,dtype=int)
dataset = pd.concat([dataset, pd.get_dummies(dataset['KPI/Used Columns'],sparse = True ,dtype=int)], axis=1).groupby('Domain report name').sum()

#grp = dataset.groupby('Report Name').sum()
dataset.drop('KPI/Used Columns',axis=1,inplace=True)
dist = pdist(dataset, metric="cosine")
s_dist = squareform(dist)

# Fill diagonals with nulls
# np.fill_diagonal(s_dist, 2)
sim = np.subtract(1, s_dist)
dataset = pd.DataFrame(sim, columns=dataset.index, index=dataset.index)
#dataset = dataset.where(np.triu(np.ones(dataset.shape)).astype(np.bool_))
#dataset.reset_index(inplace=True,drop =True)
"""#Keep only top similar reports
for i in dataset.columns:
    if i!='Domain report name':
        m = max(dataset[i])
        for j in range(len(dataset[i])):
            if not np.isnan(dataset[i][j]) and m != dataset[i][j]:
                dataset.loc[j,i]=0"""

#un_pivot columns
Domain_Dataset = dataset
Domain_Dataset.reset_index(inplace=True,drop =True)
Domain_Dataset['Reports'] = Domain_Dataset.columns
Domain_Dataset = pd.melt(Domain_Dataset, id_vars = 'Reports')
Domain_Dataset.dropna(inplace=True)

dataset = Domain_Dataset[Domain_Dataset['value'] > 0.75]

#splitting domain
dataset['Domain'] = dataset['Reports'].str.split("\_/", n=1, expand=True)[0]
dataset['Report 1'] = dataset['Reports'].str.split("\_/", n=1, expand=True)[1]
dataset['Report 2'] = dataset['Domain report name'].str.split("\_/", n=1, expand=True)[1]
dataset = dataset.drop(['Reports','Domain report name'],axis=1)

#Creating cluster
dataset['Cluster'] = dataset.groupby(['Report 2']).ngroup()

#getting position of dupicates in cluster
clist = []
dataset = dataset.sort_values(by=['Cluster'])
for i in dataset.Cluster.unique():
    df = dataset[dataset['Cluster'] == i ]
    df.reset_index(drop=True, inplace=True) 
    clist.append(list(df['Report 1']))
pos = []
for i in range(0,len(clist)):
    if i in pos:
        continue
    for j in range(i+1,len(clist)):
        if clist[i] == clist[j]:
            pos.append(j)

#removing the duplicates and flagging rows where reports names are same
finaldf = dataset[~dataset['Cluster'].isin(pos)]
finaldf['new'] = np.where((finaldf['Report 1'] == finaldf['Report 2']), 1, 0)
max_scores = finaldf[finaldf['new'] != 1]

#identify unique report1's based on max similarity index
idx = max_scores.groupby('Report 1')['value'].idxmax()
max_scores = max_scores.loc[idx]

#Concat tables
max_scores = pd.concat([finaldf[finaldf['new'] == 1],max_scores], ignore_index=True)

#max_scores.to_csv(r"C:\Users\ASHWIK.ACHARYA\Brillio\Mohana Datla - CRED\python script\finaldf.csv")

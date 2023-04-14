# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 14:15:27 2023

@author: dyans
"""

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import mode
from warnings import filterwarnings
import statistics
filterwarnings('ignore')
#print(data)


df=pd.read_csv(r'C:\Users\dyans\OneDrive\Desktop\newdt2.csv')
ndf=df[["Race","DOB","Gender","Pincode"]].copy()
ndf.loc[ndf["Race"]=="black","Race"]=0
ndf.loc[ndf["Race"]=="white","Race"]=1
ndf.loc[ndf["Gender"]=="male","Gender"]=3
ndf.loc[ndf["Gender"]=="female","Gender"]=4
print(ndf)
problems=["Heart issue","Kidney issue","Brain Issue","Gut Issue","Bladder Issue"]
nparr=[]

Sum_of_squared_distances = []
#print(len(ndf),"this 2")
K = range(1,len(ndf))
for num_clusters in K :
 kmeans = KMeans(n_clusters=num_clusters)
 kmeans.fit(ndf)
 Sum_of_squared_distances.append(kmeans.inertia_)
 nparr.append(kmeans.inertia_)
plt.plot(K,Sum_of_squared_distances,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Sum of squared distances/Inertia') 
plt.title('Elbow Method For Optimal k')
plt.show()

print(nparr)

from kneed import KneeLocator

kn = KneeLocator(range(1,len(nparr)+1),nparr,  curve='convex', direction='decreasing')
print(kn.elbow,"this")

kmeans = KMeans(n_clusters=math.ceil(kn.knee), init='k-means++', random_state=0).fit(ndf)


#%%
import statistics
fdf=pd.DataFrame()

for i in range(0,math.ceil(kn.knee)):
    idx=np.where(kmeans.labels_==i)[0]
    print(idx,"1")
    print(ndf.iloc[idx],"2")
    ntmp=ndf.iloc[idx]
    #print(ntmp,"3")
    chkrs=pd.DataFrame()
    pre1=pd.DataFrame()
    kmeans2 = KMeans(n_clusters=math.ceil(ntmp.shape[0]/4), init='k-means++', random_state=0).fit(ntmp)
    for j in range(0,math.ceil(ntmp.shape[0]/4)):
        idxm=np.where(kmeans2.labels_==j)[0]
        print(idxm,'4')
        print(ntmp.iloc[idxm],'5')
        ngmp=ntmp.iloc[idxm]
        
        
        if(len(ngmp.index)>=4):
            connection_list = []
            for kl in range(len(ngmp)): 
                t=0
                for klm in range(len(ngmp)):
                    intersection = set(ngmp.iloc[kl]).intersection(ngmp.iloc[klm])
                    t=t+len(intersection)
                    #print(i,j,'=',len(intersection))
                connection_list.append(t)
            mkx=connection_list.index(max(connection_list))
            #print(connection_list.index(max(connection_list)))
            ngmp.iloc[:,:]=ngmp.iloc[mkx,:]
            chkrs=chkrs.append(ngmp.iloc[mkx,:])
            print(ngmp)
            
            pre1=pre1.append(ngmp)
        print(chkrs,"chkrs")
        
    for j in range(0,math.ceil(ntmp.shape[0]/4)):
        idxm=np.where(kmeans2.labels_==j)[0]
        print(idxm,'4')
        print(ntmp.iloc[idxm],'5')
        ngmp=ntmp.iloc[idxm]
        
        
        if(len(ngmp.index)<4):
            connection_list = []
            for kl in range(len(chkrs)): 
                t=0
                for klm in range(len(ngmp)):
                    intersection = set(chkrs.iloc[kl]).intersection(ngmp.iloc[klm])
                    t=t+len(intersection)
                    #print(i,j,'=',len(intersection))
                connection_list.append(t)
            if connection_list:
                mkx=connection_list.index(max(connection_list))
                #print(connection_list.index(max(connection_list)))
                ngmp.iloc[:,:]=chkrs.iloc[mkx,:]
                chkrs=chkrs.append(ngmp.iloc[mkx,:])
            else:
                for kl in range(len(ntmp)): 
                    t=0
                    for klm in range(len(ngmp)):
                        intersection = set(ntmp.iloc[kl]).intersection(ngmp.iloc[klm])
                        t=t+len(intersection)
                    connection_list.append(t)
                mkx=connection_list.index(max(connection_list))
                #print(connection_list.index(max(connection_list)))
                ngmp.iloc[:,:]=ntmp.iloc[mkx,:]
                chkrs=chkrs.append(ntmp.iloc[mkx,:])
            print(ngmp,"final")
            
            pre1=pre1.append(ngmp)
    
    print(pre1.index.values)
    pre1["Problems"]=df["Problems"]
    n=pre1["Problems"].nunique()
    # if(len(n)):
    #     for x in n.index:
    #         print(x,n[x])
    # print(n.idxmax(),n[n.idxmax()])
    cngs=pd.Series(pre1.Problems)
    print(n,"LMAOES")
    print(cngs.mode())
    #print(pre1,"pre1")
    xbg=pre1["Problems"].tolist()
    ppj=pre1["Problems"].tolist()
    print(xbg,"OG")
    diff=list(set(problems)-set(xbg))
    print(diff,"DIFF")
    if(len(set(xbg))<4):
        dds=math.floor(len(xbg)/4)
        
        for kkl in range(0,dds):
            kmps=(len(set(xbg)))
            print(kmps)
            for kkm in range(0,4-kmps):
                print(xbg.index(statistics.mode(xbg)))
                xbg[xbg.index(statistics.mode(xbg))]=diff[kkm]
    print(xbg,"NEW")
    pre1["Problems"]=xbg
    print(pre1,"pre1")
    fdf=fdf.append(pre1)
    
fdf=fdf.sort_index(ascending=True)
print(fdf)
print(ndf)
print(fdf.drop_duplicates().shape[0])

#fdf.to_csv(r'C:\Users\dyans\OneDrive\Desktop\dt12.csv')
#ndf.to_csv(r'C:\Users\dyans\OneDrive\Desktop\dt123.csv')
changes=0
for ind in fdf.index:
    
    if(ndf['Race'][ind]!=fdf['Race'][ind]):
        fdf['Race'][ind]="Any Race"
        changes+=1
    if(ndf['DOB'][ind]!=fdf['DOB'][ind]):
        fdf['DOB'][ind]="196*"
        changes+=1
    if(ndf['Gender'][ind]!=fdf['Gender'][ind]):
        fdf['Gender'][ind]="Any Gender"
        changes+=1
    if(ndf['Pincode'][ind]!=fdf['Pincode'][ind]):
        fdf['Pincode'][ind]="214*"
        changes+=1
    if(fdf['Race'][ind]==0):
       fdf['Race'][ind]="Black"
    elif(fdf['Race'][ind]==1):
        fdf['Race'][ind]="White"

    if(fdf['Gender'][ind]==3):
       fdf['Gender'][ind]="Male"
    elif(fdf['Gender'][ind]==4):
        fdf['Gender'][ind]="Female"
print("Total number of changes to elements =",changes)
#print(fdf)
fdf.to_csv(r'C:\Users\dyans\OneDrive\Desktop\dtff2.csv')

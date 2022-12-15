#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 10:06:19 2020

@author: Vishnu Mohan
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

Univ = pd.read_csv("/Volumes/Data/Course Content/DS content/Day Wise_Material/Day 18 Clustering/Data/Universities_Clustering.csv")

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Univ.iloc[:,1:])

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(df_norm)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


kmeans = KMeans(n_clusters=4,random_state=0)
pred_y = kmeans.fit_predict(df_norm)


d=pd.DataFrame(pred_y)
final = Univ.join(d)

kmeans.cluster_centers_

####################################
# import hierarchical clustering libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(df_norm, method='single'))

# create clusters
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'single')
# save clusters for chart
y_hc = hc.fit_predict(df_norm)

Clusters=pd.DataFrame(y_hc,columns=['Clusters'])


final = Univ.join(Clusters)



















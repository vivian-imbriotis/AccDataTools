# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:06:33 2020

@author: uic
"""

from tslearn.clustering import TimeSeriesKMeans
import numpy as np
import random
import matplotlib.pyplot as plt

#First create a dummy dataset
#It will contain 30 sine waves and
#30 sawtooth waves
data = np.zeros((600,100))
for i in range(300):
    #sine waves
    seed = 2*np.pi*random.random()
    data[i] = np.sin(np.linspace(seed, 4*np.pi+seed,num=100))
for i in range(300,600):
    #sawtooth waves
    seed = 2*np.pi*random.random()
    data[i] = 0.4*(np.mod(np.linspace(seed, 4*np.pi+seed,num=100),2*np.pi)/np.pi-1)

model = TimeSeriesKMeans(
    n_clusters = 2,
    metric = 'dtw',
    verbose = 1
    )


fig,axes = plt.subplots(1,3)
for axis in axes:
    axis.set_xlabel('Time')
    axis.set_ylabel('Signal')
    axis.set_ylim((-1.1,1.1))


for datum in data[0:5]:
    axes[0].plot(datum,
             color = 'green')
    
for datum in data[300:305]:
    axes[1].plot(datum,
             color = 'red')
axes[0].set_title('Examples of training data')
axes[1].set_title('Examples of training data')
model.fit(data)
sines_cluster_1 = 0
tris_cluster_1 = 0
sines_cluster_2 = 0
tris_cluster_2 = 0

#this comparator is broken!!
for label in model.labels_[:300]:
    if label:
        sines_cluster_1+=1
    else:
        tris_cluster_1+=1
for label in model.labels_[300:]:
    if label:
        sines_cluster_2+=1
    else:
        tris_cluster_2+=1
sines_cluster_1 /= 3
sines_cluster_2 /= 3
tris_cluster_1  /= 3
tris_cluster_2  /= 3
a = f'''Cluster 1 (n={sum(model.labels_)}) contained {sines_cluster_1:.1f}% sine waves and
        {tris_cluster_1:.1f}% sawtooth waves'''
b = f'''Cluster 2 (n={sum(np.logical_not(model.labels_))}) contained {sines_cluster_2:.1f}% sine waves and
        {tris_cluster_2:.1f}%sawtooth waves'''
#broken code ends here

axes[2].set_title('Barycenters of clusters')
axes[2].plot(model.cluster_centers_[0])
axes[2].plot(model.cluster_centers_[1])
axes[0].annotate(a, (0,0), (0, -45), xycoords='axes fraction', textcoords='offset points', va='top')
axes[1].annotate(b, (0,0), (0, -45), xycoords='axes fraction', textcoords='offset points', va='top')
plt.tight_layout()
fig.show()
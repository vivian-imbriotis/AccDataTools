# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:52:41 2020

@author: uic
"""
import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from string import ascii_lowercase
from random import sample
from scipy.linalg import fractional_matrix_power



def edit_map(str1,str2):
    res = []
    for a,b in zip(str1,str2):
        res.append(a!=b)
    return res

edit_dist = lambda str1,str2:sum(edit_map(str1,str2))
d = edit_dist

def get_n_nearest_neighbours(point,dataset,n):
    opt = [(np.inf,None)]*n
    for idx,val in enumerate(dataset):
        if idx != point:
            d = edit_dist(dataset[point],val)
            if (d,idx)<max(opt):
                opt[-1] = (d,idx)
                opt.sort()
    return list(x[1] for x in opt)



def alter(strng,n):
    strng = list(strng)
    idxs = sample(range(len(strng)),n)
    repls = sample(ascii_lowercase,n)
    for idx,repl in zip(idxs,repls):
        strng[idx] = repl
    return ''.join(strng)

    
def to_knn_graph_laplacian(dataset,k):
    laplacian = np.zeros((len(dataset),len(dataset)))
    degree = np.copy(laplacian)                     
    for idx,value in enumerate(dataset):
        laplacian[idx][idx] = k
        degree[idx][idx] = k
        neighbors = get_n_nearest_neighbours(idx,dataset,n=k)
        for neighbor in neighbors:
            laplacian[idx][neighbor] = -1
            laplacian[neighbor][idx] = -1
    d = fractional_matrix_power(degree,-0.5)
    return d@laplacian@d

def to_knn_graph_affinity(dataset,k):
    affinity = np.zeros((len(dataset),len(dataset)))                   
    for idx,value in enumerate(dataset):
        affinity[idx][idx] = k
        neighbors = get_n_nearest_neighbours(idx,dataset,n=k)
        for neighbor in neighbors:
            affinity[idx][neighbor] = 1
            affinity[neighbor][idx] = 1
    return affinity

def pairwise_edit_distances(dataset):
    matr = np.zeros((len(dataset),len(dataset)))
    for idx0, elem0 in enumerate(dataset):
        for idx1, elem1 in enumerate(dataset):
            matr[idx0][idx1] = edit_dist(elem0,elem1)
    return matr

def mean_edit_distance(dataset):
    distances = pairwise_edit_distances(dataset)
    return (distances.sum().sum() / (len(dataset)**2))


def arrays_from_labels(labels):
    result = []
    for i in range(max(labels)+1):
        result.append([])
    for idx,label in enumerate(labels):
        result[label].append(idx)
    return result

def data_from_labels(labels,dataset):
    arrays = arrays_from_labels(labels)
    clusters = []
    for i in range(max(labels)+1):
        clusters.append([])
    for idx,ls in enumerate(clusters):
        ls += list(dataset[x] for x in arrays[idx])
    return clusters


a='absolutely'
b='background'
c='conclusion'
def gen_data(*strs,n=6):
    dataset = []
    for i in range(10):
        for strng in strs:
            dataset.append(alter(strng,n))
    return dataset

dataset = gen_data(a,b,c)

def eval_n_clusters(dataset,n):
    A = to_knn_graph_affinity(dataset, 10)
    clustering = SpectralClustering(n_clusters=n,
                   affinity = 'precomputed',
                   assign_labels = 'discretize').fit(A)
    ls = []
    for cluster in data_from_labels(clustering.labels_,dataset):
        ls.append(mean_edit_distance(cluster))
    return sum(ls)/len(ls)


def eval_up_to(n_clusters,method = 'mean_dist'):
    total_mean_edit_distance = mean_edit_distance(dataset)
    ls = []
    if method == 'mean_dist':
        ls.append(total_mean_edit_distance)
    elif method == 'variance':
        ls.append(1)
    for i in range(2,n_clusters):
        if method == 'mean_dist':
            ls.append(eval_n_clusters(dataset,i))
        elif method == 'variance':
            val = eval_n_clusters(dataset,i)
            ls.append(val/total_mean_edit_distance)
    ls[0] = ls[0] if method=='mean_dist' else 1
    return ls

plt.plot(eval_up_to(10,method = 'variance'))
plt.plot(eval_up_to(10,method = 'variance'))
plt.plot(eval_up_to(10,method = 'variance'))
plt.ylim((0,1.1))
plt.xlabel("Number of clusters")
plt.ylabel("Mean intracluster edit distance")
plt.title("""Regardless of underlying data, mean intracluster distance decreases
          with increasing number of clusters""")


















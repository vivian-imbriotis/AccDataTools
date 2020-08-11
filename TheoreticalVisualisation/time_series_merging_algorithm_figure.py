# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 18:23:19 2020

@author: Vivian Imbriotis
"""


import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from random import randint

CUTOFF = 0.9
N = 300
N_CLUSTERS = 10
MAX_COPIES = 10
MIN_COPIES = 2
NOISE = 0.2

for i in range(100):
    df = pd.DataFrame()
    name = 0
    for i in range(N_CLUSTERS):
        base = np.random.uniform(0,1,size=N)
        for j in range(randint(MIN_COPIES,MAX_COPIES)):
            df[str(name)]   = base + np.random.uniform(-NOISE/2,NOISE/2,size=N)
            name+=1
    
    
    corrs    = df.corr()
    adj_matr = (corrs - np.eye(corrs.shape[0])) > CUTOFF
    edges    = adj_matr[adj_matr > 0].stack().index.tolist()
    
    
    graph = nx.Graph()
    graph.add_nodes_from(adj_matr.columns)
    graph.add_edges_from(edges)
    
    new_df = pd.DataFrame()
    
    #nx.draw(graph, with_labels = True)
    for component in nx.connected_components(graph):
        name = " ".join(component)
        new_df[name] = df[component].mean(axis='columns')
        
    
    #Merging algorithm
    
    
    new_corrs    = new_df.corr()
    new_adj_matr = (new_corrs - np.eye(new_corrs.shape[0])) > CUTOFF
    new_edges    = new_adj_matr[new_adj_matr > 0].stack().index.tolist()
    
    new_graph = nx.Graph()
    new_graph.add_nodes_from(new_adj_matr.columns)
    new_graph.add_edges_from(new_edges)
    
    #Evaluate Results
    initially_unconnected = not adj_matr.any(axis=None)
    finally_unconnected   = not new_adj_matr.any(axis=None)
    
    
    if not initially_unconnected:
        try: assert finally_unconnected
        except AssertionError as e:
            fig,ax = plt.subplots(ncols = 2, figsize = (12,6))
            ax[0].set_title("Initial Graph")
            ax[1].set_title("Final Graph")
            nx.draw_networkx(graph, with_labels = True, ax=ax[0])        
            nx.draw_networkx(new_graph, with_labels = True, ax=ax[1])
            fig.show()
            raise e

fig,ax = plt.subplots(ncols = 2, figsize = (12,6))
ax[0].set_title("Initial Graph")
ax[1].set_title("Final Graph")
nx.draw_networkx(graph, with_labels = True, ax=ax[0])        
nx.draw_networkx(new_graph, with_labels = True, ax=ax[1])
fig.show()
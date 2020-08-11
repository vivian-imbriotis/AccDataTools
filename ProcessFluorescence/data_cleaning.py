# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:06:01 2020

@author: viviani

It's necessary to remove duplicate ROIs, ie ROIs that correspond to different
spacial sections of the same axon. This is likely to occur, for example, as
the axon weaves in and out of the plane of section.
"""


import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from random import randint


def merge_rois(roi_array):
    '''
    Merge together ROI traces with a pairwise pearson's R above 0.9.
    
    Parameters
    ----------
    roi_array : Array of float of shape (rois, timepoints)
        An array of dF/F0 values indexed by time and roi

    Returns
    -------
    merged_roi_array : Array of float of shape (merged_rois, timepoints)
        merged_rois <= rois.

    '''
    rois, timepoints = roi_array.shape
    df = pd.DataFrame(data = roi_array.transpose(), 
                      index = range(timepoints),
                      columns = range(rois))
    output = merge_correlated_columns(df)
    
    merged_roi_array = output.to_numpy().transpose()
    return merged_roi_array



def merge_correlated_columns(df):
    adjacency_matrix = get_adj_matr(df)
    nodes            = adjacency_matrix.columns
    edges            = ls_of_edges_from_adj_matr(adjacency_matrix)
    graph            = construct_graph_from(nodes,
                                            edges)
    new_df = pd.DataFrame()
    
    #nx.draw(graph, with_labels = True)
    for component in nx.connected_components(graph):
        name = ", ".join(str(component))
        name = f"[{name}]"
        new_df[name] = df[component].mean(axis='columns')
    return new_df
        


def ls_of_edges_from_adj_matr(adj_matr):
    return adj_matr[adj_matr > 0].stack().index.tolist()


def get_adj_matr(df, cutoff = 0.9):
    corrs            = df.corr()
    adjacency_matrix = (corrs - np.eye(corrs.shape[0])) > cutoff
    return adjacency_matrix


def construct_graph_from(nodes,edges):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph




def unit_test1():
    N = 100
    N_CLUSTERS = 4
    MAX_COPIES = 4
    MIN_COPIES = 2
    NOISE = 0.36
    
    for i in range(100):
        df = pd.DataFrame()
        name = 0
        for i in range(N_CLUSTERS):
            base = np.random.uniform(0,1,size=N)
            for j in range(randint(MIN_COPIES,MAX_COPIES)):
                df[str(name)]   = base + np.random.uniform(-NOISE/2,NOISE/2,size=N)
                name+=1
        
        
        adjacency_matrix = get_adj_matr(df)
        nodes            = adjacency_matrix.columns
        edges            = ls_of_edges_from_adj_matr(adjacency_matrix)
        graph            = construct_graph_from(nodes,
                                                edges)

        new_df = merge_correlated_columns(df)
        
        
        
        new_corrs    = new_df.corr()
        new_adj_matr = get_adj_matr(new_df)
        new_edges    = ls_of_edges_from_adj_matr(new_adj_matr)
        
        new_graph = construct_graph_from(new_adj_matr.columns,
                                         new_edges)
        
        #Evaluate Results
        initially_unconnected = not adjacency_matrix.any(axis=None)
        finally_unconnected   = not new_adj_matr.any(axis=None)
        
        
        if not initially_unconnected:
            try: assert finally_unconnected
            except AssertionError as e:
                # fig,ax = plt.subplots(ncols = 2, figsize = (12,6))
                # ax[0].set_title("Initial Graph")
                # ax[1].set_title("Final Graph")
                # nx.draw_networkx(graph, with_labels = True, ax=ax[0])        
                # nx.draw_networkx(new_graph, with_labels = True, ax=ax[1])
                # fig.show()
                # raise e
                pass
    
    fig,ax = plt.subplots(ncols = 2, nrows = 2, figsize = (12,6))
    ax[0][0].set_title("Time Series before merging")
    ax[0][0].imshow(df.to_numpy().transpose())
    ax[0][1].set_title("Time Series after merging")
    ax[0][1].imshow(new_df.to_numpy().transpose())
    ax[1][0].set_title("Correlation Graph (before merging)")
    ax[1][1].set_title("Correlation Graph (after merging)")
    nx.draw_networkx(graph, nx.kamada_kawai_layout(graph), 
                     with_labels = True, ax=ax[1][0])        
    nx.draw_networkx(new_graph, nx.kamada_kawai_layout(new_graph),
                     with_labels = True, ax=ax[1][1])
    fig.show()

def unit_test2():
    N = 100
    ROIS = 100
    TIMEPOINTS = 200
    NUM_CLUSTERS = 4
    NUM_PER_CLUSTER = 15
    NOISE = 0.3
    
    clusters = NUM_CLUSTERS
    i = 0
    data = np.zeros((ROIS,TIMEPOINTS))
    while i < ROIS:
        base = np.random.uniform(0,1,size=TIMEPOINTS)
        if clusters>0:
            for _ in range(NUM_PER_CLUSTER):
                print("a")
                data[i] = base + np.random.uniform(-NOISE/2,NOISE/2,size=TIMEPOINTS)
                i+=1
            clusters -= 1
        else:
            print('b')
            data[i] = base
            i+=1

    data_merged = merge_rois(data)
    fig,ax = plt.subplots(ncols = 2, figsize = (12,6))
    ax[0].set_title("Time Series before merging")
    ax[0].imshow(data)
    ax[1].set_title("Time Series after merging")
    ax[1].imshow(data_merged)
    fig.show()
    return data
    

if __name__=="__main__":
    data = unit_test2()
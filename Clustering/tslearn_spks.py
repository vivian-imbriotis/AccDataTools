# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:06:33 2020

@author: uic
"""

from tslearn.clustering import TimeSeriesKMeans, silhouette_score
import numpy as np
from accdatatools.Utils.deeploadmat import loadmat
import matplotlib.pyplot as plt



def get_fitted_model(dataset, clusters = 3, timepoints = 0, verbose = False,
                     return_dataset = True):
    if timepoints!=0:
        dataset = dataset[:,:timepoints]
    if verbose: print(f'Segmenting data into {clusters} clusters...')
    model = TimeSeriesKMeans(
        n_clusters = clusters,
        n_init = 10,
        metric = 'dtw',         #Dynamic time warping
        verbose = verbose,
        n_jobs = -1             #Use all cores
        )
    model.fit(dataset)
    return model

def get_frame_times(timeline_path):
    timeline = loadmat(timeline_path)
    timeline = timeline['Timeline']
    frame_counter = timeline['rawDAQData'][:,2]
    timestamps = timeline['rawDAQTimestamps']
    frame_times = np.zeros(int(frame_counter[-1]))
    current_frame = 0
    for timestamp,no_frames in zip(timestamps,frame_counter):
        if no_frames == (current_frame + 1):
            frame_times[current_frame] = timestamp
            current_frame+=1
        elif no_frames != current_frame:
            raise IOError('Need a better error message')
    return frame_times


def get_inertia_stats(models):
    inertias = []
    for model in models:
        inertias.append(model.inertia_)
    inertias = list(enumerate(inertias,2))
    xs = list(x[0] for x in inertias)
    ys = list(x[1] for x in inertias)
    return xs, ys


if __name__=='__main__':
    TIMELINE = 'C:/Users/Vivian Imbriotis/Desktop/2018-04-26_01_CFEB106/1-3-2018-04-26_01_CFEB106_1/metadata matlab/2018-04-26_01_CFEB106_Timeline.mat'
    SPKS = 'C:/Users/Vivian Imbriotis/Desktop/2018-04-26_01_CFEB106/1-3-2018-04-26_01_CFEB106_1/npy/spks.npy'
    TIMEPOINTS = 200
    MAX_K = 6
    stamps = get_frame_times(TIMELINE)

    scores = []
    models = []
    data = np.load(SPKS)
    for i in range(2,MAX_K+1):
        model = get_fitted_model(data, 
                            clusters = i, 
                            timepoints = TIMEPOINTS,
                            verbose = True)
        models.append(model)
    for model in models:
        scores.append(silhouette_score(
            data[:,0:TIMEPOINTS],
            model.labels_,
            metric = 'dtw',
            n_jobs = -1,
            verbose = True))
    plt.plot(list(range(2,MAX_K+1)),scores)
    plt.title("Performance of KMeans on axonal data")
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.show()
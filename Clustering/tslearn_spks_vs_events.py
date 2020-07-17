# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:06:33 2020

@author: uic
"""

from tslearn.clustering import TimeSeriesKMeans, silhouette_score
import numpy as np
from accdatatools.Utils.deeploadmat import loadmat
import matplotlib.pyplot as plt
import accdatatools.GLOBALS as g
import pickle


class Trial:
    def __init__(self,struct):
        self.stim_id        = struct.stimID
        self.correct        = struct.correct
        self.start_trial    = struct.timing.StartTrial
        self.start_stimulus = struct.timing.StimulusStart
        self.start_response = struct.timing.ResponseStart
        self.end_response   = struct.timing.ResponseEnd
        self.end_trial      = struct.timing.EndClearDelay
    def __repr__(self):
        trialtype = g.TRIALTYPE[self.stim_id]
        response = 'correct' if self.correct else 'incorrect'
        return f'{trialtype} trial with {response} response'




def get_fitted_model(dataset, clusters = 3, start = 0, end = 0, verbose = False,
                     return_dataset = True):
    if end!=0 or start!=0:
        dataset = dataset[:,start:end]
    if verbose: print(f'Segmenting data into {clusters} clusters...')
    metric_params = {
        'global_constraint':'sakoe_chiba'
        }
    model = TimeSeriesKMeans(
        n_clusters = clusters,
        n_init = 1,
        metric = 'dtw',         #Dynamic time warping
        verbose = verbose,
        n_jobs = -1,             #Use all cores
        metric_params = metric_params)
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


def _get_trial_structs(psychstim_path):
    matfile=loadmat(psychstim_path)
    expData = matfile["expData"]
    trialData = expData["trialData"]
    trials = []
    for trial in trialData:
        trials.append(trial)
    return trials

def get_trials(psychstim_path):
    structs = _get_trial_structs(psychstim_path)
    trials = []
    for struct in structs:
        trials.append(Trial(struct))
    return trials


def get_inertia_stats(models):
    inertias = []
    for model in models:
        inertias.append(model.inertia_)
    inertias = list(enumerate(inertias,2))
    xs = list(x[0] for x in inertias)
    ys = list(x[1] for x in inertias)
    return xs, ys


if __name__=='__main__':
    TIMELINE   = 'Data/2018-04-26_01_CFEB106_Timeline.mat'
    SPKS       = 'Data/spks.npy'
    PSYCHSTIM  = 'Data/2018-04-26_01_CFEB106_psychstim.mat'
    CLUSTERS   = 2
    FIRST_TRIAL= 0
    LAST_TRIAL = 10
    

    frametimes = get_frame_times(TIMELINE)
    trials = get_trials(PSYCHSTIM)[FIRST_TRIAL:LAST_TRIAL+1]
    
    start_time = trials[0].start_trial
    end_time   = trials[-1].end_trial
    

    start_idx = frametimes.searchsorted(start_time)
    end_idx = frametimes.searchsorted(end_time)
    data = np.load(SPKS)[:,start_idx:end_idx]
    print(data.shape)
    model = get_fitted_model(data, 
                            clusters = CLUSTERS, 
                            start = start_idx,
                            end = end_idx,
                            verbose = True)
    fig,ax = plt.subplots()
    for barycenter in model.cluster_centers_:
        ax.plot(barycenter)
    ylim = ax.get_ylim()
    for trial in trials:
        max_val = trial.end_trial
        response = plt.Rectangle(
            xy = (trial.start_response,ylim[0]),
            width = (trial.end_response - trial.start_response),
            height = ylim[1],
            angle = 0,
            color = 'green' if trial.correct else 'red',
            alpha = 0.5)
        ax.add_patch(response)
    ax.set_xlim((0,max_val+3))
    ax.set_title('ACC Bouton Cluster Barycenters with associated trials')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Deconvolved firing rate')
    plt.savefig('Barycentres.png')
    with open('model.pkl','wb') as file:
        pickle.dump(model,file)

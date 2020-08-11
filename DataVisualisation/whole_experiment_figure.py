# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:15:20 2020

@author: Vivian Imbriotis
"""
import os

from rastermap.mapping import Rastermap
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import seaborn


from accdatatools.Observations.recordings import Recording
from accdatatools.Observations.trials import get_trials_in_recording
from accdatatools.Timing.synchronisation import get_neural_frame_times, get_lick_state_by_frame
from accdatatools.Utils.map_across_dataset import apply_to_all_one_plane_recordings
from accdatatools.Utils.convienience import item




class ExperimentFigure:
    def __init__(self,exp_path, merge = True):
        '''
        A nice heatmap figure for all
        neurons' responses over the course of a trial.
        

        Parameters
        ----------
        exp_path : string
            The root path of an experiment.

        '''
        seaborn.set_style("dark")
        print(f"Loading data from {exp_path}")
        self.trials, self.recording = get_trials_in_recording(exp_path, return_se=True)
        print("Running rastermap on fluorescence data")
        r = Rastermap()

        #Sort by rastermap embedding
        print("Sorting traces by rastermap ordering")
        if merge:
            r.fit(self.recording.dF_on_F_merged)
            self.dF_on_F = self.recording.dF_on_F_merged[r.isort]
        else:
            r.fit(self.recording.dF_on_F)
            self.dF_on_F = self.recording.dF_on_F[r.isort]
        
        #Show overall plot
        timeline_path  = os.path.join(exp_path,
                                  item(
                                      [s for s in os.listdir(exp_path) if 'Timeline.mat' in s]))
        
        print("Fetching Frame Times...")
        frame_times = get_neural_frame_times(timeline_path, self.recording.ops["nframes"])
        print("Fetching licking information...")
        self.licks = get_lick_state_by_frame(timeline_path, frame_times)
        print("Aligning frames with trials...")
        self.start_idxs, self.end_idxs = self.get_trial_attributes(frame_times)
        print("...done")

    def get_trial_attributes(self, frame_times):
        start_times = np.array([trial.start_stimulus for trial in self.trials]) - 1
        end_times   = start_times + 5
        start_idxs  = frame_times.searchsorted(start_times)
        end_idxs    = frame_times.searchsorted(end_times)
        if np.any(end_idxs>self.recording.dF_on_F.shape[-1]):
            raise ValueError("Trial not contained in recording")
        return (start_idxs,end_idxs)
    
    def show(self,start=0,end=-1):
        fig, ax = plt.subplots()
        if end==-1: end = self.dF_on_F.shape[1]-1
        max_brightness = np.percentile(self.dF_on_F[:,start:end],99)
        ax.imshow(np.clip(self.dF_on_F[:,start:end],-0.2,max_brightness), origin = 'lower')
        ax.set_xlabel("Time (Frames)")
        ax.set_ylabel("ROIs (PCA-sorted)")
        for trial, trial_start, trial_end in zip(self.trials, self.start_idxs,self.end_idxs):
            if trial_start>start and trial_end<end:
                rect = Rectangle((trial_start,-10 if trial.isleft else -5),
                                 5*5,5,
                                 color = "green" if trial.isgo else "red",
                                 fill = True if trial.correct else False)
                ax.add_patch(rect)
        lick_frame = np.nonzero(self.licks)
        ax.vlines(lick_frame, -15,-10)
        ax.set_xlim(0,350)
        fig.show()
        return fig

class PupilExperimentFigure(ExperimentFigure):
    #TODO
    pass

class NeuropilExperimentFigure(ExperimentFigure):
    '''
    A heatmap of all the neuropil regions over an experiment.
    
    Parameters
    ----------
    exp_path : string
        The root path of an experiment.
    '''
    def __init__(self,exp_path):
        seaborn.set_style("dark")
        print(f"Loading data from {exp_path}")
        self.trials, self.recording = get_trials_in_recording(exp_path, return_se=True,
                                                              ignore_dprime=True)
        print("Running rastermap on fluorescence data")
        r = Rastermap()
        r.fit(self.recording.Fneu)
        #Sort by rastermap embedding
        print("Sorting traces by rastermap ordering")
        self.dF_on_F = self.recording.Fneu[r.isort]
        timeline_path  = os.path.join(exp_path,
                                  item(
                                      [s for s in os.listdir(exp_path) if 'Timeline.mat' in s]))
        
        print("Fetching Frame Times...")
        frame_times = get_neural_frame_times(timeline_path, self.recording.ops["nframes"])
        print("Fetching licking information...")
        self.licks = get_lick_state_by_frame(timeline_path, frame_times)
        print("Aligning frames with trials...")
        self.start_idxs, self.end_idxs = self.get_trial_attributes(frame_times)
        print("...done")


if __name__=="__main__":
    fig = ExperimentFigure(
        "H:/Local_Repository/CFEB027/2016-10-07_03_CFEB027")
    fig.show()

# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 11:45:31 2020

@author: Vivian Imbriotis

Tools for Trial-based handling of the data.
A "trial" is a single stimulus/response pair, where a mouse is presented
with a stimulus, attempts to respond in a go/nogo manner, and is 
rewarded appropriately.
Includes a Trial class encapsulating a single trial event, a constructor
that takes a recording's parent directory, a figure generation class,
and a main function for dumping all trial objects constructable from the
dataset to a binary file.'
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import os
import pickle as pkl

from accdatatools import GLOBALS

from accdatatools.Utils.path_manipulation import apply_to_all_one_plane_recordings
from accdatatools.Utils.deeploadmat import loadmat
from accdatatools.Utils.convienience import item
from accdatatools.DataCleaning.determine_dprime import calc_d_prime
from accdatatools.Timing.synchronisation import get_neural_frame_times,get_lick_state_by_frame
from accdatatools.Observations.recordings import Recording



class Trial:
    '''
    Encapusulates a single trial event.
    '''
    
    
    def __init__(self,exp_path,struct,statistic_extractor,frame_times,
                 licks):
        
        if os.path.isdir(exp_path):
            self.exp_path  = exp_path
        else: 
            raise ValueError('recording must be an existing directory')
        self.stim_id        = struct.stimID
        self.isleft         = self.stim_id in LEFT
        self.isright        = self.stim_id in RIGHT
        self.isgo           = self.stim_id in GO
        self.correct        = struct.correct
        self.affirmative    = ((self.correct and self.isgo) or 
                               (not self.correct and not self.isgo))
        self.contrast       = struct.stimAttributes.contrast
        #Get absolute timing information
        self.start_trial    = struct.timing.StartTrial
        self.start_stimulus = struct.timing.StimulusStart
        self.start_response = struct.timing.ResponseStart
        self.end_response   = struct.timing.ResponseEnd
        self.end_trial      = struct.timing.EndClearDelay
        
        #Get some relative timing as sugar
        self.rel_start_stim = self.start_stimulus - self.start_trial
        self.rel_start_resp = self.start_response - self.start_trial
        self.rel_end_resp   = self.end_response   - self.start_trial
        self.duration       = self.end_trial      - self.start_trial
        
        #Get the df/f, deconvoluted firing, and licking
        #traces of ROIs during the trial (licking info is shared
        #by all ROIs in a trial)
        self.dF_on_F, self.spks, self.licks  = self.get_traces(statistic_extractor,
                                                    frame_times,
                                                    licks)

        
        #If there's no timepoints in dF_on_F, something has gone wrong:
        if self.dF_on_F.shape[1]<1:
            raise ValueError('Trial not contained in recording')
        
        self.ROI_identifiers = np.argwhere(statistic_extractor.iscell)
        
    def get_traces(self,statistic_extractor, frame_times, licks):

        all_traces = statistic_extractor.dF_on_F[statistic_extractor.iscell]
        all_spks   = statistic_extractor.spks[statistic_extractor.iscell]
        #We need the indexes of the frames corresponding to trial
        #start and end times
        start_idx = frame_times.searchsorted(self.start_stimulus - 1)
        end_idx   = start_idx + 26
        if end_idx>all_traces.shape[1]:
            raise ValueError("Trial not contained in recording")
        return (all_traces[:,start_idx:end_idx], all_spks[:,start_idx:end_idx],
                licks[start_idx:end_idx])
        
    def __repr__(self):
        trialtype = TRIALTYPE[self.stim_id]
        response = 'correct' if self.correct else 'incorrect'
        return f'{trialtype} trial with {response} response'
    def plot(self):
        seaborn.set()
        fig, ax = plt.subplots()
        #Plot all the ROI responses plus a mean
        for ROI in self.dF_on_F:
            ax.plot(ROI, alpha = 0.25, color = "black")
        ax.plot(np.mean(ROI,axis=0), alpha = 1, color = "orange")
        
        ylim = ax.get_ylim()
        #Add rectangles indicating stimulus visibility and response
        response = plt.Rectangle(
            xy = (self.rel_start_resp*30//6,ylim[0]),
            width = (self.rel_end_resp - self.rel_start_resp)*30//6,
            height = ylim[1],
            color = 'green' if self.affirmative else 'red',
            alpha = 0.5)
        stimulus = plt.Rectangle(
            xy = (self.rel_start_stim*30//6,ylim[0]),
            width = 2*30//6,
            height = ylim[1],
            color = 'green' if self.isgo else 'red',
            alpha = 0.5)
        ax.add_patch(response)
        ax.add_patch(stimulus)
        fig.show()


    
def _get_trial_structs(psychstim_path):
    '''
    Get matlab structs for each trial recorded in an experiement.

    Parameters
    ----------
    psychstim_path : str
        A psychstim.mat file in a parent directory.

    Returns
    -------
    trials : list of matlab struct objects
    '''
    matfile = loadmat(psychstim_path)
    expData = matfile["expData"]
    trialData = expData["trialData"]
    trials = []
    for trial in trialData:
        trials.append(trial)
    return trials

def get_trials_in_recording(exp_path, return_se=False, ignore_dprime=False,
                            se = None, suppress_dprime_error=False):
    '''
    Retrieve all the trials in a recording as Trial objects

    Parameters
    ----------
    exp_path : String
        Path to the experiment folder.
    return_se : Bool, optional
        Whether to also return a StatisticExtractor object for the 
        whole experiment. The default is False.

    Returns
    -------
    [Trial] or ([Trial], StatisticExtractor)

    '''
    
    #Get the appropriate paths for the suite2p info, the timeline,
    #and the trial metadata
    files          = os.listdir(exp_path)
    s2p_path       = os.path.join(exp_path,'suite2p','plane0')
    timeline_path  = os.path.join(exp_path,
                                  item(
                                      [s for s in files if 'Timeline.mat' in s]))
    psychstim_path = os.path.join(exp_path,
                                  item(
                                      [s for s in files if 'psychstim.mat' in s]))
    trials = []
    if calc_d_prime(psychstim_path)>1 or ignore_dprime:
        if se==None:
            se      = Recording(s2p_path)
        #We need the total number of frames:
        nframes = se.ops["nframes"]
        times   = get_neural_frame_times(timeline_path,nframes)
        structs = _get_trial_structs(psychstim_path)
        licks = get_lick_state_by_frame(timeline_path, times)
        
        for struct in structs:
            trial = Trial(exp_path,struct,se,times, licks)
            trials.append(trial)
        return trials if not return_se else (trials,se)
    elif suppress_dprime_error:
        return None if not return_se else (None,None)
    else:
        raise ValueError("Dprime below 1")

def add_trials_in_rec_to_file(filestream, exp_path):
    for trial in get_trials_in_recording(exp_path):
        pkl.dump(trial,filestream)


def dump_all_trials_in_dataset_to_pkl_file(drive = 'E:\\',
                                           path = "all_one_plane_trials.pkl"):
    all_trials = open("all_one_plane_trials.pkl",'wb')
    add_trials_to_file = (lambda exp_path:
                              add_trials_in_rec_to_file(
                                  all_trials, exp_path))
    apply_to_all_one_plane_recordings(drive,add_trials_to_file)
    all_trials.close()


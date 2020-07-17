# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 20:16:38 2020

@author: Vivian Imbriotis
"""
import seaborn
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
from accdatatools.Observations.trials import dump_all_trials_in_dataset_to_pkl_file
import numpy as np
seaborn.set()


class ROITrialResponse:
    def __init__(self, trial, idx, roi_identifier, trial_identifier):
        #Copy all the attributes from the parent trial
        self.__dict__ = trial.__dict__.copy()
        #But we only want one ROI's data, so discard the others'
        self.dF_on_F = trial.dF_on_F[idx,:]
        self.spks    = trial.spks[idx,:]
        
        #Then get nice string representations of the ROI and the recording
        self.ROI_ID = self.recording.split("\\")[-1] + f" {roi_identifier}"
        self.trial_id = self.recording.split("\\")[-1] + f" {trial_identifier}"
    def to_dict(self):
        return {
            'ROI_ID':self.ROI_ID,
            'Trial_ID':self.trial_id,
            'go': self.isgo,
            'side': 'left' if self.isleft else ('right' if self.isright else 'unknown'),
            'correct': self.correct,
            'affirmative': self.affirmative,
            'rel_start_stim': self.rel_start_stim,
            "rel_start_resp": self.rel_start_resp,
            "rel_end_resp": self.rel_end_resp,
            "trial_duration": self.duration,
            "dF_on_F": self.dF_on_F,
            "lick_rate": self.licks
            }
    def to_unrolled_records(self):
        start_idx = int((self.rel_start_stim - 1)*30//6)
        end_idx = start_idx + 30*(1+2+2)//6 - 1
        #check that these idxs exist (since slice notation doesn't raise
        #  any IndexErrors)
        self.dF_on_F[start_idx]
        self.dF_on_F[end_idx]
        results = []
        for time, (df, spk, lick) in enumerate(zip(
                self.dF_on_F[start_idx:end_idx],
                self.spks[start_idx:end_idx],
                self.licks[start_idx:end_idx])):
            results.append(
                {
                'ROI_ID':self.ROI_ID,
                'Trial_ID':self.trial_id,
                'go': 'TRUE' if self.isgo else 'FALSE',
                'side_is_left': 'TRUE' if self.isleft else 'FALSE',
                'correct': 'TRUE' if self.correct else 'FALSE',
                'affirmative': 'TRUE' if self.affirmative else 'FALSE',
                "time": time/(30//6),
                "dF_on_F": df,
                "Neuron_Firing": 'TRUE' if spk>0 else 'FALSE',
                "lick_during_frame": 'TRUE' if (lick > 0) else 'FALSE'
                }
                )
        return results

class ROIActivitySummary:
    '''
    A visual summary of an ROI's behaviour across trials.'

    Parameters
    ----------
    roi_trial_responses : Iterable of ROITrialResponse objects
        The responses for which to generate a summary.

    '''
    
    def __init__(self, roi_trial_responses):
        ls = roi_trial_responses #For brevity; doing list comprehensions
        left_correct  = [t for t in ls if t.isleft and t.correct]
        left_wrong    = [t for t in ls if t.isleft and not t.correct]
        right_correct = [t for t in ls if t.isright and t.correct]
        right_wrong   = [t for t in ls if t.isright and not t.correct]
        datasets = (left_correct,left_wrong,right_correct,right_wrong)
        titles = ["Left Correct", "Left Incorrect",
                  "Right Correct", "Right Incorrect"]
        self.fig, self.ax = plt.subplots(nrows = 2, ncols = 2, 
                                         constrained_layout=True,
                                         )
        for axes, data, title in zip(self.ax.flatten(), datasets, titles):
            for trial in data:
                start_idx = int((trial.rel_start_stim - 1)*30//6)
                end_idx = start_idx + 30*(1+2+2)//6 - 1
                trial.dF_on_F[start_idx]
                trial.dF_on_F[end_idx]
                trace = trial.dF_on_F[start_idx:end_idx]
                axes.plot(trace, color='black', alpha = 0.25)
            ylim = axes.get_ylim()
            stimulus = plt.Rectangle(
                xy = (1*30//6,ylim[0]),
                width = 2*30//6,
                height = ylim[1] - ylim[0],
                color = 'blue',
                alpha = 0.5)
            axes.add_patch(stimulus)
            axes.set_title(title)
        self.fig.suptitle(f"Activity of {ls[0].ROI_ID}")
        self.show()
    def show(self):
        self.fig.show()

class ROIActivitySummaryFactory:
    def __init__(self, ls_of_rois):
        self.ls_of_rois = ls_of_rois
        self.roi_ids = [r.ROI_ID for r in self.ls_of_rois]
        self.roi_ids = list(set(self.roi_ids))
        self.roi_ids.sort()
    def plot(self,ID):
        if type(ID)==int:
            ID = self.roi_ids[ID]
        elif type(ID)==str:
            pass
        else:
            raise ValueError(f'ID must be string or int, not {type(ID)}')
        ROIActivitySummary([r for r in self.ls_of_rois if r.ROI_ID==ID])
    def __call__(self,N):
        self.plot(N)

def axon_responses_from_trial(trial, trial_id):
    res = []
    for ROI_idx, ROI_identifier in enumerate(trial.ROI_identifiers):
        res.append(ROITrialResponse(
            trial,ROI_idx,ROI_identifier, trial_id)
            )
    return res

def ls_of_ROIs_from_ls_of_Trials(ls, destructive = True):
    res = []
    for idx, trial in enumerate(ls):
        res.extend(axon_responses_from_trial(trial, idx))
    return res
    
def dataframe_from_ls_of_ROIs(ls):
    return pd.DataFrame.from_records(ROI.to_dict() for ROI in ls)

def flat_dataframe_from_ls_of_ROIs(ls):
    records = []
    for roi in ls:
        records.extend(roi.to_unrolled_records())
    return pd.DataFrame(records)
    
def load_all_trials(path):
    all_trials = []
    with open(path,'rb') as file:
        while True:
            try:
                 all_trials.append(
                     pkl.load(file)
                     )
            except EOFError:
                break
    return all_trials




if __name__ == "__main__":
    dump_all_trials_in_dataset_to_pkl_file("H:", "all_one_plane_trials.pkl")
    trials = load_all_trials("all_one_plane_trials.pkl")
    rois = ls_of_ROIs_from_ls_of_Trials(trials)
    del trials
    plotter = ROIActivitySummaryFactory(rois)
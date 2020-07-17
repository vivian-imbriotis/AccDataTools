# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:07:54 2020

@author: Vivian Imbriotis
"""


from DataCleaning.add_time_since_lick import lick_transform
from DataCleaning.automate_s2p import apply_to_all_one_plane_recordings
from trial_extraction import get_trials_in_recording, get_frame_times, item, get_lick_state_by_frame
import numpy  as np
import pandas as pd
import os

class RecordingUnroller(Recording):
    def __init__(self,exp_path,ignore_dprime=False):
        self.exp_path = exp_path
        super().__init__()
        self.trials = get_trials_in_recording(exp_path, 
                                              return_se=False,
                                              se = self,
                                              ignore_dprime = ignore_dprime,
                                              suppress_dprime_error=False)
        timeline_path  = os.path.join(exp_path,
                                  item(
                                      [s for s in os.listdir(exp_path) if 'Timeline.mat' in s]))
        
        self.frame_times = get_frame_times(timeline_path, self.se.ops["nframes"])
        #Now  we somehow need to get this into a continuous time series.
        self.trialtime, self.iscorrect, self.side, self.isgo, self.trial_id = self.get_timeseries()
        self.licks = get_lick_state_by_frame(timeline_path, self.frame_times)
        #This licks is the bool series, we want the deltaT series
        self.licks = lick_transform(self.licks)
        self.dF_on_F = self.se.dF_on_F
        self.spks    = self.se.spks
        

    def get_timeseries(self):
        start_times = np.array([trial.start_stimulus for trial in self.trials]) - 1
        end_times   = start_times + 5
        start_idxs  = self.frame_times.searchsorted(start_times)
        end_idxs    = self.frame_times.searchsorted(end_times)
        trial_s = np.zeros(self.se.ops["nframes"])
        corre_s = np.full(self.se.ops["nframes"],-1)
        side_s  = np.full(self.se.ops["nframes"],'NA',dtype = object)
        isgo_s  = np.full(self.se.ops["nframes"],-1)
        id_s = np.full(self.se.ops["nframes"],"NA", dtype= object)
        
        for idx,(trial, start_idx, end_idx) in enumerate(zip(self.trials, start_idxs, end_idxs)):
            trial_id = self.exp_path.split("\\")[-1] + f" {idx}"
            trial_s[start_idx:end_idx] = np.arange(1,end_idx-start_idx+1)
            corre_s[start_idx:end_idx] = 1 if trial.correct else 0
            side_s[start_idx:end_idx]  = 'Left' if trial.isleft else 'Right'
            isgo_s[start_idx:end_idx]  = 1 if trial.isgo else 0
            id_s[start_idx:end_idx] = trial_id
        return (trial_s,corre_s,side_s,isgo_s,id_s)
    
    def to_unrolled_records(self):
        results = []
        ROI_IDs = self.trials[0].ROI_identifiers
        for roi_df, roi_spks, roi_id in zip(self.dF_on_F, self.spks, ROI_IDs):
            for idx,(df, spk, trialtime, correct, side, go, lick, frametime, trial_id) in enumerate(zip(
                    roi_df,
                    roi_spks,
                    self.trialtime,
                    self.iscorrect,
                    self.side,
                    self.isgo,
                    self.licks,
                    self.frame_times,
                    self.trial_id)):
                        results.append(
                            {
                            'ROI_ID':self.exp_path.split("\\")[-1] + f" {roi_id.item()}",
                            'Trial_ID':trial_id,
                            'go': go,
                            'side': side,
                            'correct': correct,
                            "time": frametime,
                            "trial_time": trialtime,
                            "dF_on_F": df,
                            "spks": spk,
                            "lick": lick
                            }
                            )
        return results

def get_dataframe_from_path(path, ignore_dprime=False):
    try:
        return pd.DataFrame(RecordingUnroller(path, ignore_dprime=ignore_dprime
                                      ).to_unrolled_records())
    except ValueError as e:
        print("ValueError in get_dataframe_from_path")
        raise e
        return None

def append_recording_to_csv(filestream,path, ignore_dprime=False):
    df = get_dataframe_from_path(path,ignore_dprime=ignore_dprime)
    if type(df)==pd.DataFrame:
        df.to_csv(filestream,header=(filestream.tell()==0))
    del df

if __name__=="__main__":
    #First, delete all existing file contents
    open("C:/Users/Vivian Imbriotis/Desktop/dataset.csv",'w').close()
    #Reopen in append mode and append each experiment
    csv = open("C:/Users/Vivian Imbriotis/Desktop/dataset.csv", 'a')
    func = lambda path:append_recording_to_csv(csv,path,True)
    apply_to_all_one_plane_recordings("E:\\", func)
    csv.close()
    pass
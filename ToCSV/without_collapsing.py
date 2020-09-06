# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:07:54 2020

@author: Vivian Imbriotis
"""
import os

import numpy  as np
import pandas as pd


from accdatatools.ProcessLicking.kernel import lick_transform
from accdatatools.Utils.map_across_dataset import (apply_to_all_one_plane_recordings,
                                                   apply_to_all_recordings_of_class)
from accdatatools.Observations.trials import get_trials_in_recording
from accdatatools.Observations.recordings import Recording
from accdatatools.Utils.convienience import item
from accdatatools.Utils.path import get_exp_id, DLC_ANALYSED_VIDEOS_DIRECTORY
from accdatatools.Timing.synchronisation import (get_neural_frame_times, 
                                                get_lick_state_by_frame,
                                                get_eye_diameter_at_timepoints)




class RecordingUnroller(Recording):
    def __init__(self,exp_path,ignore_dprime=False, tolerate_lack_of_eye_video = False):
        self.exp_path = exp_path
        super().__init__(exp_path)
        self.trials = get_trials_in_recording(exp_path, 
                                              return_se=False,
                                              se = self,
                                              ignore_dprime = ignore_dprime,
                                              suppress_dprime_error=False)
        timeline_path  = os.path.join(exp_path,
                                  item(
                                      [s for s in os.listdir(exp_path) if 'Timeline.mat' in s]))
        
        self.frame_times = get_neural_frame_times(timeline_path, self.ops["nframes"])
        
        #Now  we somehow need to get this into a continuous time series.
        (self.trialtime, self.iscorrect, self.side, self.isgo, self.contrast,
         self.trial_id, self.trial_num, self.peritrialtime, 
         self.trial_component) = self.get_timeseries()
        
        self.licks = get_lick_state_by_frame(timeline_path, self.frame_times)
        #This licks is the bool series, we want the deltaT series
        self.licks = lick_transform(self.licks)
        
        #Need to check if this experiment had a video that was processed
        #by our DeepLabCut Network...
        path, exp_id = os.path.split(self.exp_path)
        for file in os.listdir(DLC_ANALYSED_VIDEOS_DIRECTORY):
            if exp_id in file and '.h5' in file:
                hdf_path = os.path.join(DLC_ANALYSED_VIDEOS_DIRECTORY,
                                        file)
                self.pupil_diameter = get_eye_diameter_at_timepoints(hdf_path, 
                                                             timeline_path, 
                                                             self.frame_times)
                break
        else:
            if not tolerate_lack_of_eye_video:
                raise ValueError(f"No associated eyecam footage found at {DLC_ANALYSED_VIDEOS_DIRECTORY}")
            self.pupil_diameter = [np.nan]*self.ops["nframes"]


    def get_timeseries(self):
        #When did trials happen in terms of neural frames?
        start_times = np.array([trial.start_stimulus for trial in self.trials]) - 1
        start_idxs  = self.frame_times.searchsorted(start_times)
        end_idxs    = start_idxs + 5*5 #each trial is 5s at 5FPS
        

        trial_s     = np.ones(self.ops["nframes"])*(-999)
        peritrial_s = np.ones(self.ops["nframes"])*(-999)
        corre_s     = np.full(self.ops["nframes"],-1)
        side_s      = np.full(self.ops["nframes"],'NA',dtype = object)
        isgo_s      = np.full(self.ops["nframes"],-1)
        con_s       = np.full(self.ops["nframes"],-1,dtype=float)
        id_s        = np.full(self.ops["nframes"],"NA", dtype= object)
        comp_s      = np.full(self.ops["nframes"],"NA", dtype= object)
        trial_num   = np.zeros(self.ops["nframes"])
        
        trial_struct = np.array(["Tone"]*5 + ["Stim"]*10 + ["Resp"]*10)
        
        for idx,(trial, start_idx, end_idx) in enumerate(zip(self.trials, start_idxs, end_idxs)):
            peristart = start_idx - 5*3
            periend = start_idx
            trial_id = self.exp_path.split("\\")[-1] + f" {idx}"
            trial_s[start_idx:end_idx] = np.arange(1,end_idx-start_idx+1)
            comp_s[start_idx:end_idx] = trial_struct
            peritrial_s[peristart:periend] = np.arange(periend-peristart,0,-1)
            corre_s[start_idx:end_idx] = 1 if trial.correct else 0
            side_s[start_idx:end_idx]  = 'Left' if trial.isleft else 'Right'
            isgo_s[start_idx:end_idx]  = 1 if trial.isgo else 0
            con_s[start_idx:end_idx]  = trial.contrast
            id_s[start_idx:end_idx] = trial_id
            trial_num[start_idx:] += 1
        return (trial_s,corre_s,side_s,isgo_s,con_s,id_s,trial_num,peritrial_s,
                comp_s)
    
    def to_unrolled_records(self):
        results = []
        ROI_IDs = self.trials[0].ROI_identifiers
        for roi_df, roi_spks, roi_id in zip(self.dF_on_F, self.spks, ROI_IDs):
            for idx,(df, spk, trialtime, peritrialtime, comp, correct, side, go,
                     contrast,
                     lick, pupil, frametime, trial_id, trial_num) in enumerate(zip(
                                                            roi_df,
                                                            roi_spks,
                                                            self.trialtime,
                                                            self.peritrialtime,
                                                            self.trial_component,
                                                            self.iscorrect,
                                                            self.side,
                                                            self.isgo,
                                                            self.contrast,
                                                            self.licks,
                                                            self.pupil_diameter,
                                                            self.frame_times,
                                                            self.trial_id,
                                                            self.trial_num)):
                        results.append(
                            {
                            'ROI_ID':self.exp_path.split("\\")[-1] + f" {roi_id.item()}",
                            'Trial_ID':trial_id,
                            'go': go,
                            'side': side,
                            'correct': correct,
                            'contrast':contrast,
                            "time": frametime,
                            "trial_factor": trialtime,
                            "peritrial_factor":peritrialtime,
                            "trial_component":comp,
                            "dF_on_F": df,
                            "spks": spk,
                            "lick_factor": lick,
                            "pupil_diameter": pupil if not np.isnan(pupil) else 'NA',
                            "number_of_trials_seen":trial_num
                            }
                            )
        return results
    def to_dataframe(self):
        records = self.to_unrolled_records()
        return pd.DataFrame(records)

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

def get_whole_dataset(drive, cls=None):
    result = []
    def func(path):
        recorder = RecordingUnroller(path, ignore_dprime = True, 
                                     tolerate_lack_of_eye_video = True)
        df = recorder.to_dataframe()
        del recorder
        result.append(df)
        del df
    if not cls:
        apply_to_all_one_plane_recordings(drive, func)
    else:
        apply_to_all_recordings_of_class(cls,drive,func)
    result = pd.concat(result,ignore_index=True)
    return result

def construct_csv_for_recording_class(csv_path,cls):
    dataset = get_whole_dataset("H:",cls = cls)
    csv = open(csv_path, "w")
    dataset.to_csv(csv)
    
if __name__=="__main__":
    construct_csv_for_recording_class("C:/Users/viviani/Desktop/low_contrast.csv",
                                      "low_contrast")
    construct_csv_for_recording_class("C:/Users/viviani/Desktop/left_only_high_contrast.csv",
                                      "left_only_high_contrast")
    construct_csv_for_recording_class("C:/Users/viviani/Desktop/both_sides_high_contrast.csv",
                                      "both_sides_high_contrast")
    

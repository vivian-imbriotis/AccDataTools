# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 11:46:41 2020

@author: Vivian Imbriotis

Converting the suite2p, DeepLabCut, and experiment metadata intermediaries
to a CSV for further analysis with R, collapsing across time, ROIs, or both
to reduce the number of dimentions in the output data.
"""

from unroll_dataset import Recording
import numpy as np
import pandas as pd
from DataCleaning.path_manipulation import apply_to_all_one_plane_recordings

class PerTrialRecording(Recording):
    def get_trial_auc(self,collapse_across_ROIs = True):
        roi_aucs = np.array([np.sum(trial.dF_on_F,axis=-1) for trial in self.trials])
        if collapse_across_ROIs:
            return np.mean(roi_aucs,axis=-1)
        else:
            return roi_aucs
        
    def get_tone_auc(self,collapse_across_ROIs = True):
        roi_aucs = np.array(
            [np.sum(trial.dF_on_F[:,:5],axis=-1) for trial in self.trials])
        if collapse_across_ROIs:
            return np.mean(roi_aucs,axis=-1)
        else:
            return roi_aucs
        
    def get_stim_auc(self,collapse_across_ROIs = True):
        roi_aucs = np.array(
            [np.sum(trial.dF_on_F[:,5:15],axis=-1) for trial in self.trials])
        if collapse_across_ROIs:
            return np.mean(roi_aucs,axis=-1)
        else:
            return roi_aucs
    def get_resp_auc(self,collapse_across_ROIs = True):
        roi_aucs = np.array(
            [np.sum(trial.dF_on_F[:,15:],axis=-1) for trial in self.trials])
        if collapse_across_ROIs:
            return np.mean(roi_aucs,axis=-1)
        else:
            return roi_aucs
    def to_unrolled_records(self,collapse_across_ROIs=True):
        output = []
        for (trial_idx,trial), full_auc, tone_auc, stim_auc, resp_auc in zip(
                enumerate(self.trials),
                self.get_trial_auc(collapse_across_ROIs),
                self.get_tone_auc(collapse_across_ROIs),
                self.get_stim_auc(collapse_across_ROIs),
                self.get_resp_auc(collapse_across_ROIs)):
            if collapse_across_ROIs:
                output.append({"TrialID": trial.recording,
                        "Correct": trial.correct,
                        "Go":      trial.isgo,
                        "Side":    "Left" if trial.isleft else "Right",
                        "TrialAUC":full_auc,
                        "ToneAUC": tone_auc,
                        "StimAUC": stim_auc,
                        "RespAUC": resp_auc
                        })
            else:
                for roi, (f,t,s,r) in enumerate(zip(full_auc,tone_auc,
                                                    stim_auc,resp_auc)):
                    output.append({
                            "TrialID": trial.recording + str(trial_idx),
                            "roiNum":  roi,
                            "Correct": trial.correct,
                            "Go":      trial.isgo,
                            "Contrast":trial.contrast,
                            "Side":    "Left" if trial.isleft else "Right",
                            "TrialAUC":f,
                            "ToneAUC": t,
                            "StimAUC": s,
                            "RespAUC": r
                            })
        return output
    
    def to_csv(self, file, collapse_across_ROIs = True):
        df = pd.DataFrame(self.to_unrolled_records(collapse_across_ROIs))
        if type(file)==str:
            df.to_csv(file)
        else:
            df.to_csv(file,header=(file.tell()==0))



if __name__=="__main__":
    #First, delete all existing file contents
    open("C:/Users/Vivian Imbriotis/Desktop/byroitrialdataset.csv",'w').close()
    #Reopen in append mode and append each experiment
    csv = open("C:/Users/Vivian Imbriotis/Desktop/byroitrialdataset.csv", 'a')
    func = lambda path:PerTrialRecording(path, True).to_csv(csv,False)
    apply_to_all_one_plane_recordings("E:\\", func)
    csv.close()

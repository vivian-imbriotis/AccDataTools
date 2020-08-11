# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 11:46:41 2020

@author: Vivian Imbriotis

Converting the suite2p, DeepLabCut, and experiment metadata intermediaries
to a CSV for further analysis with R, collapsing across time or across both
time and ROIS to reduce the number of dimentions in the output data.
Collapsing across ROIs without collapsing across time is not currently implemented.
"""

from without_collapsing import RecordingUnroller
import numpy as np
import pandas as pd
from scipy.stats import kstest, zscore, pearsonr
import matplotlib.pyplot as plt
# from scipy.stats import pearsonr
from accdatatools.Utils.map_across_dataset import apply_to_all_one_plane_recordings

# def calculate_pvalues(df):
#     df = df.dropna()._get_numeric_data()
#     dfcols = pd.DataFrame(columns=df.columns)
#     pvalues = dfcols.transpose().join(dfcols, how='outer')
#     for r in df.columns:
#         for c in df.columns:
#             pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
#     return pvalues

class CollapsingRecordingUnroller(RecordingUnroller):
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
    def get_reaction_time(self):
        reaction_times = np.zeros(len(self.trials))
        for idx,trial in enumerate(self.trials):
            #Mice can only have reaction times if they're correct on a go trial.
            if all((trial.correct, not trial.istest, trial.isgo)):
                #When was the first lick after the stimulus appeared, 
                #excluding the first 500ms?
                first_lick_after_stim = np.argmax(
                    self.lick_times>(trial.start_stimulus+0.5))
                lick_time = self.lick_times[first_lick_after_stim]
                if lick_time>trial.end_response: reaction_time = False
                else: reaction_time = lick_time - trial.start_stimulus
                if reaction_time>3.5: reaction_time = False
            else:
                reaction_time = False
            reaction_times[idx] = reaction_time
        return reaction_times
    
    def get_mean_pupil_size_over_stim_period(self):
        mean_pupil_diameters = np.zeros(len(self.trials))
        for idx,trial in enumerate(self.trials):
            pupil_size = np.nanmean(
                self.pupil_diameter[trial.start_idx:trial.end_idx]
                )
            mean_pupil_diameters[idx] = pupil_size
        return mean_pupil_diameters
    
    def to_unrolled_records(self,collapse_across_ROIs=True):
        output = []
        for ((trial_idx,trial), full_auc, tone_auc, stim_auc, resp_auc, 
             reaction_time, pupil_size) in zip(
                enumerate(self.trials),
                self.get_trial_auc(collapse_across_ROIs),
                self.get_tone_auc(collapse_across_ROIs),
                self.get_stim_auc(collapse_across_ROIs),
                self.get_resp_auc(collapse_across_ROIs),
                self.get_reaction_time(),
                self.get_mean_pupil_size_over_stim_period()):
            if collapse_across_ROIs:
                output.append({"TrialID": self.exp_path.split("\\")[-1] + f" {trial_idx}",
                        "Correct": trial.correct,
                        "Go":      trial.isgo,
                        "Side":    "Left" if trial.isleft else ("Right" if trial.isright else "Unknown"),
                        "contrast": trial.contrast,
                        "TrialAUC":full_auc,
                        "ToneAUC": tone_auc,
                        "StimAUC": stim_auc,
                        "RespAUC": resp_auc,
                        "Reaction_Time": reaction_time if reaction_time else np.nan,
                        "Pupil_size": pupil_size if pupil_size else np.nan
                        })
            else:
                for roi, (f,t,s,r) in enumerate(zip(full_auc,tone_auc,
                                                    stim_auc,resp_auc)):
                    output.append({
                            "TrialID": self.exp_path.split("\\")[-1] + f" {trial_idx}",
                            "roiNum":  roi,
                            "Correct": trial.correct,
                            "Go":      trial.isgo,
                            "Contrast":trial.contrast,
                            "Side":    "Left" if trial.isleft else "Right",
                            "TrialAUC":f,
                            "ToneAUC": t,
                            "StimAUC": s,
                            "RespAUC": r,
                            "Reaction_Time": reaction_time if reaction_time else np.nan,
                            "Pupil_size": pupil_size if pupil_size else np.nan
                            })
        return output
    
    def to_csv(self, file, collapse_across_ROIs = True):
        df = pd.DataFrame(self.to_unrolled_records(collapse_across_ROIs))
        if type(file)==str:
            df.to_csv(file)
        else:
            df.to_csv(file,header=(file.tell()==0))
            
    def to_dataframe(self,collapse_across_ROIs = True):
        df = pd.DataFrame(self.to_unrolled_records(collapse_across_ROIs))
        return df

def generate_attention_metrics_figure():
    df = CollapsingRecordingUnroller("H:/Local_Repository/CFEB013/2016-06-29_02_CFEB013",True,False).to_dataframe()
    df2 = df.loc[:,'Reaction_Time':'Pupil_size'][df.Reaction_Time!='NA'].dropna()
    df2.Reaction_Time = pd.to_numeric(df.Reaction_Time)
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
    ax[0][0].plot(df2.Reaction_Time,df2.Pupil_size,'o')
    ax[0][0].set_xlabel("Reaction Time (s)")
    ax[0][0].set_ylabel("Pupil Diameter while reacting (pixels)")
    ax[0][1].violinplot([df.Pupil_size[df.Correct==True].values,
                df.Pupil_size[df.Correct==False].values]
                   )
    ax[0][1].set_ylabel("Pupil diameter while reacting (pixels)")
    ax[0][1].set_xticks([1,2])
    ax[0][1].set_xticklabels(["Correct","Incorrect"])
    reactions = df[['contrast','Reaction_Time']].dropna()
    ax[1][0].violinplot([reactions.Reaction_Time[reactions.contrast==0.1].values,
                   reactions.Reaction_Time[reactions.contrast==0.5].values],
                  )
    ax[1][0].set_xticks([1,2])
    ax[1][0].set_xticklabels(["10%","50%"])
    ax[1][0].set_xlabel("Contrast")
    ax[1][0].set_ylabel("Reaction Time")
    pupils = df[['contrast','Pupil_size']].dropna()
    ax[1][1].violinplot([pupils.Pupil_size[pupils.contrast==0.1].values,
                   pupils.Pupil_size[pupils.contrast==0.5].values],
                  )
    ax[1][1].set_xticks([1,2])
    ax[1][1].set_xticklabels(["10%","50%"])
    ax[1][1].set_xlabel("Contrast")
    ax[1][1].set_ylabel("Pupil Diameter while reacting (pixels)")
    fig.show()

def perform_attention_metrics_testing():
    df = pd.read_csv("C:/Users/viviani/Desktop/alltrials.csv")
    df["Recording_ID"] = list(map(lambda s:s.split(" ")[-1],df.TrialID.values))
    for ID in df.Recording_ID.unique():
        subset = df[df.Recording_ID == ID].Pupil_size.values
        subset_zscore = zscore(subset, nan_policy = "omit")
        df.loc[df.Recording_ID == ID, "Pupil_size"] = subset_zscore
    df2 = df.loc[:,'Reaction_Time':'Pupil_size'][df.Reaction_Time!='NA'].dropna()
    df2.Reaction_Time = pd.to_numeric(df.Reaction_Time)
    fig,ax = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
    ax[0][0].plot(df2.Reaction_Time,df2.Pupil_size,'o',
                  markersize = 1)
    ax[0][0].set_xlabel("Reaction Time (s)")
    ax[0][0].set_ylabel("Pupil Diameter while reacting\n(z-score within recording)")
    df3 = df[['Correct','Pupil_size']].dropna()
    pupil_size_by_correctness = [df3.Pupil_size[df.Correct==True].values,
                                 df3.Pupil_size[df.Correct==False].values]
    ax[0][1].violinplot(pupil_size_by_correctness)
    ax[0][1].set_ylabel("Pupil diameter while reacting\n(z-score within recording)")
    ax[0][1].set_xticks([1,2])
    ax[0][1].set_xticklabels(["Correct","Incorrect"])
    reactions = df[['contrast','Reaction_Time']].dropna()
    rt_by_contrast = [reactions.Reaction_Time[reactions.contrast==0.1].values,
                      reactions.Reaction_Time[reactions.contrast==0.5].values]
    ax[1][0].violinplot(rt_by_contrast)
    ax[1][0].set_xticks([1,2])
    ax[1][0].set_xticklabels(["10%","50%"])
    ax[1][0].set_xlabel("Contrast")
    ax[1][0].set_ylabel("Reaction Time")
    pupils = df[['contrast','Pupil_size']].dropna()
    pupil_size_by_contrast = [pupils.Pupil_size[pupils.contrast==0.1].values,
                              pupils.Pupil_size[pupils.contrast==0.5].values]
    ax[1][1].violinplot(pupil_size_by_contrast)
    ax[1][1].set_xticks([1,2])
    ax[1][1].set_xticklabels(["10%","50%"])
    ax[1][1].set_xlabel("Contrast")
    ax[1][1].set_ylabel("Pupil Diameter while reacting\n(z-score within recording)")
    fig.show()
    
    
    print("\n\n\n\n\nPupil Size Vs Reaction Time")
    s,p = pearsonr(df2.Reaction_Time,df2.Pupil_size)
    print(f"PearsonrResult(statistic={s},\npvalue={p})")
    
    print("\nPupil Size Vs Contrast:")
    print(kstest(pupil_size_by_contrast[0],
                 pupil_size_by_contrast[1]))
    print("\nPupil Size Vs Correctness")
    print(kstest(pupil_size_by_correctness[0],
                 pupil_size_by_correctness[1]))
    print("\nReaction Time Vs Contrast")
    print(kstest(rt_by_contrast[0],
                 rt_by_contrast[1]))
    print(f"\n0.05-Critical value after Bonferroni Correction\nfor 4 comparisons is {0.05/4:.4f}")


def get_dataframe_of_full_dataset(drive):
    result = []
    def func(path):
        recorder = CollapsingRecordingUnroller(path, True)
        df = recorder.to_dataframe(collapse_across_ROIs = True)
        del recorder
        result.append(df)
        del df
    apply_to_all_one_plane_recordings(drive, func)
    result = pd.concat(result,ignore_index=True)
    return result

def dump_dataset_as_csv_to(path):
    '''Dump every recording into a CSV file'''
    #First, delete all existing file contents
    open(path,'w').close()
    #Reopen in append mode and append each experiment
    csv = open(path, 'a')
    func = lambda path:CollapsingRecordingUnroller(path, True).to_csv(csv,False)
    apply_to_all_one_plane_recordings("E:\\", func)
    csv.close()

if __name__=="__main__":
    # dump_dataset_as_csv_to(
    #     "C:/Users/Vivian Imbriotis/Desktop/byroitrialdataset.csv")


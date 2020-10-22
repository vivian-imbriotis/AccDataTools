# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 11:46:41 2020

@author: Vivian Imbriotis

Converting the suite2p, DeepLabCut, and experiment metadata intermediaries
to a CSV for further analysis with R, collapsing across time or across both
time and ROIS to reduce the number of dimentions in the output data.
Collapsing across ROIs without collapsing across time is not currently implemented.
"""
import warnings
from string import ascii_uppercase

from accdatatools.ToCSV.without_collapsing import RecordingUnroller
import numpy as np
import pandas as pd
from scipy.stats import kstest, zscore, pearsonr, ttest_ind
import matplotlib.pyplot as plt
# from scipy.stats import pearsonr
from accdatatools.Utils.map_across_dataset import apply_to_all_recordings_of_class
import seaborn as sb
from numpy import diff

sns = sb

sb.set_style('darkgrid')
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 11


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
    def get_pupil_change(self):
        mean_change_pupil_diameters = np.zeros(len(self.trials))
        for idx,trial in enumerate(self.trials):
            pupil_size = np.nanmean(
                np.diff(
                    self.pupil_diameter[trial.start_idx:trial.start_idx+5]
                ))
            mean_change_pupil_diameters[idx] = pupil_size
        return mean_change_pupil_diameters
    
    def get_reaction_time(self):
        #Going to use the Fisher-Jenks algorithm to exclude the burst of
        #licking behaviour associated with the tone/stimulus onset.
        reaction_times = np.zeros(len(self.trials))
        for idx,trial in enumerate(self.trials):
            #Mice can only have reaction times if they're correct on a go trial.
            if all((trial.correct, not trial.istest, trial.isgo)):
                licks_during_trial = self.lick_times[np.logical_and(
                                        self.lick_times>trial.start_stimulus,
                                        self.lick_times<trial.end_response)]
                try:
                    reflex_licks_start,reflex_licks_end,_ = jenks_breaks(
                        licks_during_trial,
                        nb_class = 2)
                    first_lick_after_stim = np.argmax(
                        self.lick_times>(reflex_licks_end))
                    lick_time = self.lick_times[first_lick_after_stim]
                except ValueError:
                    #We have a single lick
                    lick_time = licks_during_trial[0]
                if lick_time>trial.end_response: reaction_time = False
                else: reaction_time = lick_time - trial.start_stimulus
                if reaction_time>3.5: reaction_time = False
            else:
                reaction_time = False
            reaction_times[idx] = reaction_time
        return reaction_times
    
    def cluster_licking(self):
        #Let's build up a jagged array of [([],[])], ie so we have a list
        #of pairs of lists - each trial has two clusters of licks, one
        #for each cluster of licks as defned by Fisher-Jenks.
        trial_licking = [] #The outer list
        for idx,trial in enumerate(self.trials):
            #Mice can only have reaction times if they're correct on a go trial.
            if all((trial.correct, not trial.istest, trial.isgo)):
                #Get all licks during the trial
                licks_during_trial = self.lick_times[np.logical_and(
                                        self.lick_times>(trial.start_stimulus-1),
                                        self.lick_times<trial.start_stimulus+4)]
                #Now we want to split this list in two using Fisher-Jenks
                try:
                    reflex_licks_start,reflex_licks_end,_ = jenks_breaks(
                        licks_during_trial,
                        nb_class = 2)
                    fst_cluster = licks_during_trial[licks_during_trial<=reflex_licks_end]
                    snd_cluster = licks_during_trial[licks_during_trial>reflex_licks_end]
                    fst_cluster -= trial.start_stimulus
                    snd_cluster -= trial.start_stimulus
                except ValueError:
                    #We have a single lick
                    fst_cluster = licks_during_trial
                    snd_cluster = np.array([])
                split_ls = (fst_cluster,snd_cluster)
                trial_licking.append(split_ls)
        return trial_licking
    
    def plot_licking_clustering_approach(self):
        trial_licking = self.cluster_licking()
        sb.set_style('darkgrid')
        fig,ax = plt.subplots()
        for idx,(fst_cluster,snd_cluster) in enumerate(trial_licking):
            x1 = np.full(fst_cluster.shape,idx)
            ax.plot(fst_cluster, x1,'o',color='darksalmon',markersize=2.5)
            x2 = np.full(snd_cluster.shape,idx)
            ax.plot(snd_cluster,x2,'o',color='blue',markersize=2.5)
        ymax,ymin = ax.get_ylim()
        ax.vlines((0,1,3),ymin,ymax,linestyles='dashed',color = 'k')
        for name,pos in zip(('Tone','Stimulus','Response'),(0,1,3)):
            ax.text(pos+0.1,ymax,name,
                    horizontalalignment='left',
                    verticalalignment='bottom')
        ax.set_ylabel('Trial Number')
        ax.set_xlabel(r'$\Delta$t from trial onset (s)')
        plt.show()
    
    def get_mean_pupil_size_over_period(self,period='trial'):
        mean_pupil_diameters = np.zeros(len(self.trials))
        for idx,trial in enumerate(self.trials):
            if period=='trial':
                pupil_size = np.nanmean(
                    self.pupil_diameter[trial.start_idx:trial.end_idx]
                    )
            elif period=='tone':
                pupil_size = np.nanmean(
                    self.pupil_diameter[trial.start_idx:trial.start_idx+5]
                    )
            else:
                raise NotImplementedError("period kwarg must be 'trial' or 'tone'")
            mean_pupil_diameters[idx] = pupil_size
        return mean_pupil_diameters
    
    
    def to_unrolled_records(self,collapse_across_ROIs=True):
        output = []
        for ((trial_idx,trial), full_auc, tone_auc, stim_auc, resp_auc, 
             reaction_time, pupil_size, pupil_size_tone, pupil_change) in zip(
                enumerate(self.trials),
                self.get_trial_auc(collapse_across_ROIs),
                self.get_tone_auc(collapse_across_ROIs),
                self.get_stim_auc(collapse_across_ROIs),
                self.get_resp_auc(collapse_across_ROIs),
                self.get_reaction_time(),
                self.get_mean_pupil_size_over_period('trial'),
                self.get_mean_pupil_size_over_period('tone'),
                self.get_pupil_change()):
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
                        "Pupil_size": pupil_size if pupil_size else np.nan,
                        "Pupil_size_tone": pupil_size_tone if pupil_size_tone else np.nan,
                        "Pupil_change": pupil_change if pupil_change else np.nan
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
                            "Pupil_size": pupil_size if pupil_size else np.nan,
                            "Pupil_size_tone": pupil_size_tone if pupil_size_tone else np.nan,
                            "Pupil_change": pupil_change if pupil_change else np.nan
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
    
def perform_attention_metrics_testing_left_only():
    df = pd.read_csv("C:/Users/viviani/Desktop/OldReactionTimeCalc/reaction_time_vs_pupil_size_left_only.csv")
    df["Recording_ID"] = list(map(lambda s:s.split(" ")[-1],df.TrialID.values))
    for ID in df.Recording_ID.unique():
        subset = df[df.Recording_ID == ID].Pupil_size_tone.values
        subset_zscore = zscore(subset, nan_policy = "omit")
        df.loc[df.Recording_ID == ID, "Pupil_size_tone"] = subset_zscore
    df2 = df.loc[:,'Reaction_Time':'Pupil_size_tone'][df.Reaction_Time!='NA'].dropna()
    df2.Reaction_Time = pd.to_numeric(df.Reaction_Time)
    fig,ax = plt.subplots(ncols=2, constrained_layout=True)
    ax[0].plot(df2.Reaction_Time,df2.Pupil_size_tone,'o',
                  markersize = 1)
    ax[0].set_xlabel("Reaction Time (s)")
    ax[0].set_ylabel("Pupil Diameter (mean of second prior to stimulus)\n(z-score within recording)")
    df3 = df[['Correct','Pupil_size_tone']].dropna()
    pupil_size_by_correctness = [df3.Pupil_size_tone[df.Correct==True].values,
                                 df3.Pupil_size_tone[df.Correct==False].values]
    ax[1].violinplot(pupil_size_by_correctness)
    ax[1].set_ylabel("Pupil diameter (mean of second prior to stimulus\n(z-score within recording)")
    ax[1].set_xticks([1,2])
    ax[1].set_xticklabels(["Correct","Incorrect"])
    print("\nPupil Size Vs Reaction Time on Subsequent Trial")
    s,p = pearsonr(df2.Reaction_Time,df2.Pupil_size_tone)
    print(f"PearsonrResult(statistic={s},\npvalue={p})")
    
    print("\nPupil Size vs Correctness on subsequent trial")
    print(ttest_ind(pupil_size_by_correctness[0],
                 pupil_size_by_correctness[1],
                 equal_var=False))
    print(kstest(pupil_size_by_correctness[0],
                 pupil_size_by_correctness[1]))
    print(f"\n0.05-Critical value after Bonferroni Correction\nfor 3 comparisons is {0.05/3:.4f}")
    fig.show()

def perform_attention_metrics_testing_low_contrast():
    df = pd.read_csv("C:/Users/viviani/Desktop/OldReactionTimeCalc/reaction_time_vs_pupil_size_low_contrast.csv")
    df["Recording_ID"] = list(map(lambda s:s.split(" ")[-1],df.TrialID.values))
    for ID in df.Recording_ID.unique():
        subset = df[df.Recording_ID == ID].Pupil_size_tone.values
        subset_zscore = zscore(subset, nan_policy = "omit")
        df.loc[df.Recording_ID == ID, "Pupil_size_tone"] = subset_zscore
    df2 = df.loc[:,'Reaction_Time':'Pupil_size_tone'][df.Reaction_Time!='NA'].dropna()
    df2.Reaction_Time = pd.to_numeric(df.Reaction_Time)
    fig,ax = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
    ax[0][0].plot(df2.Reaction_Time,df2.Pupil_size_tone,'o',
                  markersize = 1)
    ax[0][0].set_xlabel("Reaction Time (s)")
    ax[0][0].set_ylabel("Pupil Diameter (mean of second prior to stimulus)\n(z-score within recording)")
    df3 = df[['Correct','Pupil_size_tone']].dropna()
    pupil_size_by_correctness = [df3.Pupil_size_tone[df.Correct==True].values,
                                 df3.Pupil_size_tone[df.Correct==False].values]
    ax[0][1].violinplot(pupil_size_by_correctness)
    ax[0][1].set_ylabel("Pupil Diameter (mean of second prior to stimulus)\n(z-score within recording)")
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
    pupils = df[['contrast','Pupil_size_tone']].dropna()
    pupil_size_by_contrast = [pupils.Pupil_size_tone[pupils.contrast==0.1].values,
                              pupils.Pupil_size_tone[pupils.contrast==0.5].values]
    ax[1][1].violinplot(pupil_size_by_contrast)
    ax[1][1].set_xticks([1,2])
    ax[1][1].set_xticklabels(["10%","50%"])
    ax[1][1].set_xlabel("Contrast")
    ax[1][1].set_ylabel("Pupil Diameter while reacting\n(z-score within recording)")
    fig.show()
    
    
    print("\nPupil Size Vs Reaction Time")
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

def perform_attention_metrics_testing_with_derivative():
    df = pd.read_csv("C:/Users/viviani/Desktop/OldReactionTimeCalc/reaction_time_vs_pupil_size_low_contrast.csv")
    df["Recording_ID"] = list(map(lambda s:s.split(" ")[-1],df.TrialID.values))
    for ID in df.Recording_ID.unique():
        subset = df[df.Recording_ID == ID].Pupil_change.values
        subset_zscore = zscore(subset, nan_policy = "omit")
        df.loc[df.Recording_ID == ID, "Pupil_change"] = subset_zscore
    df2 = df.loc[:,'Reaction_Time':'Pupil_change'][df.Reaction_Time!='NA'].dropna()
    df2.Reaction_Time = pd.to_numeric(df.Reaction_Time)
    fig,ax = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
    ax[0][0].plot(df2.Reaction_Time,df2.Pupil_change,'o',
                  markersize = 1)
    ax[0][0].set_xlabel("Reaction Time (s)")
    ax[0][0].set_ylabel("Mean Change in pupil diameter over second prior to stimulus\n(z-score within recording)")
    df3 = df[['Correct','Pupil_change']].dropna()
    pupil_size_by_correctness = [df3.Pupil_change[df.Correct==True].values,
                                 df3.Pupil_change[df.Correct==False].values]
    ax[0][1].violinplot(pupil_size_by_correctness)
    ax[0][1].set_ylabel("Mean Change in pupil diameter over second prior to stimulus\n(z-score within recording)")
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
    pupils = df[['contrast','Pupil_change']].dropna()
    pupil_size_by_contrast = [pupils.Pupil_change[pupils.contrast==0.1].values,
                              pupils.Pupil_change[pupils.contrast==0.5].values]
    ax[1][1].violinplot(pupil_size_by_contrast)
    ax[1][1].set_xticks([1,2])
    ax[1][1].set_xticklabels(["10%","50%"])
    ax[1][1].set_xlabel("Contrast")
    ax[1][1].set_ylabel("Mean Change in pupil diameter over second prior to stimulus\n(z-score within recording)")

    
    print("\nChange in Pupil Size Vs Reaction Time")
    s,p = pearsonr(df2.Reaction_Time,df2.Pupil_size)
    print(f"PearsonrResult(statistic={s},\npvalue={p})")
    
    print("\nChange in Pupil Size Vs Contrast:")
    print(kstest(pupil_size_by_contrast[0],
                 pupil_size_by_contrast[1]))
    print("\nChange in Pupil Size Vs Correctness")
    print(kstest(pupil_size_by_correctness[0],
                 pupil_size_by_correctness[1]))
    print("\nReaction Time Vs Contrast")
    print(kstest(rt_by_contrast[0],
                 rt_by_contrast[1]))
    print(f"\n0.05-Critical value after Bonferroni Correction\nfor 4 comparisons is {0.05/4:.4f}")

class AttentionMetricsVsPupilSizeFigure:
    def __init__(self):
        self.fig,ax = plt.subplots(figsize=[8,9],nrows = 3, ncols = 2,
                                   tight_layout=True)
        self.format_axes(ax)
        
    def format_axes(self, ax):
        violins = []
        for col,path in zip((0,1),
                            ("reaction_time_vs_pupil_size_left_only.csv",
                             "reaction_time_vs_pupil_size_low_contrast.csv")):
            print(path)
            df = pd.read_csv(f"C:/Users/viviani/Desktop/OldReactionTimeCalc/{path}")
            df["Recording_ID"] = list(map(lambda s:s.split(" ")[-1],df.TrialID.values))
            for ID in df.Recording_ID.unique():
                subset = df[df.Recording_ID == ID].Pupil_change.values
                subset_zscore = zscore(subset, nan_policy = "omit")
                df.loc[df.Recording_ID == ID, "Pupil_change"] = subset_zscore
            df2 = df.loc[:,'Reaction_Time':'Pupil_change'][df.Reaction_Time!='NA'].dropna()
            df2.Reaction_Time = pd.to_numeric(df.Reaction_Time)
            ax[0][col].plot(df2.Reaction_Time,df2.Pupil_change,'o',
                          markersize = 1)
            ax[0][col].set_xlabel("Reaction Time (s)")
            ax[0][col].set_ylabel("$\\Delta$pupil diameter (z-score)")
            df3 = df[['Correct','Pupil_size_tone']].dropna()
            pupil_size_by_correctness = [df3.Pupil_size_tone[df.Correct==True].values,
                                         df3.Pupil_size_tone[df.Correct==False].values]
            v1 = ax[1][col].violinplot(pupil_size_by_correctness,showmeans=True)
            ax[1][col].set_ylabel("Pupil Diameter (z-score)")
            ax[1][col].set_xticks([1,2])
            ax[1][col].set_xticklabels(["Correct","Incorrect"])
            df3 = df[['Correct','Pupil_change']].dropna()
            pupil_size_by_correctness = [df3.Pupil_change[df.Correct==True].values,
                                         df3.Pupil_change[df.Correct==False].values]
            v2 = ax[2][col].violinplot(pupil_size_by_correctness,showmeans=True)
            ax[2][col].set_ylabel("$\\Delta$Pupil Diameter (z-score)")
            ax[2][col].set_xticks([1,2])
            ax[2][col].set_xticklabels(["Correct","Incorrect"])
            violins.append(v1)
            violins.append(v2)
            
        for violin in violins:
            violin["bodies"][0].set_label("Probability Density")
            violin["cmeans"].set_color("black")
            violin["cmeans"].set_label("Mean")
            violin["cmins"].set_label("Minima and maxima")
        for a,name in zip(ax.flatten(),ascii_uppercase):
            a.set_title("$\\bf{("+name+")}$",loc='right')
        ax[1][0].legend(loc="lower center")

    
    def show(self):
        self.fig.show()

def get_dataframe_of_full_dataset(cls, drive):
    result = []
    def func(path):
        recorder = CollapsingRecordingUnroller(path, True)
        df = recorder.to_dataframe(collapse_across_ROIs = True)
        del recorder
        result.append(df)
        del df
    apply_to_all_recordings_of_class(cls, drive, func)
    result = pd.concat(result,ignore_index=True)
    return result

def dump_dataset_as_csv_to(path, recording_class,collapse_across_rois):
    '''Dump every recording into a CSV file'''
    #First, delete all existing file contents
    open(path,'w').close()
    #Reopen in append mode and append each experiment
    csv = open(path, 'a')
    func = lambda path:CollapsingRecordingUnroller(path, True).to_csv(csv,collapse_across_rois)
    apply_to_all_recordings_of_class(recording_class,"E:\\", func)
    csv.close()

if __name__=="__main__":
    plt.close('all')
    AttentionMetricsVsPupilSizeFigure().show()
    # dump_dataset_as_csv_to(
    #     "C:/Users/viviani/Desktop/reaction_time_vs_pupil_size_left_only.csv",
    #     recording_class='left_only_high_contrast',
    #     collapse_across_rois = True)
    # dump_dataset_as_csv_to(
    #     "C:/Users/viviani/Desktop/reaction_time_vs_pupil_size_low_contrast.csv",
    #     recording_class='low_contrast',
    #     collapse_across_rois = True)
    # plt.close('all')
    # with warnings.catch_warnings():
    #     warnings.simplefilter('ignore')
    #     print("In high contrast (easy task difficulty) conditions:")
    #     perform_attention_metrics_testing_left_only()
    #     print("\n\n In low contrast (hard task difficulty) conditions:")
    #     perform_attention_metrics_testing_low_contrast()
    #     print("\n\n Considering the Derivative of the pupil size")
    #     perform_attention_metrics_testing_with_derivative()
    # # trial_licking = []
    # func = lambda p:trial_licking.extend(CollapsingRecordingUnroller(p,
    #                                                                  ignore_eye_video=True
    #                                                                  ).cluster_licking())
    # apply_to_all_recordings_of_class('left_only_high_contrast','H:\\',func,
    #                                  verbose=False)
    # sb.set_style('darkgrid')
    # fig,ax = plt.subplots()
    # for idx,(fst_cluster,snd_cluster) in enumerate(trial_licking):
    #     try:
    #         x1 = np.full(fst_cluster.shape,idx)
    #         ax.plot(fst_cluster, x1,'o',color='k',markersize=2.5)
    #         x2 = np.full(snd_cluster.shape,idx)
    #         ax.plot(snd_cluster,x2,'o',color='k',markersize=2.5)
    #     except AttributeError:
    #         pass
    # ymax,ymin = ax.get_ylim()
    # ax.vlines((0,1,3),ymin,ymax,linestyles='dashed',color = 'k')
    # for name,pos in zip(('Tone','Stimulus','Response'),(0,1,3)):
    #     ax.text(pos+0.1,ymax,name,
    #             horizontalalignment='left',
    #             verticalalignment='bottom')
    # ax.set_ylabel('Trial Number')
    # ax.set_xlabel(r'$\Delta$t from trial onset (s)')
    # fig.show()

# -*- coding: utf-8 -*-

"""
Created on Fri Jul 31 11:36:46 2020

@author: viviani
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as c
import pandas as pd
import numpy as np
import seaborn
from accdatatools.ToCSV.without_collapsing import RecordingUnroller

def plot_trial_subset(axis,correct,go,df,cmap):
    x = np.arange(1,27)
    first_roi = df.ROI_ID.unique()[0]
    trial_rows = df[df.trial_factor!=-999][df.ROI_ID==first_roi]
    condn_rows   = trial_rows[df.correct==correct][df.go==go]
    pupil = np.array(condn_rows.pupil_diameter.values)
    idxs  = np.array(condn_rows.number_of_trials_seen.values)[::26]
    pupil = pupil[:pupil.shape[0] - pupil.shape[0]%26]
    trials = pupil.reshape(26,-1)
    trials[trials=="NA"] = np.nan
    for idx,trial in zip(idxs,trials.transpose()):
        axis.plot((x)/5,trial,
                color = cmap.to_rgba(idx,0.2))
    axis.plot(x/5,np.nanmean(trials,axis=-1),color='black', label="mean")
    axis.set_ylabel("Pupil Diameter (pixels)")
    axis.set_xlabel("Seconds since stimulus")
    axis.set_xlim((0,None))
    axis.set_ylim((8,25))




def create_figure(df,title=None):
    seaborn.set_style("dark")
    #Get an iterator over each peritrial period
    fig,ax = plt.subplots(ncols = 2, nrows = 2, constrained_layout = True)
    num_trials = len(df.Trial_ID.unique())
    norm = c.Normalize(vmin=0,vmax = num_trials)
    cmap = cm.ScalarMappable(norm,'plasma')
    
    (hit_axis, miss_axis), (fa_axis, cr_axis) = ax
    
    
    #HIT AXIS
    plot_trial_subset(hit_axis, correct=True, go=True, df=df, cmap=cmap)
    hit_axis.set_title("Hit Trials")
    hit_axis.legend()
    
    #MISS AXIS
    plot_trial_subset(miss_axis, correct=False, go=True, df=df, cmap=cmap)
    miss_axis.set_title("Miss Trials")
    
    #FALSE ALARM AXIS
    plot_trial_subset(fa_axis, correct=False, go=False ,df=df, cmap=cmap)
    fa_axis.set_title("False-Alarm Trials")
    
    #CORRECT REJECTION AXIS
    plot_trial_subset(cr_axis, correct=True, go=False, df=df, cmap=cmap)
    cr_axis.set_title("Correct-Rejection Trials")
    
    #ADD A COLORBAR
    cmap._A = []
    cb = fig.colorbar(cmap,ax = ax, location="bottom", shrink = 0.4)
    cb.set_label("Trial Number")
    if title:
        fig.suptitle(title)
    fig.show()

if __name__=="__main__":
    records = RecordingUnroller("H:/Local_Repository/CFEB013/2016-06-29_02_CFEB013",
                       ignore_dprime = True,
                       tolerate_lack_of_eye_video = False).to_unrolled_records()

    df = pd.DataFrame(records)
    create_figure(df[df.contrast == 0.1],title='10% Contrast')
    create_figure(df[df.contrast == 0.5], title = "50% Contrast")
    
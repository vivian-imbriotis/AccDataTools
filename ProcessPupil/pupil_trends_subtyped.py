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
from accdatatools.ToCSV.without_collapsing import RecordingUnroller, get_whole_dataset

def plot_trial_subset(axis,correct,go,df,cmap, normalize = False, render = True):
    print(f"Initial shape of df in plot_trial_subset call is {df.shape}")
    x = np.arange(1,27)
    df = df[df.go==go]
    df = df[df.correct==correct]
    df["roi_num"] = np.fromiter(map(lambda s:s.split(" ")[-1],df.ROI_ID),int)
    trial_ids = df[(df.trial_factor==1)&(df.roi_num==0)].Trial_ID.values
    recordings= list(map(lambda s:s.split(" ")[0],trial_ids))
    trial_ids = map(lambda s:s.split(" ")[-1],trial_ids)
    trial_ids = map(int,trial_ids)
    # pupils_per_timepoint = [df[df.trial_factor==i][df.roi_num==0].pupil_diameter.values for i in x]
    # length = max(map(len, pupils_per_timepoint))
    # pupils_per_timepoint= [list(ls)+[np.nan]*(length-len(ls)) for ls in pupils_per_timepoint]
    # pupils_per_timepoint = np.array(pupils_per_timepoint)
    pupils_per_timepoint = df[df.roi_num==0].pivot(index = "Trial_ID", 
                                                   columns = "trial_factor", 
                                                   values = "pupil_diameter"
                                                   ).to_numpy()
    pupils_per_timepoint[pupils_per_timepoint=="NA"] = np.nan
    pupils_per_timepoint = pupils_per_timepoint.astype(float)
    print(f"Shape of plottable array is {pupils_per_timepoint.shape}")
    if normalize:
        #means of first second
        means = np.nanmean(pupils_per_timepoint[:,:5], axis = -1)
        #divide each row by it's own mean!
        pupils_per_timepoint = pupils_per_timepoint / means[:,None]
    for idx,trial in zip(trial_ids,pupils_per_timepoint):
        if render: axis.plot((x)/5,trial,
                color = cmap.to_rgba(idx,0.2))
    axis.plot(x/5,np.nanmean(pupils_per_timepoint,axis=0),color='black',
              label = 'Mean across trials')
    axis.set_xlim((1,5))
    axis.set_xlabel("time (s)")
    axis.set_ylabel(
        "size of pupil / initial size of pupil" if normalize else "pupil diameter (pixels)"
        )
    if not normalize: axis.set_ylim((0,30))
    else: axis.set_ylim((0,3))
    return (recordings, pupils_per_timepoint)





def create_figure(df,title=None, render = True, normalize = False):
    print(f"Initial shape of dataframe in create_figure call is {df.shape}")
    seaborn.set_style("dark")
    #Clean up dataframe
    df = df[~pd.isnull(df.Trial_ID)]
    #Get an iterator over each peritrial period
    fig,ax = plt.subplots(ncols = 2, nrows = 2, constrained_layout = True,
                          figsize = (12,8))
    num_trials = 250
    norm = c.Normalize(vmin=0,vmax = num_trials)
    cmap = cm.ScalarMappable(norm,'plasma')
    
    (hit_axis, miss_axis), (fa_axis, cr_axis) = ax
    
    
    #HIT AXIS
    hits = plot_trial_subset(hit_axis, correct=True, go=True, df=df, cmap=cmap,
                             normalize = normalize, render = render)
    hit_axis.set_title("Hit Trials")
    hit_axis.legend()
    
    #MISS AXIS
    misses = plot_trial_subset(miss_axis, correct=False, go=True, df=df, cmap=cmap,
                               normalize = normalize, render = render)
    miss_axis.set_title("Miss Trials")
    
    #FALSE ALARM AXIS
    fas = plot_trial_subset(fa_axis, correct=False, go=False ,df=df, cmap=cmap,
                            normalize = normalize, render = render)
    fa_axis.set_title("False-Alarm Trials")
    
    #CORRECT REJECTION AXIS
    crs = plot_trial_subset(cr_axis, correct=True, go=False, df=df, cmap=cmap,
                            normalize = normalize, render = render)
    cr_axis.set_title("Correct-Rejection Trials")
    
    #Turn off inner axis ticks and labels
    hit_axis.set_xticks([])
    hit_axis.set_xlabel("")
    miss_axis.set_xticks([])    
    miss_axis.set_xlabel("")
    miss_axis.set_yticks([])
    miss_axis.set_ylabel("")
    cr_axis.set_yticks([])
    cr_axis.set_ylabel("")
    
    #ADD A COLORBAR
    cmap._A = []
    cb = fig.colorbar(cmap,ax = ax, location="bottom", shrink = 0.4)
    cb.set_label("Trial Number")
    if title:
        fig.suptitle(title)
    if render: fig.show()
    return (hits,misses,fas,crs)

def perform_testing(df, normalize = False):
    hits,misses,fas,crs = create_figure(df,render=False, normalize = normalize)
    hits_df   = pd.DataFrame(data = hits[1])
    hits_df["recording"] = hits[0]
    # hits["trial_no"] = hits.index
    # hits = hits.melt(id_vars = "trial_no")
    # hits.columns = ['trial_no','trial_frame','pupil_diameter']
    # hits["trial_type"] = "hit"
    # return hits
    misses_df = pd.DataFrame(data=misses[1])
    misses_df["recording"] = misses[0]
    fas_df    = pd.DataFrame(data = fas[1])
    fas_df["recording"] = fas[0]
    crs_df    = pd.DataFrame(data=crs[1])
    crs_df["recording"] = crs[0]
    result = []
    for df,trial_type in zip((hits_df,misses_df,fas_df,crs_df),
                             ("hit","miss","fa","cr")):
        df["trial_no"] = df.index
        df = df.melt(id_vars = ["trial_no","recording"])
        df.columns = ['trial_no','recording','trial_frame','pupil_diameter']
        df["trial_type"] = trial_type
        result.append(df)
    df = pd.concat(result)
    return df
    



if __name__=="__main__":
    # records = RecordingUnroller("H:/Local_Repository/CFEB026/2016-09-29_06_CFEB026",
    #                     ignore_dprime = True,
    #                     tolerate_lack_of_eye_video = False).to_unrolled_records()

    # df = pd.DataFrame(records)
    # create_figure(df)
    #df = pd.read_csv("C:/Users/viviani/Desktop/unrolled_dataset.csv")
    df2 = perform_testing(df, normalize = False)

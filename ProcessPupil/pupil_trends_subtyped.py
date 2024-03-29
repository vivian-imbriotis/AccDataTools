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
from accdatatools.DataVisualisation.pupil_mixed_model_output_visualisation import PupilModelPredictionFigure

seaborn.set_style("darkgrid")
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 11

def plot_trial_subset(axis,correct,go,df,cmap, normalize = False, render = True):
    print(f"Initial shape of df in plot_trial_subset call is {df.shape}")
    x = np.arange(1,26)
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
        means = np.nanmean(pupils_per_timepoint[:,:5], axis = 1)
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
        "size of pupil (ratio to initial size of pupil)" if normalize else "pupil diameter (pixels)"
        )
    if not normalize: axis.set_ylim((0,30))
    else: axis.set_ylim((0,2.5))
    return (recordings, pupils_per_timepoint)

a = None
def plot_trial_subset_with_range(axis,correct,go,df, normalize = False, 
                                 range_type = "error", color= "k"):
    if range_type not in ("error","deviation"):
        raise ValueError("range_type must be one of 'error' or 'deviation'")
    print(f"correct = {correct}; go={go}")
    x = np.arange(0,25)
    df = df[df.go==go]
    df = df[df.correct==correct]
    df["roi_num"] = np.fromiter(map(lambda s:s.split(" ")[-1],df.ROI_ID),int)
    trial_ids = df[(df.trial_factor==1)&(df.roi_num==0)].Trial_ID.values
    recordings= pd.Series(map(lambda s:s.split(" ")[0],trial_ids)).unique()
    pupils_per_timepoint = df[df.roi_num==0][~pd.isnull(df.Trial_ID)].pivot(index = "Trial_ID", 
                                                   columns = "trial_factor", 
                                                   values = "pupil_diameter"
                                                   ).to_numpy()

    print(pupils_per_timepoint.shape)
    pupils_per_timepoint[pupils_per_timepoint=="NA"] = np.nan
    pupils_per_timepoint = pupils_per_timepoint.astype(float)
    if normalize:
        means = np.nanmean(pupils_per_timepoint[:,:5], axis = -1)
        #divide each row by it's own mean!
        pupils_per_timepoint = pupils_per_timepoint / means[:,None]
    mean = np.nanmean(pupils_per_timepoint[:,:25],axis=0)
    rang = np.nanstd(pupils_per_timepoint[:,:25],axis=0)
    if range_type=="error":
        #Convert to standard error!
        n_points = np.sum(~np.isnan(pupils_per_timepoint[:,:25]), axis = 0)
        rang /= (n_points**0.5)
    axis.plot(x/5,mean,color='black',
              label = 'Mean across trials')
    axis.fill_between(x/5,mean+rang,mean-rang,
                  color = color,
                  alpha = 0.3,
                  label = f"Standard {range_type}")
    if not normalize: axis.set_ylim((10,30)); axis.set_ylabel("Pupil diameter (pixels)")
    else: axis.set_ylim((0.6,1.2)); axis.set_ylabel("Pupil diameter (normalized)")
    axis.set_xlabel("Time since trial onset (s)")


def create_figure(df,title=None, render = True, normalize = False,
                  return_figure_and_artists=False):
    print(f"Initial shape of dataframe in create_figure call is {df.shape}")
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
    hit_axis.set_xticklabels([])
    hit_axis.set_xlabel("")
    miss_axis.set_xticklabels([])    
    miss_axis.set_xlabel("")
    miss_axis.set_yticklabels([])
    miss_axis.set_ylabel("")
    cr_axis.set_yticklabels([])
    cr_axis.set_ylabel("")
    
    #ADD A COLORBAR
    cmap._A = []
    cb = fig.colorbar(cmap,ax = ax, location="bottom", shrink = 0.4)
    cb.set_label("Trial Number")
    if title:
        fig.suptitle(title)
    if return_figure_and_artists:
        return (fig,ax)
    if render: fig.show()
    return (hits,misses,fas,crs)


def create_range_figure(df,title=None, normalize = False, 
                        range_type = "error",
                        return_figure_and_artists = False):
    df = df[~pd.isnull(df.Trial_ID)]
    fig,ax = plt.subplots(ncols = 2, nrows = 2, constrained_layout = True,
                          figsize = (12,8))
    (hit_axis, miss_axis), (fa_axis, cr_axis) = ax
    
    #HIT AXIS
    hits = plot_trial_subset_with_range(hit_axis, correct=True, go=True, df=df,
                             normalize = normalize, range_type = range_type, color="green")
    hit_axis.set_title("Hit Trials")
    hit_axis.legend()
    
    #MISS AXIS
    misses = plot_trial_subset_with_range(miss_axis, correct=False, go=True, df=df,
                               normalize = normalize,  range_type = range_type, color="palegreen")
    miss_axis.set_title("Miss Trials")
    
    #FALSE ALARM AXIS
    fas = plot_trial_subset_with_range(fa_axis, correct=False, go=False ,df=df,
                            normalize = normalize,  range_type = range_type, color="darksalmon")
    fa_axis.set_title("False-Alarm Trials")
    
    #CORRECT REJECTION AXIS
    crs = plot_trial_subset_with_range(cr_axis, correct=True, go=False, df=df,
                            normalize = normalize, range_type = range_type, color="red")
    cr_axis.set_title("Correct-Rejection Trials")
    
    
    #Turn off inner axis ticks and labels
    hit_axis.set_xticklabels([])
    hit_axis.set_xlabel("")
    miss_axis.set_xticklabels([])    
    miss_axis.set_xlabel("")
    miss_axis.set_yticklabels([])
    miss_axis.set_ylabel("")
    cr_axis.set_yticklabels([])
    cr_axis.set_ylabel("")
    
    if title:
        fig.suptitle(title)
    fig.show()
    if return_figure_and_artists: return (fig,ax)
    return (hits,misses,fas,crs)

def perform_testing(df, render = False, normalize = False):
    hits,misses,fas,crs = create_figure(df,render=render, normalize = normalize)
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

a = None

def plot_peritrial_subset(axis,correct,go,df,cmap, normalize = False, render = True):
    print(f"Initial shape of df in plot_trial_subset call is {df.shape}")
    x = np.arange(1,27)
    cond_list = [(df.trial_factor>0),(df.peritrial_factor>0)]
    df['extended_trial_factor'] = np.select(cond_list,(df.trial_factor,-1*df.peritrial_factor),default=np.nan)
    df["roi_num"] = np.fromiter(map(lambda s:s.split(" ")[-1],df.ROI_ID),int)
    
    
    trial_ids = df[(df.trial_factor==1)&(df.roi_num==0)].Trial_ID.values
    
    #We need to copy information about each trial to the pre-trial datapoints
    df2 = df.copy(deep=True)
    first_loop=True
    for trial in trial_ids:
        #Left rotate the trial mask 15 elements
        trial_idxs = (df.Trial_ID==trial)
        peritrial_idxs = trial_idxs.shift(periods = -15, fill_value=False)
        if first_loop:
            print(trial_idxs[30:50])
            print(peritrial_idxs[30:50])
            first_loop=False
        #Propogate trial information to pre-trial timepoints
        df2.loc[peritrial_idxs,'Trial_ID'] = df.loc[trial_idxs,'Trial_ID']
        df2.loc[peritrial_idxs,'go']       = df.loc[trial_idxs,'go']
        df2.loc[peritrial_idxs,'correct']  = df.loc[trial_idxs,'correct']
        
    global a
    a=df2
    df = df[df.go==go]
    df = df[df.correct==correct]
    recordings= list(map(lambda s:s.split(" ")[0],trial_ids))
    trial_ids = map(lambda s:s.split(" ")[-1],trial_ids)
    trial_ids = map(int,trial_ids)
    # pupils_per_timepoint = [df[df.trial_factor==i][df.roi_num==0].pupil_diameter.values for i in x]
    # length = max(map(len, pupils_per_timepoint))
    # pupils_per_timepoint= [list(ls)+[np.nan]*(length-len(ls)) for ls in pupils_per_timepoint]
    # pupils_per_timepoint = np.array(pupils_per_timepoint)
    pupils_per_timepoint = df[df.roi_num==0].pivot(index = "Trial_ID", 
                                                   columns = "extended_trial_factor", 
                                                   values = "pupil_diameter"
                                                   ).to_numpy()
    pupils_per_timepoint[pupils_per_timepoint=="NA"] = np.nan
    pupils_per_timepoint = pupils_per_timepoint.astype(float)

    print(f"Shape of plottable array is {pupils_per_timepoint.shape}")
    if normalize:
        #means of first second
        means = np.nanmean(pupils_per_timepoint[:,:5], axis = 0)
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


def create_peritrial_figure(df,title=None, render = True, normalize = False):
    print(f"Initial shape of dataframe in create_figure call is {df.shape}")

    #Clean up dataframe
    df = df[~(pd.isnull(df.Trial_ID) & df.peritrial_factor<0)]
    #Get an iterator over each peritrial period
    fig,ax = plt.subplots(ncols = 2, nrows = 2, constrained_layout = True,
                          figsize = (12,8))
    num_trials = 250
    norm = c.Normalize(vmin=0,vmax = num_trials)
    cmap = cm.ScalarMappable(norm,'plasma')
    
    (hit_axis, miss_axis), (fa_axis, cr_axis) = ax
    
    
    #HIT AXIS
    hits = plot_peritrial_subset(hit_axis, correct=True, go=True, df=df, cmap=cmap,
                             normalize = normalize, render = render)
    hit_axis.set_title("Hit Trials")
    hit_axis.legend()
    
    #MISS AXIS
    misses = plot_peritrial_subset(miss_axis, correct=False, go=True, df=df, cmap=cmap,
                               normalize = normalize, render = render)
    miss_axis.set_title("Miss Trials")
    
    #FALSE ALARM AXIS
    fas = plot_peritrial_subset(fa_axis, correct=False, go=False ,df=df, cmap=cmap,
                            normalize = normalize, render = render)
    fa_axis.set_title("False-Alarm Trials")
    
    #CORRECT REJECTION AXIS
    crs = plot_peritrial_subset(cr_axis, correct=True, go=False, df=df, cmap=cmap,
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


def perform_peritrial_testing(df, render = False, normalize = False):
    hits,misses,fas,crs = create_peritrial_figure(df,render=render, normalize = normalize)
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
    

def create_heatmap_figure(df):
    print(f"Initial shape of dataframe in create_figure call is {df.shape}")
    seaborn.set_style("dark")
    
    #Clean up dataframe
    df = df[~pd.isnull(df.Trial_ID)]
    df["roi_num"] = np.fromiter(map(lambda s:s.split(" ")[-1],df.ROI_ID),int)
    df = df[df.roi_num==0]
    df[df=="NA"] = np.nan
    
    
    hits = df[(df.correct==1)&(df.go==1)].pivot(index = "Trial_ID", 
                                               columns = "trial_factor", 
                                               values = "pupil_diameter"
                                               ).dropna(how='all').to_numpy()
    misses = df[(df.correct==0)&(df.go==1)].pivot(index = "Trial_ID", 
                                               columns = "trial_factor", 
                                               values = "pupil_diameter"
                                               ).dropna(how='all').to_numpy()
    fas = df[(df.correct==0)&(df.go==0)].pivot(index = "Trial_ID", 
                                               columns = "trial_factor", 
                                               values = "pupil_diameter"
                                               ).dropna(how='all').to_numpy()
    crs = df[(df.correct==1)&(df.go==0)].pivot(index = "Trial_ID", 
                                               columns = "trial_factor", 
                                               values = "pupil_diameter"
                                               ).dropna(how='all').to_numpy()
    fig,ax = plt.subplots(ncols = 4, nrows = 1, constrained_layout = True,
                          figsize = (12,8))
    
    (hit_axis, miss_axis, fa_axis, cr_axis) = ax
    
    min_len = min(len(a) for a in (hits,misses,fas,crs))
    #HIT AXIS
    x0 = hit_axis.imshow(hits[:min_len])
    hit_axis.set_title("Hit")
    hit_axis.set_ylabel("Trial")
    
    #MISS AXIS
    x1 = miss_axis.imshow(misses[:min_len])
    miss_axis.set_title("Miss")
    
    #FALSE ALARM AXIS
    x2 = fa_axis.imshow(fas[:min_len])
    fa_axis.set_title("False-Alarm")
    
    #CORRECT REJECTION AXIS
    x3 = cr_axis.imshow(crs[:min_len])
    cr_axis.set_title("Correct-Rejection")
    
    #Turn off inner axis ticks and labels
    for axis in ax:
        axis.set_xticks([])
        if not axis is hit_axis:
            axis.set_yticks([])
    fig.text(0.5,0.04,"Time",ha="center")
    
    images = [x0,x1,x2,x3]
    # Find the min and max of all colors for use in setting the color scale.
    vmin = min(image.get_array().min() for image in [x0,x1,x2,x3])
    vmax = max(image.get_array().max() for image in [x0,x1,x2,x3])
    norm = c.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=ax, orientation='vertical', fraction=.02)
    
    
    fig.show()

def create_final_figure():
    df = pd.read_csv("C:/Users/viviani/Desktop/full_datasets_for_analysis/left_only_high_contrast.csv")
    fig,ax = create_range_figure(df, 
                                 return_figure_and_artists=True,
                                 range_type="deviation")
    (hit_ax, miss_ax), (fa_ax, cr_ax) = ax
    df2 = pd.read_csv("C:/Users/viviani/Desktop/model predictions/mixed_linear_model_pupil_prediction.csv")
    PupilModelPredictionFigure.plot_with_confidence_interval(hit_ax, 
                                  df2[df2.trial_type=="hit"], 
                                  color="k",
                                  method="lines")
    PupilModelPredictionFigure.plot_with_confidence_interval(miss_ax, 
                                  df2[df2.trial_type=="miss"], 
                                  color="k",
                                  method="lines")
    PupilModelPredictionFigure.plot_with_confidence_interval(fa_ax, 
                                  df2[df2.trial_type=="fa"], 
                                  color="k",
                                  method="lines")
    PupilModelPredictionFigure.plot_with_confidence_interval(cr_ax, 
                                  df2[df2.trial_type=="cr"], 
                                  color="k",
                                  method="lines")
    hit_ax.legend()
    for a in ax.flatten(): a.set_ylim((11,19))
    return fig
if __name__=="__main__":
    # plt.close('all')
    # with np.errstate(all='raise'):
    #     df = pd.read_csv("C:/Users/viviani/Desktop/full_datasets_for_analysis/left_only_high_contrast.csv")
    #     # create_figure(df,normalize=True)
    #     create_range_figure(df,normalize=True,range_type="deviation")
    #     create_range_figure(df,normalize=True,range_type="error")
    fig = create_final_figure()
    fig.show()
    
    


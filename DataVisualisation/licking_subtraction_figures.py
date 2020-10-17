# -*- coding: __maiutf-8 -*-
"""
Created on Mon Oct  5 22:36:27 2020

@author: viviani
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns

import rpy2.robjects as r
from rpy2.robjects.packages import importr


base = importr("base")
stats = importr("stats")

from accdatatools.Utils.signal_processing import rising_edges,falling_edges
sns.set_style("darkgrid")
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 11



class LickingSubtractionValidationFigure:
    nrows = 5
    nplots = 8
    total_points = 4000
    training_points = total_points*4//5
        
    def __init__(self, df, state = np):
        #select a random ROI
        roi = df.ROI_ID.unique()[state]
        df  = df[df.ROI_ID == roi]
        df_on_f = df.dF_on_F.values[:self.total_points]
        x = np.linspace(0,len(df_on_f)//5, len(df_on_f))
        nontrial_idxs = pd.isnull(df.Trial_ID.values)[:self.total_points]
        trial_idxs = ~nontrial_idxs
        trial_starts, = np.array(np.where(rising_edges(trial_idxs,cutoff=0.5)))//5
        trial_ends,   = np.array(np.where(falling_edges(trial_idxs,cutoff=0.5)))//5
        self.fig = plt.figure(figsize=(10,11), tight_layout=True)
        gs = GridSpec(self.nrows, 8, self.fig)
        ax = np.zeros(self.nplots,dtype=object)
        ax[0] = self.fig.add_subplot(gs[0,:])
        ax[1] = self.fig.add_subplot(gs[1,:])
        ax[2] = self.fig.add_subplot(gs[2,:-2])
        ax[3] = self.fig.add_subplot(gs[2,-2:])
        ax[4] = self.fig.add_subplot(gs[3,0:2])
        ax[5] = self.fig.add_subplot(gs[3,3:5])
        ax[6] = self.fig.add_subplot(gs[3,6:8])
        ax[7] = self.fig.add_subplot(gs[4,:])
        minus_plot = self.fig.add_subplot(gs[3,2])
        minus_plot.set_axis_off(); minus_plot.text(0.5,0.5,"$\\bf{-}$",
                                                   transform=minus_plot.transAxes)
        equals_plot = self.fig.add_subplot(gs[3,5])
        equals_plot.set_axis_off(); equals_plot.text(0.5,0.5,"$\\bf{=}$",
                                                   transform=equals_plot.transAxes)
        dff_trial = df_on_f.copy(); dff_trial[nontrial_idxs]=np.nan
        dff_nontrial = df_on_f.copy(); dff_nontrial[trial_idxs]=np.nan
        
        ax[0].plot(x,dff_nontrial, color='k', label = "Fluorescnece outside trials")
        ax[0].plot(x,dff_trial,color=sns.color_palette()[3],
                   label="Fluorescence during trials (discarded)")
        ymax,ymin = ax[0].get_ylim()
        for start,end in zip(trial_starts,trial_ends):
            xy = (start,ymin)
            width = end-start
            height = ymax - ymin
            patch = Rectangle(xy,width,height, alpha=0.6)
            ax[0].add_patch(patch)
        patch.set_label("Trials")
        
        ax[1].plot(x,dff_nontrial, color = 'k')
        for a in ax[:2]:
            a.set_xlim(0,self.total_points//5)

        ymin,ymax = ax[1].get_ylim()
        xmin,xmax = ax[1].get_xlim()
        training_dataset = Rectangle((xmin,ymin), 0.8*(xmax-xmin),
                                     (ymax-ymin),
                                     color=sns.color_palette()[1],
                                     label = "Training Dataset",
                                     alpha = 0.6)
        testing_dataset = Rectangle(((xmin+ 0.8*(xmax-xmin)),ymin), 
                                    0.2*(xmax-xmin),
                                     (ymax-ymin),
                                     color=sns.color_palette()[2],
                                     label = "Testing Dataset",
                                     alpha = 0.6)
        for patch in (training_dataset,testing_dataset):
            ax[1].add_patch(patch)
        
        training_x = x[:self.training_points]
        testing_x  = x[self.training_points:self.total_points]
        training_dat = dff_nontrial[:self.training_points]
        training_licking = df.lick_factor[:self.training_points].values
        testing_dat = dff_nontrial[self.training_points:self.total_points]
        testing_licking = df.lick_factor[self.training_points:self.total_points]
        #define some R variables
        r.globalenv["dff"]    = r.FloatVector(dff_nontrial)
        r.globalenv["licking"]= r.FloatVector(df.lick_factor[:self.total_points].values)
        r.r("dat   <- data.frame(dff,licking)")
        r.r(f"train <- dat[1:{self.training_points},]")
        r.r(f"test  <- dat[-(1:{self.training_points}),]")
        r.r("model <- lm(dff~as.factor(licking),na.action=na.omit,dat=train)")
        r.r("preds <- predict(model,test)")
        model = r.r.model
        fitted_vals = np.array(r.r("predict(model,train)"))
        fitted_vals[trial_idxs[:self.training_points]] = np.nan
        preds = np.array(r.r.preds)
        preds[trial_idxs[self.training_points:self.total_points]] = np.nan
        ax[2].plot(training_x,training_dat, 
                   color = 'k')
        ax[2].plot(training_x,fitted_vals, color=sns.color_palette()[1],
                   label = "Fitted model values based on licking (significant)")
        ax[3].plot(x[self.total_points*4//5:self.total_points],dff_nontrial[self.total_points*4//5:self.total_points], 
                   color = 'k')
        ax[3].plot(testing_x,preds, color = sns.color_palette()[2],
                   label = "Model predictions")
        ax[3].set_yticklabels([])
        ax[4].plot(testing_x,testing_dat,color='k',label="Testing data")
        ax[5].plot(testing_x,preds, color=sns.color_palette()[2],label="Prediction")
        ax[6].plot(testing_x,testing_dat - preds, color = 'k',label="Licking-adjusted fluorescence")
        for idx,a in enumerate(ax):
            if idx in (0,1,2,3,4,7):
                a.set_ylabel("Fluorescence ($\\Delta$F/F0 units)")
        ymin,ymax = (min(a.get_ylim()[0] for a in ax), 
                     max(a.get_ylim()[1] for a in ax))

        for a in ax[-3:-1]:
            a.set_yticklabels([])
        ax[-1].plot(testing_x,testing_dat-preds, color='k',
                    label = "Licking-adjusted fluroescence")
        r.r("res <- test")
        r.r("res$dff = res$dff - preds")
        r.r("model2 <- lm(dff~as.factor(licking),na.action=na.omit,dat=res)")
        newpreds = np.array(r.r("predict(model2,test)"))
        newpreds[trial_idxs[self.training_points:self.total_points]] = np.nan
        ax[-1].plot(testing_x,newpreds,color=sns.color_palette()[3],
                    label = "A new fitted model, based on licking (NOT significant))")
        for a in ax:
            a.set_ylim((ymin,ymax))
            a.legend(loc="upper right")
            a.set_xticklabels([])
        ax[3].set_ylabel("")
        for a,char in zip(ax,"ABCDEFGE"):
            a.set_title("$\\bf{(" + f"{char}" +")}$", loc="right")
    def show(self):
        self.fig.show()
        
#This should probably inherit from the previous figure and just
#overrive the __init__ method
class LickingSubtractionThenCollapsingWorkflow:
    npoints = 750
    def __init__(self,df,state=0):
        fig,ax = plt.subplots(nrows = 5, figsize = (10,11), tight_layout=True)
        roi = df.sample(random_state=state).ROI_ID.values[0]
        df  = df[df.ROI_ID == roi]
        df_on_f = df.dF_on_F.values[:self.npoints]
        x = np.linspace(0,len(df_on_f)//5, len(df_on_f))
        nontrial_idxs = pd.isnull(df.Trial_ID.values)[:self.npoints]
        trial_idxs = ~nontrial_idxs
        trial_starts, = np.array(np.where(rising_edges(trial_idxs,cutoff=0.5)))//5
        trial_ends,   = np.array(np.where(falling_edges(trial_idxs,cutoff=0.5)))//5
        dff_trial = df_on_f.copy(); dff_trial[nontrial_idxs]=np.nan
        dff_nontrial = df_on_f.copy(); dff_nontrial[trial_idxs]=np.nan
        ax[0].plot(x,dff_trial,color=sns.color_palette('bright')[0],
                   label="Fluorescence during trials")
        ax[0].plot(x,dff_nontrial, color=sns.color_palette()[1], 
                   label = "Fluorescnece outside trials")
        ymax,ymin = ax[0].get_ylim()
        for start,end in zip(trial_starts,trial_ends):
            xy = (start,ymin)
            width = end-start
            height = ymax - ymin
            patch = Rectangle(xy,width,height, alpha=0.6)
            ax[0].add_patch(patch)
        r.globalenv["dff"]    = r.FloatVector(dff_nontrial)
        r.globalenv["licking"]= r.FloatVector(df.lick_factor[:self.npoints].values)
        r.r("nontrial   <- data.frame(dff,licking)")
        r.globalenv["dff"]    = r.FloatVector(dff_trial)
        r.globalenv["licking"]= r.FloatVector(df.lick_factor[:self.npoints].values)
        r.r("trial <- data.frame(dff,licking)")
        r.r("model <- lm(dff~as.factor(licking),na.action=na.omit,dat=nontrial)")
        r.r("preds <- predict(model,trial)")
        patch.set_label("Trials")
        ax[1].plot(x,dff_nontrial, color = sns.color_palette()[1],label="Fluorescence outside trials")
        ax[1].plot(x,np.array(r.r.preds),color=sns.color_palette()[3], 
                   label="Fitted licking model")
        ax[2].plot(x,dff_trial,color=sns.color_palette('bright')[0])
        ax[2].plot(x,np.array(r.r.preds),color=sns.color_palette()[3],
                    label = "Model predictions")
        ax[3].plot(x,(dff_trial-np.array(r.r.preds)),
                   color = sns.color_palette('bright')[0],
                   label = "Licking-corrected trial fluorescence")
        for start,end in zip(trial_starts,trial_ends):
            xy = (start,ymin)
            width = end-start
            height = ymax - ymin
            patch1 = Rectangle(xy,width,height, alpha=0.3)
            patch2 = Rectangle(xy,width,height, alpha=0.3)
            ax[3].add_patch(patch1)
            ax[4].add_patch(patch2)
        trials = dff_trial[~np.isnan(dff_trial)].reshape(-1,25)
        tone = trials[:,:5].mean(axis=-1)
        stim = trials[:,5:15].mean(axis=-1)
        resp = trials[:,15:25].mean(axis=-1)
        tone_idxs = trial_starts + 0.5
        stim_idxs = trial_starts + 2
        resp_idxs = trial_starts + 4
        for x1,x2,x3,y1,y2,y3 in zip(tone_idxs,stim_idxs,resp_idxs,tone,stim,resp):
            ax[4].plot((x1,x2,x3),(y1,y2,y3),color='k',marker='o',
                       label = "Collapsed data" if x1==tone_idxs[0] else "")
        for a in ax[:-1]: a.set_xticklabels([])
        for a in ax: a.set_xlim(ax[0].get_xlim()); a.legend(loc='upper right')
        for a,char in zip(ax,"ABCDEFG"):
            a.set_title("$\\bf{(" + f"{char}" +")}$", loc="right")
        ax[2].set_ylabel("Fluorescence ($\Delta$F/F0 units)")
        ax[-1].set_xlabel("Time (s)")
        fig.show()
        
        
        

if __name__=="__main__":
    plt.close('all')
    df = pd.read_csv(r"C:\Users\viviani\Desktop\single_experiments_for_testing\2016-11-01_03_CFEB027.csv")
    LickingSubtractionValidationFigure(df,180).show()
    LickingSubtractionThenCollapsingWorkflow(df,5).show()

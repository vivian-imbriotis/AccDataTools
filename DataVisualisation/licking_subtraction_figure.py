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
    def __init__(self, df, state = np):
        #select a random ROI
        roi = df.sample(random_state=state).ROI_ID.values[0]
        df  = df[df.ROI_ID == roi]
        df_on_f = df.dF_on_F.values[:1250]
        x = np.linspace(0,len(df_on_f)//5, len(df_on_f))
        nontrial_idxs = pd.isnull(df.Trial_ID.values)[:1250]
        trial_idxs = ~nontrial_idxs
        trial_starts, = np.array(np.where(rising_edges(trial_idxs,cutoff=0.5)))//5
        trial_ends,   = np.array(np.where(falling_edges(trial_idxs,cutoff=0.5)))//5
        fig = plt.figure(figsize=(10,11), tight_layout=True)
        gs = GridSpec(self.nrows, 8, fig)
        ax = np.zeros(self.nplots,dtype=object)
        ax[0] = fig.add_subplot(gs[0,:])
        ax[1] = fig.add_subplot(gs[1,:])
        ax[2] = fig.add_subplot(gs[2,:-2])
        ax[3] = fig.add_subplot(gs[2,-2:])
        ax[4] = fig.add_subplot(gs[3,0:2])
        ax[5] = fig.add_subplot(gs[3,3:5])
        ax[6] = fig.add_subplot(gs[3,6:8])
        ax[7] = fig.add_subplot(gs[4,:])
        minus_plot = fig.add_subplot(gs[3,2])
        minus_plot.set_axis_off(); minus_plot.text(0.5,0.5,"$\\bf{-}$",
                                                   transform=minus_plot.transAxes)
        equals_plot = fig.add_subplot(gs[3,5])
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
            a.set_xlim(0,1250//5)

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
        
        training_x = x[:1000]
        testing_x  = x[1000:1250]
        training_dat = dff_nontrial[:1000]
        training_licking = df.lick_factor[:1000].values
        testing_dat = dff_nontrial[1000:1250]
        testing_licking = df.lick_factor[1000:1250]
        #define some R variables
        r.globalenv["dff"]    = r.FloatVector(dff_nontrial)
        r.globalenv["licking"]= r.FloatVector(df.lick_factor[:1250].values)
        r.r("dat   <- data.frame(dff,licking)")
        r.r("train <- dat[1:1000,]")
        r.r("test  <- dat[-(1:1000),]")
        r.r("model <- lm(dff~as.factor(licking),na.action=na.omit,dat=train)")
        r.r("preds <- predict(model,test)")
        model = r.r.model
        fitted_vals = np.array(r.r("predict(model,train)"))
        fitted_vals[trial_idxs[:1000]] = np.nan
        preds = np.array(r.r.preds)
        preds[trial_idxs[1000:1250]] = np.nan
        ax[2].plot(training_x,training_dat, 
                   color = 'k')
        ax[2].plot(training_x,fitted_vals, color=sns.color_palette()[1],
                   label = "Fitted model values based on licking (significant)")
        ax[3].plot(x[1250*4//5:1250],dff_nontrial[1250*4//5:1250], 
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
        newpreds[trial_idxs[1000:1250]] = np.nan
        ax[-1].plot(testing_x,newpreds,color=sns.color_palette()[3],
                    label = "A new fitted model, based on licking (NOT significant))")
        for a in ax:
            a.set_ylim((ymin,ymax))
            a.legend(loc="upper right")
            a.set_xticklabels([])
        ax[3].set_ylabel("")
        fig.show()
        
        
        

        

if __name__=="__main__":
    plt.close('all')
    df = pd.read_csv(r"C:\Users\viviani\Desktop\single_experiments_for_testing\2016-11-01_03_CFEB027.csv")
    i=1
    LickingSubtractionValidationFigure(df,i)

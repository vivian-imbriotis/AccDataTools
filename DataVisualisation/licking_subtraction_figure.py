# -*- coding: __maiutf-8 -*-
"""
Created on Mon Oct  5 22:36:27 2020

@author: viviani
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns

from accdatatools.Utils.signal_processing import rising_edges,falling_edges


class LickingSubtractionValidationFigure:
    def __init__(self, df):
        #select a random ROI
        roi = df.sample().ROI_ID.values[0]
        df  = df[df.ROI_ID == roi]
        df_on_f = df.dF_on_F.values
        nontrial_idxs = pd.isnull(df.Trial_ID.values)
        trial_idxs = ~nontrial_idxs
        trial_starts, = np.where(rising_edges(trial_idxs,cutoff=0.5))
        trial_ends,   = np.where(falling_edges(trial_idxs,cutoff=0.5))
        fig,ax = plt.subplots(nrows = 4, figsize=(10,11))
        ax[0].plot(df_on_f, color='k', label="Fluorescence of ROI")
        ymax,ymin = ax[0].get_ylim()
        for start,end in zip(trial_starts,trial_ends):
            print(start,end)
            xy = (start,ymin)
            width = end-start
            height = ymax - ymin
            print(width,height)
            patch = Rectangle(xy,width,height, alpha=0.6)
            ax[0].add_patch(patch)
        patch.set_label("Trials")
        ax[0].legend()
        dff_trial = df_on_f.copy(); dff_trial[nontrial_idxs]=np.nan
        dff_nontrial = df_on_f.copy(); dff_nontrial[trial_idxs]=np.nan
        
        ax[1].plot(dff_nontrial, color='k', label = "Fluorescnece outside trials")
        ax[1].plot(dff_trial,color=sns.color_palette()[3],
                   label="Fluorescence during trials (discarded)")
        
        ax[2].plot(dff_nontrial, color = 'k')
        for a in ax:
            a.set_xlim(0,1250)
        ymin,ymax = ax[2].get_ylim()
        xmin,xmax = ax[2].get_xlim()
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
            ax[2].add_patch(patch)
        for a in ax:
            a.legend(loc="upper right")
            a.set_ylabel("Fluorescence ($\\Delta$F/F0 units)")
        fig.show()
        
        
        

        

if __name__=="__main__":
    df = pd.read_csv(r"C:\Users\viviani\Desktop\single_experiments_for_testing\2016-11-01_03_CFEB027.csv")
    LickingSubtractionValidationFigure(df)

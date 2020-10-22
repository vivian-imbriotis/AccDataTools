# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 21:26:47 2020

@author: viviani
"""

import matplotlib.pyplot as plt
import numpy as np

from accdatatools.Timing.synchronisation import get_lick_times
from accdatatools.Observations.trials import get_trials_in_recording
from accdatatools.Utils.path import get_timeline_path, get_exp_path

class LickingRasterFigure:
    def __init__(self, exp_id, drive="H:"):
        exp_path      = get_exp_path(exp_id,drive)
        timeline_path = get_timeline_path(exp_path)
        trials        = get_trials_in_recording(exp_path,ignore_dprime=True,
                                         use_sparse=True)
        licks = get_lick_times(timeline_path)
        
        self.fig,ax = plt.subplots(ncols=2,figsize=[8,5])
        for idx,trial in enumerate(trials):
            trial_licks = licks-trial.start_stimulus
            trial_licks = trial_licks[np.logical_and(trial_licks>0,trial_licks<5)]
            ax[0 if trial.isgo else 1].plot(
                trial_licks,
                np.full(trial_licks.shape,idx),
                'o',markersize=2,
                color="k")
            
    
        for a in ax:
            ymin,ymax = a.get_ylim()
            a.vlines((0,1,3),ymin,ymax,linestyles='dashed',color = 'k')
            for name,pos in zip(('Tone','Stimulus','Response'),(0,1,3)):
                    a.text(pos+0.1,ymax,name,
                        horizontalalignment='left',
                        verticalalignment='bottom')
                    a.set_xlabel(r'$\Delta$t from trial onset (s)')
                    a.set_xlim((-1,5))
        ax[0].set_title("Go Trials")
        ax[1].set_title("No-Go Trials")
        ax[0].set_ylabel('Trial Number')
    def show(self):
        self.fig.show()
                
if __name__=="__main__":
    LickingRasterFigure('2016-11-01_03_CFEB027').show()
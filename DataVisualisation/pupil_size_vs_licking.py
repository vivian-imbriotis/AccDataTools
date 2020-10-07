# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 11:36:41 2020

@author: Vivian Imbriotis
"""



import numpy as np
import seaborn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns


from accdatatools.Timing.synchronisation import (get_lick_times,
                                                 get_eyecam_frame_times)
from accdatatools.ProcessPupil.size import get_pupil_size_at_each_eyecam_frame
from accdatatool.Observations.trials import (SparseTrial,
                                             _get_trial_structs)
from accdatatools.Utils.path import (get_timeline_path, 
                                     get_psychstim_path,
                                     get_pupil_hdf_path)

sns.set_style("darkgrid")
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 11

class PupilDiameterLickingFigure:
    @staticmethod
    def color_of_trial(trial):
        if trial.isgo:
            if trial.correct: return "green"
            else:             return "palegreen"
        else:
            if trial.correct: return "red"
            else:             return "darksalmon"

    def __init__(self,exp_path):
        timeline_path  = get_timeline_path(exp_path)
        psychstim_path = get_psychstim_path(exp_path)
        hdf_path       = get_pupil_hdf_path(exp_path)
        structs        = _get_trial_structs(psychstim_path)
        self.trials             = [SparseTrial(struct) for struct in structs]
        self.eyecam_frame_times = get_eyecam_frame_times(timeline_path)
        self.licking_times      = get_lick_times(timeline_path)
        self.pupil_diameters    = get_pupil_size_at_each_eyecam_frame(hdf_path)
        self.render()
        
    def render(self):
        self.fig, ax  = plt.subplots()
        y_val_for_licks = np.full(self.licking_times.shape, 6)
        ax.scatter(self.licking_times,
                   y_val_for_licks,
                   c = "black")
        ax.plot(self.eyecam_frame_times,
                self.pupil_diameters)
        trial_starts = np.array([trial.start_stimulus for trial in self.trials])-1
        trial_ends  = trial_starts + 5
        colors       = [self.color_of_trial(trial) for trial in self.trials]
        ymax,ymin = ax.get_ylim()
        for trial_start,trial_end,c in zip(trial_starts,trial_ends,colors):
            rect = Rectangle((trial_start,ymin),
                             trial_end - trial_start,
                             ymax - ymin,
                             color = c,
                             fill = True,
                             alpha = 0.2)
            ax.add_patch(rect)
    def show(self):
        self.fig.show()
         
        
        
        
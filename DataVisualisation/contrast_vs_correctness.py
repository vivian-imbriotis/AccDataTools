# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:51:45 2020

@author: Vivian Imbriotis
"""
from string import ascii_uppercase

import matplotlib.pyplot as plt
import seaborn as sns

from accdatatools.Observations.trials import get_trials_in_recording
from accdatatools.Utils.map_across_dataset import apply_to_all_recordings_of_class

sns.set_style("darkgrid")
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 11

class CorrectnessContrastFigure:
    def __init__(self, drive = "H:\\"):
        ls = []
        f =  lambda pth: ls.extend(get_trials_in_recording(pth,
                                                 use_sparse=True))
        apply_to_all_recordings_of_class("both_sides_high_contrast",
                                         drive, f)
        apply_to_all_recordings_of_class("low_contrast",
                                         drive, f)
        contrast_10_percent = [t for t in ls if t.contrast==0.1]
        contrast_50_percent = [t for t in ls if t.contrast==0.5]
        contrast_100_percent = [t for t in ls if t.contrast==1]
        self.fig,ax = plt.subplots(ncols=3, figsize=(8,4))
        for axis,trials,name,letter in zip(ax,(contrast_100_percent,
                                               contrast_50_percent,
                                               contrast_10_percent),
                                               ("100%","50%","10%"),
                                               ascii_uppercase):
            correct   = len([t for t in trials if t.correct])
            incorrect = len([t for t in trials if not t.correct])
            axis.pie((correct,incorrect),labels = ("Correct","Incorrect"),
                     autopct='%1.1f%%')
            axis.set_xlabel(f"Stimulus Contrast {name}")
            axis.set_title("$\\bf{(" + letter + ")}$",loc="right")
    def show(self):
        self.fig.show()
    def save(self,path,dpi=600):
        self.fig.savefig(path,dpi=dpi)

if __name__=="__main__":
    CorrectnessContrastFigure().save("C:/Users/Viviani/Desktop/newfigure.png")
        
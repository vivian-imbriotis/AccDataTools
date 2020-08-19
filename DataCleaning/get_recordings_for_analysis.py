# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 21:40:19 2020

@author: Vivian Imbriotis
"""

#yikes, both of these files should probably get this function
#from an abstract file
import os
import time
import pickle as pkl
from copy import copy
import random

seed = 987654321
random.seed(seed)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from accdatatools.Summaries.produce_trial_summary_document import (
    get_classes_of_recording,
    get_count_and_stats)
from accdatatools.Utils.path import get_exp_path,get_psychstim_path,get_exp_id
from accdatatools.DataCleaning.determine_dprime import calc_d_prime
from accdatatools.DataVisualisation.mean_videos import AnimalMeanVideoFigure

class RecordingClassSummary:
    def __init__(self,df,signature,dic,name,drive = "H://"):
        ls_of_recordings = dic[signature]
        self.recordings = ls_of_recordings
        self.df = df
        recording_paths = list(map(lambda p:get_exp_path(p,drive),
                                        self.recordings))
        psychstims = [get_psychstim_path(r) for r in recording_paths]
        self.dprimes = [calc_d_prime(r) for r in psychstims]
        self.df_repr = pd.DataFrame((self.recordings,self.dprimes)).transpose()
        self.df_repr.columns = ["recording_id","dprime"]
        self.df_repr[self.df_repr==np.inf] = np.nan
        self.df_repr = self.df_repr[self.df_repr.dprime>1]
        self.recordings = self.df_repr.recording_id.values
        subset = self.df[self.df.recording_id.isin(self.recordings)]
        self.pooled_stats = get_count_and_stats(subset,subsetting=False)
        self.name = name
        self.signature = signature
        self.drive = drive
        self.recording_paths = list(map(lambda p:get_exp_path(p,drive),
                                        self.recordings))
    def set_recordings(self,ls_of_exp_paths):
        ls_of_exp_ids = [get_exp_id(i) for i in ls_of_exp_paths]
        self.recordings = ls_of_exp_ids
        self.df_repr[self.df_repr.recording_id.isin(self.recordings)]
        subset = self.df[self.df.recording_id.isin(self.recordings)]
        self.pooled_stats = get_count_and_stats(subset,subsetting=False)
        self.recording_paths = ls_of_exp_paths
    def __repr__(self):
        intro = f"{self.name} recordings with pooled stats {self.pooled_stats}"
        varying = "This class varied across parameters like so:"
        params = map(lambda a:a.__repr__(),self.signature)
        sep = "-------------\n"
        num = f"There are {len(self.df_repr)} recordings in this class with dprime>1"
        lines_to_print = [intro,varying,*params,sep,num,sep,
                          self.df_repr.__repr__()]
        return "\n".join(lines_to_print)


recording_classes, df, planes = get_classes_of_recording(root="D://")


both_sides_high_contrast = (('affirmative', (False, 1)),
                            ('contrast', (1.0,)),
                            ('correct', (0.0, 1.0)),
                            ('go', (0.0, 1.0)),
                            ('side', ('left', 'right')),
                            ('task', ('bGoNoGoLickAdapt',)),
                            ('test', (False, True)))
 

low_contrast =  (('affirmative', (False, 1)),
                  ('contrast', (0.1, 0.5)),
                  ('correct', (0.0, 1.0)),
                  ('go', (0.0, 1.0)),
                  ('side', ('left', 'right')),
                  ('task', ('bGoNoGoLickFull',)),
                  ('test', (False, True)))
                 

left_only_high_contrast = (('affirmative', (False, 1)),
                            ('contrast', (1.0,)),
                            ('correct', (0.0, 1.0)),
                            ('go', (0.0, 1.0)),
                            ('side', ('left',)),
                            ('task', ('bGoNoGoLickAdaptOne',)),
                            ('test', (False, True)))
 
both_sides_high_contrast_summary = RecordingClassSummary(
                                            df,
                                            both_sides_high_contrast,
                                            recording_classes,
                                            name = "Bilateral High Contrast",
                                            drive = "D://"
                                            )

left_only_high_contrast_summary = RecordingClassSummary(
                                            df,
                                            left_only_high_contrast,
                                            recording_classes,
                                            name = "Left High Contrast",
                                            drive = "D://"
                                            )

low_contrast_summary = RecordingClassSummary(df,
                                            low_contrast,
                                            recording_classes,
                                            name = "Low Contrast",
                                            drive = "D://"
                                            )


# Now we need to pare down the recording classes to remove excess duplicate
# recordings of the same cortical region of the same animal, so one neuron
# is not assigned multiple ROIs across recordings!
    
# we need a function :: [exp_ids] -> [[exp_ids]]
# that maps a list of exp_ids of one animal to a list of lists of experiment ids
# that imaged a single cortical area. The best way to do this without training
# data is probably just visual inspection, because we don't need to do it for very
# many experiments.

We do this by composing two functions - one to go from a list
of all recordings to a list of lists of each animals recordings,
one to go from a list of 

def subtype_experiments_by_mouse(ls_of_exp_paths):
    ls_of_exp_paths = copy(ls_of_exp_paths)
    exps_by_mouse = []
    mouse_paths = set([os.path.split(s)[0] for s in ls_of_exp_paths])
    exps_by_mouse = [list(filter(lambda s:mouse_path in s,ls_of_exp_paths))
                     for mouse_path in mouse_paths]
    return sorted(exps_by_mouse)

def manually_group_recordings_by_cortical_area(ls_of_exp_paths):
    plt.ion()
    subgroups = []
    remaining_exp_paths = copy(ls_of_exp_paths)
    print("""
          Identify a group of these mean images that show the same
          area of cortex. Provide a space-seperated 0-based list of 
          intergers to indicate each member of the group, then press enter.
          If all remaining mean images show different regions of cortex, 
          press enter without entering a list.""")
    while remaining_exp_paths:
        fig = AnimalMeanVideoFigure(ls_of_exp_paths = remaining_exp_paths)
        try:
            print(remaining_exp_paths)
            fig.canvas.manager.window.setGeometry(0,50,900,1000)
            ls = []
            line = ""
            fig.show()
            for i in range(15):
                time.sleep(0.05)
                fig.canvas.draw_idle()
                plt.pause(.1)
            line = input("-> ")
            fig.show()
            ls.extend([int(i) for i in line.split(" ") if i!=""])
            if ls:
                subgroups.append([remaining_exp_paths[i] for i in ls])
                remaining_exp_paths = [e for i,e in enumerate(remaining_exp_paths)
                                       if i not in ls]
            else:
                subgroups.extend([[path] for path in remaining_exp_paths])
                remaining_exp_paths = []
        except Exception as e:
            raise e #debugging
            print("Something went wrong. Enter a space-seperated list of ints.")
        finally:
            plt.close(fig.fig)
    return subgroups
           
def manually_group_recording_class_by_cortical_area(summary_obj):
    res = []
    for single_mouses_exps in subtype_experiments_by_mouse(
            summary_obj.recording_paths):
        res.extend(
            manually_group_recordings_by_cortical_area(
                single_mouses_exps)
            )
    #drop multiple recordings of the same cortical area
    return
    unique_cortical_areas = [random.choice(i) for i in res]
    return_obj = copy(summary_obj)
    return_obj.set_recordings(unique_cortical_areas)
    return return_obj





if __name__=="__main__":

    both_sides_high_contrast = (('affirmative', (False, 1)),
                            ('contrast', (1.0,)),
                            ('correct', (0.0, 1.0)),
                            ('go', (0.0, 1.0)),
                            ('side', ('left', 'right')),
                            ('task', ('bGoNoGoLickAdapt',)),
                            ('test', (False, True)))
 

    low_contrast =  (('affirmative', (False, 1)),
                      ('contrast', (0.1, 0.5)),
                      ('correct', (0.0, 1.0)),
                      ('go', (0.0, 1.0)),
                      ('side', ('left', 'right')),
                      ('task', ('bGoNoGoLickFull',)),
                      ('test', (False, True)))
                     
    
    left_only_high_contrast = (('affirmative', (False, 1)),
                                ('contrast', (1.0,)),
                                ('correct', (0.0, 1.0)),
                                ('go', (0.0, 1.0)),
                                ('side', ('left',)),
                                ('task', ('bGoNoGoLickAdaptOne',)),
                                ('test', (False, True)))
     
    both_sides_high_contrast_summary = RecordingClassSummary(
                                                df,
                                                both_sides_high_contrast,
                                                recording_classes,
                                                name = "Bilateral High Contrast",
                                                drive = "D://"
                                                )
    
    left_only_high_contrast_summary = RecordingClassSummary(
                                                df,
                                                left_only_high_contrast,
                                                recording_classes,
                                                name = "Left High Contrast",
                                                drive = "D://"
                                                )
    
    low_contrast_summary = RecordingClassSummary(df,
                                                low_contrast,
                                                recording_classes,
                                                name = "Low Contrast",
                                                drive = "D://"
                                                )

    lc_grouped = manually_group_recording_class_by_cortical_area(
        low_contrast_summary)
    lohc_grouped = manually_group_recording_class_by_cortical_area(
        left_only_high_contrast_summary)
    bshc_grouped = manually_group_recording_class_by_cortical_area(
        both_sides_high_contrast_summary)

    with open("both_sides_high_contrast_uncleaned.pkl",'wb') as file:
        pkl.dump(both_sides_high_contrast_summary.recording_paths, file)
    with open("left_only_high_contrast_uncleaned.pkl",'wb') as file:
        pkl.dump(left_only_high_contrast_summary.recording_paths, file)
    with open("low_contrast_uncleaned.pkl",'wb') as file:
        pkl.dump(low_contrast_summary.recording_paths, file)
    with open("both_sides_high_contrast.pkl",'wb') as file:
        pkl.dump(bshc_grouped.recording_paths, file)
    with open("left_only_high_contrast.pkl",'wb') as file:
        pkl.dump(lohc_grouped.recording_paths, file)
    with open("low_contrast.pkl",'wb') as file:
        pkl.dump(lc_grouped.recording_paths, file)
    
    for i in (bshc_grouped,
              lohc_grouped,
              lc_grouped):
        print("\n\n\n")
        print(i)
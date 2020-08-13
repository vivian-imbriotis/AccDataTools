# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 21:40:19 2020

@author: Vivian Imbriotis
"""

#yikes, both of these files should probably get this function
#from an abstract file
import pickle as pkl

import pandas as pd
import numpy as np

from accdatatools.Summaries.produce_trial_summary_document import (
    get_classes_of_recording,
    get_count_and_stats)
from accdatatools.Utils.path import get_exp_path,get_psychstim_path
from accdatatools.DataCleaning.determine_dprime import calc_d_prime

class RecordingClassSummary:
    def __init__(self,df,signature,dic,name,drive = "D://"):
        ls_of_recordings = dic[signature]
        self.recordings = ls_of_recordings
        recording_paths = list(map(lambda p:get_exp_path(p,drive),
                                        self.recordings))
        psychstims = [get_psychstim_path(r) for r in recording_paths]
        self.dprimes = [calc_d_prime(r) for r in psychstims]
        self.df_repr = pd.DataFrame((self.recordings,self.dprimes)).transpose()
        self.df_repr.columns = ["recording_id","dprime"]
        self.df_repr[self.df_repr==np.inf] = np.nan
        self.df_repr = self.df_repr[self.df_repr.dprime>1]
        self.recordings = self.df_repr.recording_id.values
        subset = df[df.recording_id.isin(self.recordings)]
        self.pooled_stats = get_count_and_stats(subset,subsetting=False)
        self.name = name
        self.signature = signature
        self.recording_paths = list(map(lambda p:get_exp_path(p,drive),
                                        self.recordings))
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
                                            )

left_only_high_contrast_summary = RecordingClassSummary(
                                            df,
                                            left_only_high_contrast,
                                            recording_classes,
                                            name = "Left High Contrast",
                                            )

low_contrast_summary = RecordingClassSummary(df,
                                            low_contrast,
                                            recording_classes,
                                            name = "Low Contrast",
                                            )

for i in (both_sides_high_contrast_summary,
          left_only_high_contrast_summary,
          low_contrast_summary):
    print(i)
    print("\n\n\n")

with open("both_sides_high_contrast.pkl",'wb') as file:
    pkl.dump(both_sides_high_contrast_summary.recording_paths, file)
with open("left_only_high_contrast.pkl",'wb') as file:
    pkl.dump(left_only_high_contrast_summary.recording_paths, file)
with open("low_contrast.pkl",'wb') as file:
    pkl.dump(low_contrast_summary.recording_paths, file)

    
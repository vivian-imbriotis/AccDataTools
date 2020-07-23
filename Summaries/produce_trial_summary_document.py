# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 11:43:16 2020

@author: viviani

Create a summary dataframe and summary document detailing the composition
of each recording session in terms of the trial attributes - eg how often
mice were correct, their d-prime statistics, whether the side varied in the 
trial, whether contrast varied in the trial, and so on...

"""
import os
from collections import namedtuple
from contextlib import redirect_stdout

import pandas as pd

from accdatatools.Observations.trials import _get_trial_structs, SparseTrial
from accdatatools.Utils.map_across_dataset import apply_to_all_recordings
from accdatatools.Utils.convienience import item
from accdatatools.Utils.path import exp_id
from accdatatools.Utils.map_across_dataset import no_of_planes
from accdatatools.DataCleaning.determine_dprime import d_prime

class StatisticRepresenter:
    def __init__(self,count,acc,sen,spe,d,test,leftside,go):
        self.strings = []
        for var in (count,acc,sen,spe,d,test,leftside,go):
            if var in (count,d):
                self.strings.append(f"{var:.1f}" if var!=None else "NA")
            else:
                self.strings.append(f"{100*var:.0f}%" if var!=None else "NA")
    def __repr__(self):
        return(f"(Count={self.strings[0]}, "+
               f"Acc={self.strings[1]}, "+
               f"Sen={self.strings[2]}, "+
               f"Spe={self.strings[3]}, "+
               f"D'={self.strings[4]}, "+
               f"test={self.strings[5]}, "+
               f"left={self.strings[6]}, "+
               f"go={self.strings[7]})"
               )


class ParentedSparseTrial(SparseTrial):
    '''
    A SparseTrial Object that also knows in which recording it was collected.
    '''
    def __init__(self,struct,exp_path,tolerant=False):
        self.recording_id = exp_id(exp_path)
        super().__init__(struct,tolerant)
    def to_dict(self):
        result = super().to_dict()
        result["recording_id"] = self.recording_id
        return result


def get_count_and_stats(dataframe,attribute,value):
    subset = dataframe[dataframe[attribute]==value]
    
    count             = len(subset.index)
    try: tests        = len(subset[subset.test==True].index)/count
    except ZeroDivisionError: tests = None
    try:gos           = len(subset[subset.go==True].index)/count
    except ZeroDivisionError: gos = None
    left              = len(subset[subset.side=='left'])
    right             = len(subset[subset.side=='right'])
    try:left          = left / (left+right)
    except ZeroDivisionError: left = None
    corrects          = subset[subset.correct==True]
    incorrects        = subset[subset.correct==False]
    
    hit               = len(corrects[corrects.go==True].index)
    miss              = len(incorrects[incorrects.go==True].index)
    correct_rejection = len(corrects[corrects.go==False].index)
    false_alarm       = len(incorrects[incorrects.go==False].index)
    try: accurracy   = len(corrects.index) / (len(corrects.index) + len(incorrects.index))
    except ZeroDivisionError: accurracy = None
    try: sensitivity = hit / (hit + false_alarm)
    except ZeroDivisionError: sensitivity = None
    try: specificity = correct_rejection / (correct_rejection + miss)
    except ZeroDivisionError: specificity = None
    d = d_prime(sensitivity, (1-specificity)) if sensitivity and specificity else None
    return StatisticRepresenter(count,accurracy,sensitivity,specificity,d,
                                tests,left,gos)

def get_counts_accuracies_dprimes_for_each_unique_attribute(df):
    pass
    

def get_timeline_path(exp_path):
    file = item([file for file in os.listdir(exp_path) if "Timeline" in file])
    return os.path.join(exp_path,file)

def get_psychstim_path(exp_path):
    file = item([file for file in os.listdir(exp_path) if "psychstim" in file])
    return os.path.join(exp_path,file)

def get_unique_attribute_values(df):
    result = {}
    for column in df.columns:
        result[column] = df[column].unique()
    return result
          

def get_every_trial():
    ls = []
    planes = []
    def func(path):
        path2 = get_psychstim_path(path)
        structs = _get_trial_structs(path2)
        try:
            trial_objects = [ParentedSparseTrial(struct,path,tolerant=True).to_dict() for struct in structs]
            ls.extend(trial_objects)
            plane = no_of_planes(path)
            planes.append(plane)
        except AttributeError as e:
            print(f"{e} occured at {path}")
    apply_to_all_recordings("H:\\", func)
    return ls, planes

def get_unique_trial_attrs_by_recording(return_df=False):
    trials, planes = get_every_trial()
    df = pd.DataFrame(trials)
    recordings = {}
    for recording in df.recording_id.unique():
        subset = df[df.recording_id==recording]
        subset = subset.loc[:,
                            ['recording_id', 'test', 'go', 'side', 
                             'correct', 'affirmative', 'contrast']]
        recordings[recording] = get_unique_attribute_values(subset)
    return (df,recordings,planes) if return_df else (recordings, planes)
        

def simple_summary():
    recordings, planes = get_unique_trial_attrs_by_recording()
    with open("C:/Users/viviani/Desktop/recording_descriptions.txt","w") as f:
        for plane, (recording,value) in zip(planes,recordings.items()):
            f.write(f"{recording}\n")
            f.write(f"    {plane} plane{'s' if plane>1 else ''}\n")
            for attr, array in value.items():
                if array.shape[0]>1:
                    f.write(f"    {attr:13}{list(array)}\n")
            f.write("\n\n")


        
    # df = pd.DataFrame(recordings)
    # #Drop all the timing information
    # df = df.loc[:,['recording_id', 'test', 'go', 'side', 'correct', 'affirmative', 'contrast']]
    
if __name__=="__main__":
    df, uniques,planes = get_unique_trial_attrs_by_recording(True)
    with open("C:/Users/viviani/Desktop/fullsummary.txt","w") as file:
        for recording, attr_dict in uniques.items():
            file.write(f"{recording}\n")
            subset = df[df.recording_id==recording]
            for attribute,unique_vals in attr_dict.items():
                if attribute not in ("recording_id","correct","affirmative"):
                    file.write(f"    {attribute}\n")
                    for value in unique_vals:
                        res = get_count_and_stats(subset,attribute,value)
                        file.write(f"        {value:5} {res}\n")
            file.write("\n\n")

    

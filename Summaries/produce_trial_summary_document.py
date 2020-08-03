# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 11:43:16 2020

@author: viviani

Create a summary dataframe and summary document detailing the composition
of each recording session in terms of the trial attributes - eg how often
mice were correct, their d-prime statistics, whether the side varied in the 
trial, whether contrast varied in the trial, and so on...

Alternatively, create a summary document detailing the different KINDS of
recording session and their POOLED statistics.

"""
import os
from collections import namedtuple
from contextlib import redirect_stdout

import pandas as pd

from accdatatools.Observations.trials import _get_trial_structs, SparseTrial
from accdatatools.Utils.map_across_dataset import apply_to_all_recordings
from accdatatools.Utils.convienience import item
from accdatatools.Utils.path import get_exp_id, get_timeline_path, get_psychstim_path
from accdatatools.Utils.map_across_dataset import no_of_planes
from accdatatools.Utils.deeploadmat import loadmat
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
    A SparseTrial Object that also knows in which recording it was collected,
    and the type of that recording
    '''
    def __init__(self,struct,trial_type,exp_path,tolerant=False):
        self.recording_id = get_exp_id(exp_path)
        self.type=trial_type
        super().__init__(struct,tolerant)
    def to_dict(self):
        result = super().to_dict()
        result["recording_id"] = self.recording_id
        result["task"] = self.type
        return result


def get_count_and_stats(dataframe,attribute='not_provided',value='not_provided', subsetting=True):
    if subsetting:
        if attribute=="not_provided" or value=="not_provided":
            raise ValueError("Subset requested but no attribute/value pair provided")
        else:
            subset = dataframe[dataframe[attribute]==value]
    else:
        subset = dataframe
    
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



def get_unique_attribute_values(df):
    result = {}
    for column in df.columns:
        unique_values = df[column].unique().astype("object")
        #nan values aren't equal to themselves, so we need to turn them into
        #something nice
        unique_values[pd.isna(unique_values)] = "Unspecified"
        result[column] = unique_values
    return result




def get_every_trial(root="D:\\"):
    ls = []
    planes = {}
    def func(path):
        path2 = get_psychstim_path(path)
        structs = _get_trial_structs(path2)
        psychstim = loadmat(path2)
        trial_type =  psychstim["expData"]["stim"]["stimType"]
        try:
            trial_objects = [ParentedSparseTrial(struct,trial_type,path,tolerant=True).to_dict() for struct in structs]
            ls.extend(trial_objects)
            plane = no_of_planes(path)
            planes[trial_objects[-1]["recording_id"]] = plane
        except AttributeError as e:
            print(f"{e} occured at {path}")
    apply_to_all_recordings(root, func)
    return ls, planes

def get_unique_trial_attrs_by_recording(root = "D:\\", return_df=False):
    trials, planes = get_every_trial(root = root)
    df = pd.DataFrame(trials)
    recordings = {}
    for recording in df.recording_id.unique():
        subset = df[df.recording_id==recording]
        subset = subset.loc[:,
                            ['recording_id', 'test', 'go', 'side', 
                             'correct', 'affirmative', 'contrast',
                             'task']]
        recordings[recording] = get_unique_attribute_values(subset)
    return (df,recordings,planes) if return_df else (recordings, planes)
        

def simple_summary():
    recordings, planes = get_unique_trial_attrs_by_recording()
    with open("C:/Users/viviani/Desktop/recording_descriptions.txt","w") as f:
        for (recording,value) in zip(planes,recordings.items()):
            f.write(f"{recording}\n")
            f.write(f"    {planes[recording]} plane{'s' if planes[recording]>1 else ''}\n")
            for attr, array in value.items():
                if array.shape[0]>1:
                    f.write(f"    {attr:13}{list(array)}\n")
            f.write("\n\n")


def get_classes_of_recording(root="D:\\"):
    df, recordings, planes = get_unique_trial_attrs_by_recording(root,True)
    # recordings is a dict with keys of recording_ids and values of
    # the unique values of trial attribute (side, contrast, etc) that
    # occured in that trial.
    # We want the opposite: a mapping from the KIND of recording (ie the unique
    # trial attributes) to a LIST of recording_IDs. To do this we invert the
    # dictionary...
    recording_classes = {} 
      
    for recording, varying_attributes in recordings.items():
        #Drop the recording_id, else each recording class would be 1 recording!
        varying_attributes.pop("recording_id",None)
        #The problem is, tuples and arrays can't be keys in python, so we
        #convert varying_attributes, which is a dictionary with typing
        #dict{attribute::str -> occuring_values::ndarray of str}, to nested tuples,
        #ie tuple(attribute::str, occurring_values::tuple of str)
        varying_attributes = tuple(
                                sorted(
                                    (k,
                                     tuple(sorted(v))
                                     ) for k,v in sorted(varying_attributes.items()))
                                    )
        if varying_attributes not in recording_classes: 
            recording_classes[varying_attributes] = [recording] 
        else: 
            recording_classes[varying_attributes].append(recording) 
    return recording_classes, df, planes


def classwise_summary(root = "D:\\"):
    recording_classes, df, planes = get_classes_of_recording(root=root)
    with open("classwise_summary.txt","w") as file:
        for idx,(attribute_values,ls_of_recordings) in enumerate(recording_classes.items()):
            file.write(f"CLASS {idx} ")
            subset = df[df.recording_id.isin(ls_of_recordings)]
            stats = get_count_and_stats(subset,subsetting=False)
            file.write(f"with pooled stats {stats}\n")
            file.write("Recordings in this class have trials that vary across the following parameters:\n")
            for attribute,unique_vals in attribute_values:
                if attribute not in ("recording_id","correct","affirmative"):
                    if len(unique_vals)>1:
                        file.write(f"    {attribute}\n")
                        for value in unique_vals:
                            stats = get_count_and_stats(subset,attribute,value)
                            file.write(f"        {value:5} {stats}\n")
                    elif len(unique_vals)==1:
                        file.write(f"    {attribute}={unique_vals[0]}\n")
            file.write(f"This class contained the following {len(ls_of_recordings)} recordings:\n")
            for idx,r in enumerate(ls_of_recordings):
                file.write(f"    {r} ({planes[r] if planes[r]!=0 else '?'} plane)")
                if idx%3==2:
                    file.write("\n")
            file.write("\n\n\n")
                    


def full_summary():
    df, uniques,planes = get_unique_trial_attrs_by_recording(True)
    with open("fullsummary.txt","w") as file:
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


        
    # df = pd.DataFrame(recordings)
    # #Drop all the timing information
    # df = df.loc[:,['recording_id', 'test', 'go', 'side', 'correct', 'affirmative', 'contrast']]
    
if __name__=="__main__":
    classwise_summary("H:\\")
    

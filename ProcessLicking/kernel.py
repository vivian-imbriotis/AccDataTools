# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 12:45:42 2020

@author: Vivian Imbriotis
"""

import pandas as pd
import numpy as np

def main(path):
    df = pd.read_csv(path)
    df["lick_transformed"] = lick_transform(df.lick_during_frame)
    df.to_csv(path)
    return df



def get_times_since_lick(lick):
    '''
    Get times since a lick has last occured in frames

    Parameters
    ----------
    lick : [bool or int]
        A list of bools of whether a lick occurs at the timepoints by which
        the list is indexed.

    Returns
    -------
    [int]
        Frames until a lick next occurs, prepended with 999.

    '''
    if type(lick)!=np.ndarray:
        lick = lick.to_numpy()
    start_idx = np.nonzero(lick>0)[0][0]
    source_array = lick[start_idx:]
    prepend = (999)*np.ones(start_idx)
    result = np.zeros(source_array.shape)
    for timepoint, lick in enumerate(source_array):
        if lick:
            result[timepoint] = 0
            counter = 1
        else:
            result[timepoint] = counter
            counter += 1
    return np.append(prepend, result)
        
        
def get_times_until_lick(lick):
    '''
    Get times until a lick occurs in frames.

    Parameters
    ----------
    lick : [bool or int]
        A list of bools of whether a lick occurs at the timepoints by which
        the list is indexed.

    Returns
    -------
    [int]
        Frames until a lick next occurs, appended with -999.

    '''
    if type(lick)!=np.ndarray:
        lick = lick.to_numpy()
    end_idx = np.nonzero(lick>0)[0][-1]
    source_array = lick[:end_idx+ 1]
    append = (999)*np.ones(len(lick) - end_idx - 1)
    result = np.zeros(source_array.shape)
    for timepoint, lick in reversed(list(enumerate(source_array))):
        if lick:
            result[timepoint] = 0
            counter = 1
        else:
            result[timepoint] = counter
            counter += 1
    return np.append(result,append)


def collapse_lick_timing(time_since,time_to, cuttoff=10):
    '''
    Intended behavior: Return the smaller of time_since or
    time_to at each time point, converting time_to to negative
    numbers. Then if any value of the resultant array is outside
    [-10,10], replace it with -999.

    Parameters
    ----------
    time_since : [int]
        Array of times since a lick.
    time_to : [int]
        array of times until a lick.

    Returns
    -------
    timing_info : [[int]].
    '''
    
    condition_list = [time_since<time_to, time_to<=time_since]
    timing_info = np.select(condition_list,
                            [-1*time_since,
                             time_to])
    timing_info = -1*timing_info
    timing_info[np.abs(timing_info)>cuttoff] = -999
    
    return timing_info

def lick_transform(lick, cuttoff=10):
    time_since_lick = get_times_since_lick(lick)
    time_until_lick  = get_times_until_lick(lick)
    timing_info = collapse_lick_timing(time_since_lick, 
                                       time_until_lick,
                                       cuttoff)
    return timing_info


if __name__=='__main__':
    df = main("C:/Users/Vivian Imbriotis/Desktop/first_1000_ROITrials.csv")
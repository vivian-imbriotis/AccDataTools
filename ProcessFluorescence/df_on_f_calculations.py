# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:06:01 2020

@author: viviani
"""
import numpy as np
from scipy.ndimage.filters import minimum_filter1d, uniform_filter1d



def subtract_neuropil_trace(F, Fneu, alpha = 0.8):
    subtracted_trace =  F - alpha*Fneu
    #add values until the lowest point has unit fluorescence
    return subtracted_trace - np.min(subtracted_trace) + 1

def get_smoothed_running_minimum(timeseries, tau1 = 30, tau2 = 100):
    mode = 'nearest'
    result = minimum_filter1d(uniform_filter1d(timeseries,tau1,mode=mode),
                            tau2,
                            mode = 'reflect')
    return result

def get_df_on_f0(F,F0=None):
    if type(F0)!=type(None):
        return (F - F0) / F0
    else:
        F0 = get_smoothed_running_minimum(F)
        return get_df_on_f0(F,F0)
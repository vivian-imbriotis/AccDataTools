# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 20:51:41 2020

@author: viviani
Simple script to determine what proportion 
"""

from accdatatools.Observations.recordings import Recording
from accdatatools.Utils.map_across_dataset import apply_to_all_recordings_of_class
import numpy as np

if __name__=="__main__":
    ls = []
    f = lambda pth : ls.append(Recording(pth).gen_iscell())
    apply_to_all_recordings_of_class("left_only_high_contrast", "H:", f)
    all_iscells = np.concatenate(ls)
    rate_of_inclusion = np.count_nonzero(all_iscells)/len(all_iscells)
    exl = 1-rate_of_inclusion
    print("On average, {exl*100}% of ROIs weere excluded from analysis.")
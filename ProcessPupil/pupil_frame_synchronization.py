# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 17:52:47 2020

@author: viviani
"""


from accdatatools.Utils.deeploadmat import loadmat
from accdatatools.Utils.signal_processing import rising_or_falling_edges

import numpy as np
import matplotlib.pyplot as plt



timeline_path = ("H:\\Local_Repository\\CFEB014\\2016-05-28_02_CFEB014\\"+
"2016-05-28_02_CFEB014_Timeline.mat")

h5_path = ("C:/Users/viviani/Desktop/micepupils-viviani-2020-07-09/videos/"+
           "2016-05-28_02_CFEB014_eyeDLC_resnet50_micepupilsJul9shuffle1_1030000.h5")



if __name__=="__main__":
    plt.plot(get_frame_times(timeline_path,h5_path))
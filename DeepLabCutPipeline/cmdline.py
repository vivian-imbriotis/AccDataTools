# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 17:19:36 2020

@author: viviani

This is a script designed to be run with:
    $python -i dlc_cmdline.py
    
to make it pleasant to perform deeplabcut operations from the
command line!
"""

import os
import numpy as np
import deeplabcut as dlc


if __name__=="__main__":
    root = "C:/Users/viviani/Desktop/micepupils-viviani-2020-07-09/"
    config_path = root + "config.yaml"
    vid_dir = root + "videos/"
    vids = [vid_dir + file for file in os.listdir(vid_dir) if ".mp4" in file]

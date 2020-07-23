# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:10:24 2020

@author: viviani
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


from accdatatools.Utils.path import exp_path
from accdatatools.Observations.recordings import Recording
from accdatatools.Timing.synchonisation import (get_neural_frame_times,
                                                 get_lick_state_by_frame)


experiment_ID = "2016-10-07_03_CFEB027"
experiment_path = exp_path(experiment_ID, "H:\\")

suite2p_path = os.path.join(experiment_path,
                            "suite2p",
                            "plane0")


timeline_path  = os.path.join(experiment_path,
                              "2016-10-07_03_CFEB027_Timeline.mat")


exp_recording = Recording(suite2p_path)


frame_times = get_neural_frame_times(
    timeline_path,exp_recording.ops["nframes"])

licking = get_lick_state_by_frame(timeline_path, frame_times)

corrs = [pearsonr(x,licking)[0] for x in exp_recording.dF_on_F]
corrs_isort = np.argsort(corrs)
to_plot = exp_recording.dF_on_F[corrs_isort]
fig,ax = plt.subplots()
lick_frame = np.nonzero(licking)
max_brightness = np.percentile(to_plot,99)
ax.imshow(np.clip(to_plot[-20:],-0.2,max_brightness), origin = 'lower',
          aspect = 5)
ax.vlines(lick_frame, -15,-10)
ax.set_xlim(0,1000)
fig.show()


# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 19:10:57 2020

@author: viviani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from accdatatools.DataCleaning.determine_dprime import d_prime
from accdatatools.Observations.trials import get_trials_in_recording
sns.set_style("darkgrid")

#vectorize a bunch of attribute lookups

isleft  = np.vectorize(lambda o : o.isleft)
isright = np.vectorize(lambda o : o.isright)
isgo    = np.vectorize(lambda o : o.isgo)
istest  = np.vectorize(lambda o : o.istest)
correct = np.vectorize(lambda o : o.correct)

def construct_sliding_dprime(trials):
    #Get a sliding bin of 50 trials:
    trials = np.array(trials, dtype=object)
    left_dprimes = np.zeros(trials.shape[0]-50)
    right_dprimes = np.zeros(trials.shape[0]-50)
    for idx in range(len(trials)-50):
        trial_bin = trials[idx:idx+50]
        trial_bin = trial_bin[~istest(trial_bin)]
        left      = trial_bin[isleft(trial_bin)]
        right     = trial_bin[isright(trial_bin)]
        for side in (left,right):
            gos   =  side[isgo(side)]
            nogos  =  side[~isgo(side)]
            hits  = gos[correct[gos]]
            fas  = nogos[~correct[nogos]]
            hit_rate = len(hits) / len(gos)
            fa_rate =  len(fas) / len(nogos)
            d = d_prime(hit_rate,fa_rate)
            if side is left:
                left_dprimes.append(d)
            else:
                right_dprimes.append(d)
    return (left_dprimes,right_dprimes)

        
        
if __name__=="__main__":
    from accdatatools.Utils.map_across_dataset import apply_to_all_recordings_of_class
    ls = []
    func = lambda pth : ls.append(get_trials_in_recording(pth,
                                                          use_sparse=True))
    apply_to_all_recordings_of_class("both_sides_high_contrast","H:\\",func)
    left_d, right_d = construct_sliding_dprime(ls[0])
    plt.plot(left_d, label="D' of contralateral trials")
    plt.plot(right_d, label = "D' of ipelateral trials")
    plt.legend()
    plt.show()
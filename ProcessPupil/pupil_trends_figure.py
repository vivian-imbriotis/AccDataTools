# -*- coding: utf-8 -*-

"""
Created on Fri Jul 31 11:36:46 2020

@author: viviani
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as c
import pandas as pd
import numpy as np
import seaborn
from accdatatools.ToCSV.without_collapsing import RecordingUnroller


seaborn.set_style("dark")

records = RecordingUnroller("H:/Local_Repository/CFEB033/2016-12-16_01_CFEB033",
                       ignore_dprime = True,
                       tolerate_lack_of_eye_video = False).to_unrolled_records()

df = pd.DataFrame(records)




fig,ax = plt.subplots(ncols = 2, gridspec_kw={'width_ratios': [15, 26]},
                      constrained_layout = True)
num_trials = len(df.Trial_ID.unique())
norm = c.Normalize(vmin=0,vmax = num_trials)
cmap = cm.ScalarMappable(norm,'plasma')
first_roi = df.ROI_ID[0]
i=0
x = np.arange(15,0,-1)
pupils_per_timepoint = np.array([df[df.peritrial_factor==i][df.ROI_ID==first_roi].pupil_diameter.values for i in x])
print(pupils_per_timepoint.shape)
pupils_per_timepoint[pupils_per_timepoint=="NA"] = np.nan
for idx,peritrial in enumerate(pupils_per_timepoint.transpose()):
    print(f"\r{i:5} of {num_trials:5}",end=""); i+=1
    peritrial = np.array(peritrial)
    ax[0].plot((x)/5,peritrial,
            color = cmap.to_rgba(idx,0.2))

ax[0].plot(x/5,np.nanmean(pupils_per_timepoint,axis=-1),color='black')
ax[0].set_xlim((3,0.2))
ax[0].set_ylim((8,25))
ax[0].set_xticks([3,2,1])
ax[0].set_ylabel("Pupil Diameter (pixels)")
ax[0].set_xlabel("Seconds preceding stimulus onset")
ax[0].set_title("Pretrial pupil behavior")

# x = np.arange(1,27)
# pupils_per_timepoint = np.array(df[df.trial_factor!=-999][df.ROI_ID==first_roi].pupil_diameter.values)
# pupils_per_timepoint = pupils_per_timepoint[:pupils_per_timepoint.shape[0] - pupils_per_timepoint.shape[0]%26]
# trials = pupils_per_timepoint.reshape(26,-1)
# print(trials.shape)
# trials[trials=="NA"] = np.nan
# for idx,trial in enumerate(trials.transpose()):
#     ax[1].plot((x)/5,trial,
#             color = cmap.to_rgba(idx,0.2))


i=0
x = np.arange(1,27)
pupils_per_timepoint = [df[df.trial_factor==i][df.ROI_ID==first_roi].pupil_diameter.values for i in x]
length = max(map(len, pupils_per_timepoint))
pupils_per_timepoint= [list(ls)+[np.nan]*(length-len(ls)) for ls in pupils_per_timepoint]
pupils_per_timepoint = np.array(pupils_per_timepoint)
pupils_per_timepoint[pupils_per_timepoint=="NA"] = np.nan
pupils_per_timepoint = pupils_per_timepoint.astype(float)
for idx,trial in enumerate(pupils_per_timepoint.transpose()):
    print(pupils_per_timepoint.shape)
    print(f"\r{i:5} of {num_trials:5}",end=""); i+=1
    trial = np.array(trial)
    ax[1].plot((x)/5,trial,
            color = cmap.to_rgba(idx,0.2))
ax[1].plot(x/5,np.nanmean(pupils_per_timepoint,axis=-1),color='black')

ax[1].set_xlim((1,5))
ax[1].set_ylim((8,25))
ax[1].set_ylabel("Pupil Diameter (pixels)")
ax[1].set_xlabel("Seconds since stimulus")
ax[1].set_title("Within-trial pupil behavior")
ax[1].legend()
cmap._A = []
cb = fig.colorbar(cmap,ax = ax, location="bottom", shrink = 0.4)
cb.set_label("Trial Number")
fig.show()

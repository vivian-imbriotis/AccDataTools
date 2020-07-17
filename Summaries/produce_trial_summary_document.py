# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:25:12 2020

@author: viviani
"""

import pandas as pd
import numpy as np
from collections import Counter


df = pd.read_csv("byroitrialdataset.csv")

#get just the trials by discarding every nonzero ROI number!
#df = df[df.roiNum==0]

#drop columns we're not interested in grouping by
variables = ["Contrast","Correct","Go"]

df_vars = df[variables]
# df.apply(Counter, axis='columns').value_counts()
info = df_vars.groupby(variables).size().to_frame('count').reset_index()
print(info)
    

# for contrast_level in (1,0.5,0.1,0):
#     subset = df[df.Contrast==contrast_level]
#     correct, incorrect = df.Correct.value_counts()
#     true_hit = subset[subset.Correct==1][subset.Go==1].shape[0]
#     false_hit = subset[subset.Correct==0][subset.Go==1].shape[0]
#     true_miss = subset[subset.Correct==1][subset.Go==0].shape[0]
#     false_miss = subset[subset.Correct==0][subset.Go==0].shape[0]
#     print(f"At {contrast_level:.1f}:")
#     print(f"Correctness Rate:           {100*correct/(correct+incorrect):.0f}%")
#     try:
#         print(f"Correctness on Go trials:   {100*true_hit/(true_hit+false_hit):.0f}%")
#     except:
#         pass
#     print(f"Correctness on noGo trials: {100*true_miss/(true_miss+false_miss):.0f}%")

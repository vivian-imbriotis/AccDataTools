# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 19:23:32 2020

@author: viviani
"""
import numpy as np

from accdatatools.Observations.recordings import Recording
from accdatatools.Utils.map_across_dataset import iterate_across_recordings

for exp in iterate_across_recordings(drive="H:\\"):
    print(f"\rProcessing {exp}...")

    try:
        a = Recording(exp)
        if np.any(a.F<0) and np.any(a.Fneu<0):
            for i in range(a.F.shape[0]):
                if np.any(a.F<0) and np.any(a.Fneu<0):
                    df = pd.DataFrame
                    df["F"] = a.F[i,:]
                    df["Fneu"] = a.Fneu[i,:]
                    df.to_csv(
                        "C:/users/viviani/desktop/single_recording_F_Fneu.csv"
                        )
        else:
            print(f"min F = {np.min(a.F).item():.1f}"+
                  f"min Fneu = {np.min(a.Fneu).item():.1F}")
    except Exception as e:
        pass
else:
    del a
    print("\nNot located in any experiment")
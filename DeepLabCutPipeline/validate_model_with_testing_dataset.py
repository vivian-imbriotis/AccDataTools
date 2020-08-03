# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 17:09:59 2020

@author: viviani
"""
import os

import pandas as pd
from deeplabcut.pose_estimation_tensorflow.evaluate import pairwisedistances



#Deeplabcut doesn't have inbuilt tools for making seperate testing
#datasets...so we're going to have to hack something together.
#The authors have provided some instruction for how to make this happen
#see here: https://forum.image.sc/t/how-to-move-partial-labeled-frame-to-a-new-project-how-to-quickly-evaluate-a-trained-dataset-on-analyzing-new-frames-without-retraining/29793
#and here: https://forum.image.sc/t/is-there-a-way-to-evaluate-a-deeplabcut-network-on-data-not-present-in-the-test-training-set/32222

def get_combined_data(training_data_file, machine_results_file):        
    Data = pd.read_hdf(training_data_file,
            "df_with_missing")
    long_names = Data.index.values
    short_names = [os.path.split(name)[1] for name in long_names]
    rename_dict = {long:short for long,short in zip(long_names,short_names)}
    Data = Data.rename(rename_dict)
    DataMachine = pd.read_hdf(machine_results_file, "df_with_missing")
    DataCombined = pd.concat([Data.T, DataMachine.T], axis=0).T
    return DataCombined

def get_error(training_data_file,  machine_results_file, certainty_cutoff=0.3):
    combined_data = get_combined_data(training_data_file, 
                                      machine_results_file,
                                      )
    return pairwisedistances(combined_data,
                             scorer1 = 'viviani',
                             scorer2 = "DLC_resnet50_micepupilsJul9shuffle1_1030000",
                             pcutoff = certainty_cutoff)[1]

if __name__=="__main__":
    os.chdir("C:/Users/viviani/Desktop/testing_dataset_factory-viviani-2020-07-27/labeled-data/2017-02-08_01_CFEB040_eye")
    error = get_error(
        "CollectedData_viviani.h5",
        "2017-02-08_01_CFEB040_eyeDLC_resnet50_micepupilsJul9shuffle1_1030000.h5"
        )
    print(f"Root mean squared error was {(error**2).mean().mean()**0.5}")
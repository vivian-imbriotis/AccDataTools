# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 10:48:23 2020

@author: viviani
"""

import deeplabcut as dlc

from accdatatools.DeepLabCutPipeline.cmdline_utils import config_path as machine_learning_config_path

#Deeplabcut doesn't have inbuilt tools for making seperate testing
#datasets...so we're going to have to hack something together.
#The authors have provided some instruction for how to make this happen
#see here: https://forum.image.sc/t/how-to-move-partial-labeled-frame-to-a-new-project-how-to-quickly-evaluate-a-trained-dataset-on-analyzing-new-frames-without-retraining/29793
#and here: https://forum.image.sc/t/is-there-a-way-to-evaluate-a-deeplabcut-network-on-data-not-present-in-the-test-training-set/32222



video_list = ["C:/Users/viviani/Desktop/2017-02-08_01_CFEB040_eye.mp4"]
testing_config = "C:\\Users\\viviani\\Desktop\\testing_dataset_factory-viviani-2020-07-27\\config.yaml"



if __name__=="__main__":
    # dlc.create_new_project("testing_dataset_factory",
    #                         "viviani",
    #                         videos = video_list,
    #                         working_directory = "C:/Users/viviani/Desktop",
    #                         copy_videos=True)
    # dlc.extract_frames(testing_config, mode='automatic', algo='kmeans', crop=False)
    dlc.label_frames(testing_config)
    dlc.analyze_time_lapse_frames(machine_learning_config_path,
    "C:/Users/viviani/Desktop/testing_dataset_factory-viviani-2020-07-27/labeled-data/2017-02-08_01_CFEB040_eye",
    save_as_csv=True)
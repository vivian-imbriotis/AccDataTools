# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 14:31:53 2020

@author: Vivian Imbriotis

Create a new DLC project.
"""


from .Utils.map_across_dataset import apply_to_all_one_plane_recordings
import deeplabcut as dlc
import os

def get_video(exp_path):
    for file in os.listdir(exp_path):
        if "eye.mp4" in file:
            return os.path.join(exp_path,file)
    raise ValueError(f"No eye.mp4 found in {exp_path}")

def get_ls_of_all_videos(drive):
    ls = []
    add2ls = lambda path:ls.append(get_video(path))
    apply_to_all_one_plane_recordings(drive, add2ls)
    return ls


if __name__=="__main__":
    project_name = "mousepupils"
    creator      = "viviani"
    vids         = get_ls_of_all_videos("H:\\")
    dlc.create_new_project(project_name,
                           creator,
                           vids,
                           copy_vids = True)
    root = "C:/Users/viviani/Desktop/micepupils-viviani-2020-07-09/"
    config_path = root + "config.yaml"
    vid_dir = root + "videos/"
    vids = [vid_dir + file for file in os.listdir(vid_dir) if ".mp4" in file]
    with open(config_path,"w") as file:
        with open("config.yaml","r") as source:
            for line in source:
                file.write(line)
    dlc.label_frames(config_path,vids)
    #Once frames are labelled, call dlc.create_training_dataset
    #and then dlc.train_network

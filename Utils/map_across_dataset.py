# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 15:00:38 2020

@author: Vivian Imbriotis
"""
import os
def no_of_planes(experiment_path):
    '''
    Returns the number of planes in a preprocessed suite2p folder.

    Parameters
    ----------
    experiment_path : string
        The path to a recording directory.

    Returns
    -------
    count : int
        Returns the number of directories named plane* contained in 
        experiment_path. If experiment_path does not exist, returns 0.

    '''
    #I'm sure there's an elegant way to do this with regex
    #or something
    count = 0
    try:
        for file in os.listdir(os.path.join(experiment_path,"suite2p")):
            if file[:-1] == "plane":
                count +=1
        if count!=1:
            print(f"{experiment_path} has non-singulate plane!")
        return count
    except NotADirectoryError:
        return 0

def apply_to_all_one_plane_recordings(drive,func):
    root = os.path.join(drive, 'Local_Repository')
    for animal in os.listdir(root):
        animal_path = os.path.join(root,animal)
        if os.path.isdir(animal_path):
            print(f"Processing {animal} experiments")
            for recording in os.listdir(animal_path):
                try:
                    rec_path = os.path.join(animal_path,recording)
                    if os.path.isdir(rec_path) and no_of_planes(rec_path) == 1:
                        func(rec_path)
                except (FileNotFoundError,ValueError) as e:
                    print(f"{e}: {recording} not to spec; passing over it...")

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
        return count
    except (NotADirectoryError, FileNotFoundError):
        return 0


def apply_to_all_one_plane_recordings(drive,func,verbose=False):
    root = os.path.join(drive, 'Local_Repository')
    for animal in os.listdir(root):
        animal_path = os.path.join(root,animal)
        if os.path.isdir(animal_path):
            if verbose: print(f"Processing {animal} experiments")
            for recording in os.listdir(animal_path):
                try:
                    rec_path = os.path.join(animal_path,recording)
                    if os.path.isdir(rec_path) and no_of_planes(rec_path) == 1:
                        func(rec_path)
                except (FileNotFoundError,ValueError) as e:
                    if verbose:
                        print(
                            f"LOG: {func.__name__} on {recording} resulted in {e}"
                            )


def apply_to_all_recordings(drive,func,verbose=False):
    root = os.path.join(drive, 'Local_Repository')
    for animal in os.listdir(root):
        animal_path = os.path.join(root,animal)
        if os.path.isdir(animal_path):
            if verbose: print(f"Processing {animal} experiments")
            for recording in os.listdir(animal_path):
                try:
                    rec_path = os.path.join(animal_path,recording)
                    func(rec_path)
                except (FileNotFoundError,ValueError) as e:
                    if verbose: print(f"LOG: {func.__name__} on {recording} resulted in {e}")

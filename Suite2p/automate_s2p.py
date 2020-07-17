# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:45:31 2020

@author: Vivian Imbriotis
"""

import numpy as np
from sys import argv
import os
from suite2p.run_s2p import run_s2p, default_ops




def run_s2p_on(path, ops_file=None):
    db = {'data_path':path}
    if ops_file==None:
        os.path.join(path,"suite2p","plane0","ops.npy")
    ops = np.load(ops_file,allow_pickle=True).item()
    ops["keep_movie_raw"] #This is so we get a KeyError
    ops["keep_movie_raw"] = True                
    ops["connected"]
    ops["connected"] = False
    ops["max_overlap"]
    ops["max_overlap"] = 0.2
    ops["do_registration"]
    ops["do_registration"] = 2 #ALWAYS redo registration
    ops["look_one_level_down"]
    ops["look_one_level_down"] = True
    run_s2p(ops=ops, db=db)


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

if __name__ == "__main__":
    if len(argv)<2:
        print("Usage: python3 automate_suite2p.py <drive>")
    drive = argv[1]
    apply_to_all_one_plane_recordings(drive, run_s2p_on)

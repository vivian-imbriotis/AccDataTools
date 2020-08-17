# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 15:00:38 2020

@author: Vivian Imbriotis
"""
import os
import pickle as pkl


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
    '''
    Map a function func across every experiment with 1 plane in a dataset in 
    drive.

    Parameters
    ----------
    drive : str
        The drive containing the dataset, eg "E://". The dataset root directory
        must be at top level of the drive
    func : callable
        The function to map. Is passed a single experiment path as an argument.
    verbose : bool, optional
        Whether to print information to console. The default is False.

    Returns
    -------
    None.

    '''
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
    '''
    Map a function func across every single experiment in a dataset in drive.

    Parameters
    ----------
    drive : str
        The drive containing the dataset, eg "E://". The dataset root directory
        must be at top level of the drive
    func : callable
        The function to map. Is passed a single experiment path as an argument.
    verbose : bool, optional
        Whether to print information to console. The default is False.

    Returns
    -------
    None.

    '''
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


def apply_to_all_recordings_of_class(cls, drive, func, verbose=True):
    '''
    Map a function across each experiment in Drive with features determined
    by cls.

    Parameters
    ----------
    cls : string
        The class of recording to which to apply func. One of
        {"both_sides_high_contrast", "left_only_high_contrast",
         "low_contrast"}.
    drive : str
        The drive containing the dataset, eg "E://". The dataset root directory
        must be at top level of the drive
    func : callable
        The function to map. Is passed a single experiment path as an argument.

    Returns
    -------
    None.

    '''
    file_path = f"../DataCleaning/{cls}.pkl"
    print(file_path)
    try:
        with open(file_path,'rb') as file:
            paths = pkl.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(
            """
            Hey, looks like that recording class doesn't exist on disk yet.
            Before calling this function, you need to run 
            get_recordings_for_analysis.py to generate the recording classes!
            Alternatively, you might have attempted to reference a recording
            class that doesn't exist, or might have mistyped the class name.""")
    for path in paths:
        try:
            func(path)
        except (FileNotFoundError,ValueError) as e:
            if verbose: print(f"LOG: {func.__name__} on {path} resulted in {e}")


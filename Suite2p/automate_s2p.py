# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:45:31 2020

@author: Vivian Imbriotis
"""

from statistics import mode, StatisticsError

import numpy as np
import os
from suite2p.run_s2p import run_s2p, default_ops
from accdatatools.Utils.map_across_dataset import apply_to_all_recordings
from accdatatools.Utils.path import exp_id
from accdatatools.Summaries.produce_trial_summary_document import get_classes_of_recording


def run_s2p_on(path, ops_file=None, reprocess=False, 
               infer_from_recording_classes=False,
               inferrer=None):
    db = {'data_path':[path]}
    if ops_file==None:
        ops_file = os.path.join(path,"suite2p","plane0","ops.npy")
    try:
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
    except FileNotFoundError:
        if all((any(["tif" in file for file in os.listdir(path)]),
                infer_from_recording_classes,
                inferrer!=None)):
            #How many planes?
            no_of_planes = inferrer(exp_id(path))
            if no_of_planes==1:
                ops = default_ops()
                ops["nchannels"] = 2
                ops["look_one_level_down"] = False
                ops["do_registration"] = 2
                ops["keep_movie_raw"] = True
                ops["align_by_chan"] = 2
                ops["nonrigid"] = False
                ops["connected"] = False
                ops["max_overlap"] = 0.2
                ops["bidi_corrected"] =  True
                ops["two_step_reigstration"] = True
                ops["sparse_mode"] = True
                try:
                    run_s2p(ops=ops,db=db)
                except Exception as e:
                    print(f"exception {e} raised at path {path} in response to run s2p call")
                    print(f"db file:")
                    for key,value in db.items():
                        print(f"{key}: {value}")
                    print(f"ops file:")
                    for key,value in ops.items():
                        print(f"{key}: {value}")
        else: 
            print(f"No TIFFS at {path}" if infer_from_recording_classes else f"{path} not yet processed.")
            return
    else:
        if not reprocess:
             print(f"{path} has already been processed by suite2p")
             return
        run_s2p(ops=ops,db=db)



def no_of_planes(experiment_path, infer_from_recording_class=False,
                 inferrer = None):
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
    except NotADirectoryError:
        if infer_from_recording_class and inferrer != None:
            raise NotImplementedError
            exp_id(experiment_path)
            count = inferrer(exp_id)
        else:
            count = 0
    

class PlaneNumberInferrer:
    def __init__(self):
        recording_classes, df, planes = get_classes_of_recording(root = "H:\\")
        what_class_is_this_recording = {}
        plane_number_idxed_by_rec_class = []
        for idx,(attribute_values,ls_of_recordings) in enumerate(recording_classes.items()):
            plane_values = []
            for r in ls_of_recordings:
                what_class_is_this_recording[r] = idx
                if planes[r]!=0:
                    plane_values.append(planes[r])
                else:
                    planes.pop(r)
            try:
                plane_number_idxed_by_rec_class.append(mode(plane_values))
            except StatisticsError:
                #guess 1
                plane_number_idxed_by_rec_class.append(0)
        self.explicit_values = planes
        self.recording_classifier = what_class_is_this_recording
        self.plane_number = plane_number_idxed_by_rec_class
            
    def __call__(self,recording):
        try:
            return self.explicit_values[recording]
        except KeyError:
            try:
                cls = self.recording_classifier[recording]
                plane_num = self.plane_number[cls]
                return plane_num
            except KeyError:
                return 0
            


if __name__ == "__main__":
    drive = "H:\\"
    inferrer = PlaneNumberInferrer()
    run_s2p_on_no_reprocessing = lambda path:run_s2p_on(path,reprocess=False,
                                                        infer_from_recording_classes=True,
                                                        inferrer = inferrer
                                                        )
    #Do this to every one-plane experiment:
    apply_to_all_recordings(drive, run_s2p_on_no_reprocessing)

    

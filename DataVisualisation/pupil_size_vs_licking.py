# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 11:36:41 2020

@author: Vivian Imbriotis
"""

from accdatatools.Timing.synchronisation import (get_lick_times,
                                                 get_eyecam_frame_times)
from accdatatools.ProcessPupil.size import get_pupil_size_at_each_eyecam_frame
from accdatatool.Observations.trials import (SparseTrial,
                                             _get_trial_structs)
from accdatatools.Utils.path import (get_timeline_path, 
                                     get_psychstim_path,
                                     get_pupil_hdf_path)

class PupilDiameterLickingFigure:
    def __init__(self,exp_path):
        timeline_path  = get_timeline_path(exp_path)
        psychstim_path = get_psychstim_path(exp_path)
        hdf_path       = get_pupil_hdf_path(exp_path)
        structs        = _get_trial_structs(psychstim_path)
        self.trials             = [SparseTrial(struct) for struct in structs]
        self.eyecam_frame_times = get_eyecam_frame_times(timeline_path)
        self.licking_times      = get_lick_times(timeline_path)
        self.pupil_diameters    = get_pupil_size_at_each_eyecam_frame(hdf_path)
    def show(self):
        raise NotImplementedError()
        
        
        
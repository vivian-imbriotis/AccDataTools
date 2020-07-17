# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:50:32 2020

@author: viviani
"""

import numpy as np


from accdatatools.ProcessPupil.size import get_pupil_siz_at_each_eyecam_frame
from accdatatools.Utils.deeploadmat import loadmat
from accdatatools.Utils.signal_processing import (rising_edges, 
                                                  rising_or_falling_edges)
from accdatatools.Observations import recordings



def get_neural_frame_times(timeline_path, number_of_frames):
    '''
    Helper function to get the mean time at which each frame in a recording
    was captured.

    Parameters
    ----------
    timeline_path : str
        path to Timeline.mat file in experiment directory.
    number_of_frames : int
        Total number of frames in the recording.

    Raises
    ------
    IOError
        The Timeline.mat object has a frame counter which describes the number
        of frames captured before the current tick. This is raised if the 
        counter is either not monotonically increasing, or if it increases
        by more than 1 at each clock tick.

    Returns
    -------
    frame_times : list of floats
        The times at which each frame occured. Because every 6 frames are
        averaged together, this is the mean of the times of those six frames.

    '''
    timeline = loadmat(timeline_path)
    timeline = timeline['Timeline']
    frame_counter = timeline['rawDAQData'][:,2]
    timestamps = timeline['rawDAQTimestamps']
    frame_times = np.zeros(int(frame_counter[-1]))
    current_frame = 0
    for timestamp,no_frames in zip(timestamps,frame_counter):
        if no_frames == (current_frame + 1):
            frame_times[current_frame] = timestamp
            current_frame+=1
        elif no_frames != current_frame:
            raise IOError('Frame counter in timeline.mat not monotonic')
    #Each suite2p frame is the average of 6 raw frames
    #extra ticks are extraneous
    frame_times = frame_times[:number_of_frames*6]
    frame_times = frame_times.reshape(number_of_frames, 6)
    frame_times = np.mean(frame_times, axis = -1)
    return frame_times

def get_lick_state_by_neural_frame(timeline_path, frame_times):
    '''
    For each frame, whether one or more licks occurred during the
    time over which that frame was captured.
    
    Parameters
    ----------
    timeline_path : string
        Path to Timeline.mat file
    frame_times : array of floats
        List of times of averaged frames; output of get_frame_times()

    Returns
    -------
    licks : array of ints; same shape as frame_times.shape
        Number of licks was initiated during the corresponding frame.

    '''
    timeline = loadmat(timeline_path)
    timeline = timeline['Timeline']
    lick_voltages = timeline['rawDAQData'][:,5]
    edges = rising_edges(lick_voltages)
    #When did these rising edges happen?
    lick_times = timeline['rawDAQTimestamps'][edges]
    #For each frame, how many licks occured in the time between that frame
    #and the next?
    #To find this we can just find the cumulative number of licks:
    cumulative_licks = np.zeros(frame_times.shape)
    #(surely there's a vectorised way to do this)
    for idx, frame_time in enumerate(frame_times):
        cumulative_licks[idx] = np.count_nonzero(lick_times<frame_time)
    #and then get the elementwise differences:
    return np.append(np.diff(cumulative_licks),0)

def get_pupil_at_each_timepoint(timeline_path, h5_path):
    timeline = loadmat(timeline_path)
    timeline = timeline['Timeline']
    eye_cam_voltages = timeline['rawDAQData'][:,3]
    edges = rising_or_falling_edges(eye_cam_voltages, cutoff=0.5)
    eye_cam_frame_times = timeline['rawDAQTimestamps'][edges]
    pupil_sizes = get_pupil_size_at_each_eyecam_frame(h5_path)
    print(eye_cam_frame_times.shape)
    print(pupil_sizes.shape)
    return np.stack(eye_cam_frame_times,pupil_sizes)

def get_pupil_size_by_neural_frame(timeline_path, h5_path, frame_times):
    pass
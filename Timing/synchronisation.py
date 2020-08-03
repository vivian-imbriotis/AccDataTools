# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:50:32 2020

@author: viviani
"""

import numpy as np


from accdatatools.ProcessPupil.size import get_pupil_size_at_each_eyecam_frame
from accdatatools.Utils.deeploadmat import loadmat
from accdatatools.Utils.signal_processing import (rising_edges, 
                                                  rising_or_falling_edges)
from accdatatools.ProcessPupil.size import get_pupil_size_at_each_eyecam_frame


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
        by more than 1 in a single millisecond clock tick.

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
    try:
        frame_times = frame_times.reshape(number_of_frames, 6)
    except ValueError:
            #in the case where number_of_frames*6 > len(frame_times)
            missing_frames = number_of_frames - frame_times.shape[0]//6
            frame_times = frame_times[:(number_of_frames-missing_frames)*6]
            frame_times = frame_times.reshape(number_of_frames-missing_frames, 6)
            frame_times = np.mean(frame_times, axis = -1)
            #In this case, we don't know when the last frame happens, but
            #we can just linearly extrapolate
            last_frames = np.empty(missing_frames)
            delta_T = np.mean(np.diff(frame_times))
            for idx, _ in enumerate(last_frames):
                last_frames[idx] = frame_times[-1] + (idx+1)*delta_T
            frame_times = np.append(frame_times,last_frames)
            return frame_times
    frame_times = np.mean(frame_times, axis = -1)
    return frame_times


def get_lick_times(timeline_path):
    timeline = loadmat(timeline_path)
    timeline = timeline['Timeline']
    lick_voltages = timeline['rawDAQData'][:,5]
    edges = rising_edges(lick_voltages)
    #When did these rising edges happen?
    lick_times = timeline['rawDAQTimestamps'][edges]
    return lick_times

def get_lick_state_by_frame(timeline_path, frame_times):
    '''
    For each frame, whether one or more licks occurred during the
    time over which that frame was captured.
    
    Parameters
    ----------
    timeline_path : string
        Path to Timeline.mat file
    frame_times : array of floats
        List of times of averaged frames; output of get_neural_frame_times()
        or of get_eyecam_frame_times()

    Returns
    -------
    licks : array of ints; same shape as frame_times.shape
        Number of licks was initiated during the corresponding frame.

    '''
    lick_times = get_lick_times(timeline_path)
    #For each frame, how many licks occured in the time between that frame
    #and the next?
    #To find this we can just find the cumulative number of licks:
    cumulative_licks = np.zeros(frame_times.shape)
    #(surely there's a vectorised way to do this)
    for idx, frame_time in enumerate(frame_times):
        cumulative_licks[idx] = np.count_nonzero(lick_times<frame_time)
    #and then get the elementwise differences:
    return np.append(np.diff(cumulative_licks),0)


def get_eye_diameter_at_timepoints(hdf_path,timeline_path,timepoints):
    pupil_sizes = get_pupil_size_at_each_eyecam_frame(hdf_path)
    eyecam_frame_times = get_eyecam_frame_times(timeline_path)
    eyecam_frame_times = eyecam_frame_times[:len(pupil_sizes)]
    #What was the nearest eyecam frame to each timepoints?
    nearest_eyecam_frames = get_nearest_frame_to_each_timepoint(
                                eyecam_frame_times,
                                timepoints)
    # try:
    #     assert len(pupil_sizes) == len(eyecam_frame_times)
    # except AssertionError as e:
    #     print(f'len(pupil_sizes) = {len(pupil_sizes)}')
    #     print(f'len(eyecam_frame_times) = {len(eyecam_frame_times)}')
    #     raise e
    return pupil_sizes[nearest_eyecam_frames]
    

def get_eyecam_frame_times(matlab_timeline_file):
    obj = loadmat(matlab_timeline_file)
    timeline = obj["Timeline"]
    columns = np.array([i.name for i in timeline["hw"]["inputs"]])
    eye_camera_strobe = timeline["rawDAQData"][:,np.where(columns=="eyeCameraStrobe")]
    eye_camera_strobe = eye_camera_strobe.reshape(-1) #Remove excess axes
    edges = rising_or_falling_edges(eye_camera_strobe, 0.5)
    eye_frames = intersperse_events(edges,n=20)
    timestamps = timeline['rawDAQTimestamps']
    eye_frame_times = timestamps[eye_frames]
    return eye_frame_times


def intersperse_events(input_array, n):
    '''
    Replaces each single True value in the input array with N evenly 
    distributed True values.
    
    >>> intersperse_events(np.array([1,0,0,0,1,0,1]),2).astype('int')
    Out: array([1, 0, 1, 0, 1, 1, 0])

    Parameters
    ----------
    input_array : Array of Bool or Array of Bool-like
    n : int

    Returns
    -------
    output_array : Array of Bool
    '''
    #get an array of true indexes
    events, = np.where(input_array)
    output_array = np.zeros(input_array.shape,dtype=bool)
    #in terms of indexes, where should all the new events go?
    #Well, just construct a linearly spaced vector between
    #each event, including the starting but not the stopping
    #event. This will get (input_array-1)*N events!
    for (this_event,next_event) in zip(events[:-1],events[1:]):
        interspersed_events = np.linspace(this_event,next_event,n,endpoint=False)
        interspersed_events = np.round(interspersed_events).astype('int')
        output_array[interspersed_events] = True
    return output_array


def get_nearest_frame_to_each_timepoint(frame_times, ls_of_timepoints):
    '''
    Get an array of the indexes of the frames captured closest to a list
    of times. 
    
    Example usage:
        You have an array of frame times of a camera viewing the mouses's head
        and an array of times the mouse licked an electrode.
        To get the frame captured closest to each lick, call
        >>>get_nearest_frame_to_each_timepoint(mouse_head_frame_times, 
                                               licking_times)

    Parameters
    ----------
    frame_times : array of float
        The times each frame occurred.
    ls_of_timepoints : array of float
        The event times of interest.

    Returns
    -------
    frameidx_list : array of int
        The frame indexes corresponding to ls_of_timepoints.

    '''
    #find all the smallest idxs such that all frame_times[idxs]>ls_of_timepoints
    idxs = np.searchsorted(frame_times, ls_of_timepoints, side="left")
    idxs[idxs==len(frame_times)] -= 1
    #Now there are only two options:
    #Either frame_times[idxs][n] is closest to ls_of_timepoints[n],
    #  or frame_times[idxs][n-1] is. Set up a condition list:
    suprema = frame_times[idxs]
    infima  = frame_times[idxs-1]
    condlist = [np.abs(suprema - ls_of_timepoints) < np.abs(infima - ls_of_timepoints),
                True]
    choicelist = [idxs, idxs-1]
    frame_idx_list = np.select(condlist, choicelist)

    return frame_idx_list


def get_pupil_size_by_neural_frame(timeline_path, h5_path, frame_times):
    pass
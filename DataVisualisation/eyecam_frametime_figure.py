# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 17:52:47 2020

@author: viviani
"""

from pupil_size_extraction import get_pupil_size_over_time
from deeploadmat import loadmat
import numpy as np
from trial_extraction import rising_edges, falling_edges, rising_or_falling_edges
import matplotlib.pyplot as plt

def as_time(n):
    return f"{int(n)//60}:{n%60:.2f}"

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

timeline_path = ("H:\\Local_Repository\\CFEB014\\2016-05-28_02_CFEB014\\"+
"2016-05-28_02_CFEB014_Timeline.mat")

h5_path = ("C:/Users/viviani/Desktop/micepupils-viviani-2020-07-09/videos/"+
           "2016-05-28_02_CFEB014_eyeDLC_resnet50_micepupilsJul9shuffle1_1030000.h5")

def timing_plot():
    obj = loadmat(timeline_path)
    timeline = obj["Timeline"]
    columns = np.array([i.name for i in timeline["hw"]["inputs"]])
    eye_camera_strobe = timeline["rawDAQData"][:,np.where(columns=="eyeCameraStrobe")]
    eye_camera_strobe = eye_camera_strobe.reshape(-1) #Remove excess axes
    edges = rising_or_falling_edges(eye_camera_strobe, 0.5)
    eye_frames = intersperse_events(edges,n=20)
    timestamps = timeline['rawDAQTimestamps']
    eye_frame_times = timestamps[eye_frames]

    plt.plot(timestamps[7500:15000:10], eye_camera_strobe[7500:15000:10], label = 'eyeCameraStrobe',
             linewidth = 3); 
    plt.plot(timestamps[7500:15000],eye_frames[7500:15000],label="Predicted eyecamera frames")
    plt.legend(); 
    plt.xlabel("Time (s) {from Timeline.rawDAQTimestamps}")
    plt.ylabel("Signal")
    plt.show()
if __name__=="__main__":
	timing_plot()


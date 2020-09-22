# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:45:31 2020

@author: viviani
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import cv2

from accdatatools.Timing.synchronisation import (get_lick_state_by_frame,
                                                get_eyecam_frame_times,
                                                get_nearest_frame_to_each_timepoint,
                                                get_neural_frame_times)
from accdatatools.ProcessPupil.size import get_pupil_size_at_each_eyecam_frame
from accdatatools.Observations.trials import get_trials_in_recording


def video(path, nframes = -1):
    '''
    A generator that runs through a video and yields each frame as a 
    (width,height,3) uint8 numpy array. Use it like this:
        >>>for frame in video('example.mp4'):
        >>>    do_something_to(frame)
    '''
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buffer = np.empty((frame_height, frame_width, 3), np.dtype('uint8'))
    i = 0
    previous_frame_capture_was_successful = True
    if nframes > 0:
        #If you only want the first n frames
        frame_count = nframes

    while (i < frame_count  and previous_frame_capture_was_successful):
        previous_frame_capture_was_successful,buffer = cap.read()
        i+=1
        yield buffer
    cap.release()
    cv2.waitKey(0)
    return

class EyeVideoSyncFigure:
    def __init__(self, eyecam_video_file, behav_video_file, matlab_timeline_file, dlc_h5_file, 
                 exp_path):
        self.eyecam_video_file = eyecam_video_file
        self.behav_video_file = behav_video_file
        self.fig,axes = plt.subplots(nrows = 2, ncols = 3)
        
        ((self.behavvid_axis,self.eyevid_axis,self.pupilsize_axis),
         (self.trial_axis,self.lick_axis,self.neuropil_axis)) = axes
        
        self.all_axes = axes.flatten()
        
        self.frame_times = get_eyecam_frame_times(matlab_timeline_file, dlc_h5_file)
        self.pupil_sizes = get_pupil_size_at_each_eyecam_frame(dlc_h5_file)
        self.frame_times = self.frame_times[:self.pupil_sizes.shape[0]]
        self.licks = get_lick_state_by_frame(matlab_timeline_file, self.frame_times)

        self.trials, self.recording = get_trials_in_recording(exp_path,
                                              ignore_dprime=True,
                                              return_se = True)
        
        self.neuropil_traces = self.get_neuropil_for_each_eyecam_frame(
            matlab_timeline_file)
        
        nframes = self.recording.ops['nframes']
        neural_frame_times = get_neural_frame_times(matlab_timeline_file,nframes)
        self.neural_frame_times = neural_frame_times
    
        self.trial_state = self.get_trial_state_for_each_eyecam_frame()
        
        # assert self.frame_times.shape == self.licks.shape
        # assert self.frame_times.shape == self.pupil_sizes.shape
        # assert self.frame_times.shape[0] == len(self.trial_state)
        # assert self.frame_times.shape[0] == self.neuropil_traces.shape[0]
        
    def get_trial_state_for_each_eyecam_frame(self):
        '''
        Array of Trial objects or None indexed by eyecam frame.
        Trial object if the frame occurs in the start_stimulus -> end response
        period, else None.

        Returns
        -------
        result : array of object
        '''
        result = []
        for time in self.times_of_closest_neural:
            for trial in self.trials:
                if trial.is_occuring(time):
                    result.append(trial)
                    break
            else:
                result.append(None)
        return result
    
    def get_neuropil_for_each_eyecam_frame(self, matlab_timeline_file):
        '''
        Get the neuropil brightness at the time each eyecam frame was captured.

        Parameters
        ----------
        matlab_timeline_file : TYPE
            DESCRIPTION.

        Returns
        -------
        neuropil_series : array of float, shape (timepoints,K,K)
            Array of neuropil traces, reshaped into KxK squares for 
            displaying with imshow.

        '''
        #Each value of this array is a neural frame's index,
        #each index of this list is an eyecam frame's index,
        #such that the series are closest to aligned!
        nframes = self.recording.ops['nframes']
        neural_frame_times = get_neural_frame_times(matlab_timeline_file,nframes)
        closest_neural_to_each_eyecam = get_nearest_frame_to_each_timepoint(
            neural_frame_times,
            self.frame_times)
        self.times_of_closest_neural = neural_frame_times[closest_neural_to_each_eyecam]
        #So now we can get the neuropil df/f at the time each eyecam frame
        #was captured
        neuropil_series = self.recording.Fneu[:,closest_neural_to_each_eyecam]
        #But I want to put this in an imshow format...so time for some hacky
        #garbage
        rois, timepoints  = neuropil_series.shape
        #find the largest lesser square number and its root
        root_rois_to_plot = np.floor(rois**0.5)
        root_rois_to_plot = root_rois_to_plot.astype(int)
        rois_to_plot = root_rois_to_plot**2
        #lop off extraneous neuropil regions
        neuropil_series = neuropil_series[:rois_to_plot,:]
        neuropil_series = neuropil_series.reshape(root_rois_to_plot,
                                                  root_rois_to_plot,
                                                  -1)
        neuropil_series = neuropil_series.transpose(2,0,1)
        return neuropil_series
        
        
        
            
    
    def render_frames(self):
        '''
        Renders a figure for each frame in the eyecam video file.
        Usage is as a generator, eg:
            >>>for figure in self.render_frames():
            >>>    do_stuff_with(figure)

        Yields
        ------
        matplotlib figure

        '''
        for idx,(eye_vid_frame, behav_frame) in enumerate(zip(video(self.eyecam_video_file),video(self.behav_video_file))):

            is_licking = self.licks[idx]
            pupil_size = self.pupil_sizes[idx]
            trial = self.trial_state[idx]
            neuropil = self.neuropil_traces[idx]
            
            if idx==0:
                eyeframe = self.eyevid_axis.imshow(eye_vid_frame)
                self.eyevid_axis.set_title("Eye Camera")
                
                xmin, xmax = self.lick_axis.get_xlim()
                ymin, ymax = self.lick_axis.get_ylim()
                self.lick_axis.set_title("Licking state")
                licktext = self.lick_axis.text(xmin+0.05,
                                    (ymax-ymin)/2, 
                                    "Lick" if is_licking else "",
                                    color = "black",
                                    fontsize = '24',
                                    va = 'center')
                
                self.pupilsize_axis.set_title("Pupil size (not to scale)")
                self.pupilsize_axis.set_xlim([-100,100])
                self.pupilsize_axis.set_ylim([-100,100])
                self.pupilsize_axis.set_aspect('equal', adjustable = 'box')
                if pupil_size != np.nan:
                    radius = (pupil_size/np.pi)**0.5
                    pupil_image = plt.Circle((0,0),radius,
                                             color = "black")
                    self.pupilsize_axis.add_artist(pupil_image)
                else:
                    pupil_image = plt.Circle((0,0),0,
                                             color = "black")
                    self.pupilsize_axis.add_artist(pupil_image)
                    
                
                
                self.trial_axis.set_title("Trial state")
                xmin, xmax = self.lick_axis.get_xlim()
                ymin, ymax = self.lick_axis.get_ylim()
                if trial != None:
                    trial_text = self.trial_axis.text(xmin+0.05,
                                         ymin + 0.1,
                                         ("Left\n" if trial.isleft else "Right\n") + 
                                          ("Go\n" if trial.isgo else "No-Go\n") + 
                                          "Trial",
                                          color = 'green' if trial.isgo else "red",
                                          fontsize = '24')
                else:
                    trial_text = self.trial_axis.text(xmin+0.05,
                                                      ymin + 0.1,
                                                      "",
                                                      color = "black",
                                                      fontsize = '24')
                
                self.behavvid_axis.set_title("Body Camera")
                bodyframe = self.behavvid_axis.imshow(behav_frame)
                
                self.neuropil_axis.set_title("Neuropil brightness")
                vmin = np.percentile(myfigure.neuropil_traces,1)
                vmax = np.percentile(myfigure.neuropil_traces,99)
                neuropil_image = self.neuropil_axis.imshow(neuropil,
                                                           vmin = vmin,
                                                           vmax = vmax)
                
                for axis in self.all_axes:
                    axis.set_xticks([])
                    axis.set_yticks([])
            else:
                eyeframe.set_data(eye_vid_frame)
                self.eyevid_axis.draw_artist(eyeframe)
                
                bodyframe.set_data(behav_frame)
                self.behavvid_axis.draw_artist(bodyframe)

                pupil_image.set_radius((pupil_size/np.pi)**0.5)
                self.pupilsize_axis.draw_artist(pupil_image)
                
                licktext.set_text("Lick! :D" if is_licking else "")
                licktext.set_color("green" if is_licking else "red")
                self.lick_axis.draw_artist(licktext)
                if trial != None:
                    trial_text.set_text(("Left\n" if trial.isleft else "Right\n") + 
                                              ("Go\n" if trial.isgo else "No-Go\n") + 
                                              "Trial")
                    trial_text.set_color('green' if trial.isgo else "red")
                else:
                    trial_text.set_text("")
                self.trial_axis.draw_artist(trial_text)
                
                neuropil_image.set_data(neuropil)
                self.neuropil_axis.draw_artist(neuropil_image)
                
                self.fig.canvas.update()
                self.fig.canvas.flush_events()
            yield self.fig
        plt.close('all')
        return
        
        
        
    def dump_all_frames_to_dir(self,dirpath):
        '''
        Create a figure for each frame in the eyecam video and then dump
        all of them into dirpath as png files.

        Parameters
        ----------
        dirpath : str
            path to target directory. If this doesn't exist it will be created.

        Returns
        -------
        None.

        '''
        try:
            os.mkdir(dirpath)
        except FileExistsError:
            pass
        for idx,frame in enumerate(self.render_frames()):
            target_path = os.path.join(dirpath,f"{idx}.png")
            frame.savefig(target_path)
    
    def test_render(self):
        '''Render a figure from the first frame in a window'''
        for figure in self.render_frames():
            figure.show()
            break

def video_from_dir_of_frames(dirpath):
        os.system(
          "C:\\Users\\viviani\\ffmpeg\\bin\\.\\ffmpeg.exe -r 20 -f image2 -i"+
          f" {dirpath}\\%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p"+
          f" {dirpath}.gif")
        
if __name__=="__main__":
    eyecam_video_file = ("C:/Users/viviani/Desktop/micepupils-viviani-2020-07-09/videos/"+
                         "2016-10-07_03_CFEB027_eye.mp4")
    behav_video_file  = "H:/Local_Repository/CFEB027/2016-10-07_03_CFEB027/2016-10-07_03_CFEB027_behav.mp4"
    timeline_file     = "H:/Local_Repository/CFEB027/2016-10-07_03_CFEB027/2016-10-07_03_CFEB027_Timeline.mat"
    exp_path          = "H:/Local_Repository/CFEB027/2016-10-07_03_CFEB027"
    h5_file           = ("C:/Users/viviani/Desktop/micepupils-viviani-2020-07-09/videos/"+
                "2016-10-07_03_CFEB027_eyeDLC_resnet50_micepupilsJul9shuffle1_1030000.h5")
    myfigure = EyeVideoSyncFigure(eyecam_video_file, behav_video_file, timeline_file, h5_file, exp_path)
    myfigure.dump_all_frames_to_dir("C:/Users/viviani/Desktop/framedump")
    video_from_dir_of_frames("C:/Users/viviani/Desktop/framedump")

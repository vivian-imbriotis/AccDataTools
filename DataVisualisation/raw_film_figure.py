# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 13:27:10 2020

@author: Vivian Imbriotis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
sns.set_style("dark")

_res = 256*256 #Assumed to always be the resolution of these tiff files

def get_frames(fd,n_frames,offset=0, verbose = False):
    '''
    Extracts frame data from a numpy binary file and dumps it as a tiff file.

    Parameters
    ----------
    fd : numpy.memmap instance
        A memory mapping of  suite2p binary file from which to read.
    n_frames : int
        Number of frames to include in the tiff file.
    offset : int, optional
        The frame from which to begin extraction. The default is 0.
    verbose : bool, optional
        Whether to print operations to stdout. The default is False.

    Returns
    -------
    None.

    '''
    if verbose:
        print("reading...",end='', flush=True)
    ims = fd[(offset*_res):((offset + n_frames)*_res)]
    length = ims.shape[0]
    frames = length//_res
    ims = ims.reshape((frames,256,256))
    return ims



class Raw2PhotonRecordingFigure:
    def __init__(self):
        path = r"D:\Local_Repository\CFEB013\2016-05-31_02_CFEB013\suite2p\plane0\data_raw.bin"
        fd = np.memmap(path,
                        dtype = np.int16,
                        mode = 'r')
        frames = get_frames(fd,n_frames=100)
        self.fig,ax = plt.subplots(tight_layout=True, figsize = [6,6])
        ax.set_xticks([]); ax.set_yticks([])
        im = ax.imshow(frames[0],cmap='gray')
        f = lambda i : im.set_data(frames[i])
        self.ani = FuncAnimation(self.fig, f, interval=1,frames=100)
    def show(self):
        self.fig.show()
    def save(self,savepath):
        self.ani.save(savepath, writer = PillowWriter(fps=30))

class RecordingWithROIsFigure:
    def __init__(self,show_rois=True):
        path = r"D:\Local_Repository\CFEB013\2016-05-31_02_CFEB013\suite2p\plane0\data.bin"
        fd = np.memmap(path,
                        dtype = np.int16,
                        mode = 'r')
        frames = get_frames(fd,n_frames=100)
        self.fig,ax = plt.subplots(tight_layout=True, figsize = [6,6])
        stat = np.load(r"D:\Local_Repository\CFEB013\2016-05-31_02_CFEB013\suite2p\plane0\stat.npy",
                       allow_pickle=True)
        within_rois = set()
        for i in stat[:60]:
            pixels = set(zip(i["xpix"],i["ypix"]))
            within_rois = within_rois.union(pixels)
        ROI_locations = np.zeros((256,256))
        for idx,_ in np.ndenumerate(ROI_locations):
            if tuple(reversed(idx)) in within_rois:
                ROI_locations[idx] = 1
        ROI_locations = np.stack((0.2*ROI_locations,0.8*ROI_locations,
                                  0.3*ROI_locations,0.3*ROI_locations),
                                 axis=-1)
        
        ax.set_xticks([]); ax.set_yticks([])
        im = ax.imshow(frames[0],cmap='gray')
        if show_rois:
            ax.imshow(ROI_locations)
        f = lambda i : im.set_data(frames[i])
        self.ani = FuncAnimation(self.fig, f, interval=1,frames=100)
    def show(self):
        self.fig.show()
    def save(self,savepath):
        self.ani.save(savepath, writer = PillowWriter(fps=30))
if __name__=="__main__":
    # rawfig = Raw2PhotonRecordingFigure()
    # rawfig.save(r"C:\Users\Vivian Imbriotis\Desktop\raw.gif")
    # fig = RecordingWithROIsFigure()
    # fig.save(r"C:\Users\Vivian Imbriotis\Desktop\processed.gif")
    stablefig = RecordingWithROIsFigure(show_rois=False)
    stablefig.save(r"C:\Users\Vivian Imbriotis\Desktop\stable.gif")
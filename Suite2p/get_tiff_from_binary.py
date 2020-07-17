# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:02:15 2020

@author: Vivian Imbriotis
"""

import skimage.external.tifffile as tif
import numpy as np
import os


_res = 256*256 #Assumed to always be the resolution of these tiff files

def extract_interleaved_frames(fds, save_path,n_frames,offset=0,verbose=False):
    '''
    MUST TEST OFFSET FUNCTIONALITY

    Parameters
    ----------
    fds : list
        list of numpy.memmap instances.
    save_path : TYPE
        DESCRIPTION.
    n_frames : TYPE
        DESCRIPTION.
    offset : TYPE, optional
        DESCRIPTION. The default is 0.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    '''
    stack = []
    if verbose:
        print("reading...",end='', flush=True)
    for frame in range(n_frames):
        source = fds[frame%len(fds)]
        start_loc = ((offset+frame)//len(fds))*_res
        stack.append(source[start_loc:start_loc+_res].reshape(1,256,256))
    res = np.stack(stack)
    if verbose:
        print("writing...", end='', flush=True)
    tif.imsave(save_path,
               res)
    if verbose:
        print('done.')

def extract_all_frames(read_paths,target_directory, verbose=False):
    '''
    Extract all frames from a set of suite2p binary files and dumps them as 1000-frame
    (maximum) tiff files in a target directory. The files are named 'data0.tif' 
    through to 'dataN.tif'. Frames from multiple read_paths are interleaved in
    the ordering of paths in read_paths. If files contain different numbers of
    frames, only the frames up to the minimum length of the binary files are
    processed and included.

    Parameters
    ----------
    read_path : list of str
        The paths to the suite2p binary files to read.
    target_directory : str
        The path to the directory to contain the tif files.
    verbose : bool, optional
        Whether to print operations to stdout. The default is False.

    Returns
    -------
    None.

    '''
    orig_wd = os.getcwd()
    try:
        os.chdir(target_directory)
    except FileNotFoundError:
        os.makedirs(target_directory)
        os.chdir(target_directory)
    frame_size = 2*_res
    frames_in_each_file = list(os.path.getsize(x)/frame_size for x in read_paths)
    frames_in_file = min(frames_in_each_file)
    del frame_size, frames_in_each_file
    frames_read = 0
    i = 0
    fds = []
    for idx,read_path in enumerate(read_paths):
        fds.append(np.memmap(read_path,
                             dtype = np.int16,
                             mode = 'r'))
    while frames_read < (frames_in_file - 1000):
        #Extract a 1000-frame tiff
        if verbose:
            print("Extracting a 1000 frame tiff...", end="", flush=True)
        extract_interleaved_frames(fds,
                         save_path = f"data{i}.tif",
                         n_frames = 1000,
                         offset = frames_read,
                         verbose = verbose)
        frames_read+=1000
        i+=1
    if frames_read!=frames_in_file:
        #Extract all remaining frames
        if verbose:
            print(
                f"Exctracting all remaining {frames_in_file-frames_read} frames...",
                end='',
                flush=True
                )
        extract_interleaved_frames(fds,
                         save_path = f"data{i}.tif",
                         n_frames = int(frames_in_file-frames_read),
                         offset = frames_read,
                         verbose = verbose)
    os.chdir(orig_wd)


def extract_n_frames_from_fd(fd, save_path,n_frames,offset=0, verbose = False):
    '''
    Extracts frame data from a numpy binary file and dumps it as a tiff file.

    Parameters
    ----------
    fd : numpy.memmap instance
        A memory mapping of  suite2p binary file from which to read.
    save_path : str
        Path to the tiff file.
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
    if verbose:
        print("writing...", end='', flush=True)
    tif.imsave(save_path,
           ims)
    if verbose:
        print('done.')



# if __name__=='__main__':
#     save_path = "C:\\Users\\uic\\Desktop\\tiff_files"
#     PATH = "D:\\Local_Repository\\CFEB026\\2016-09-21_05_CFeb026\\suite2p\\plane0\\data_raw.bin"
#     extract_all_frames(PATH,save_path, verbose=True)
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 19:53:44 2020

@author: Vivian Imbriotis
"""


import os
import seaborn

import numpy as np
import matplotlib.pyplot as plt

from accdatatools.Utils.path import get_timeline_path, get_exp_id
from accdatatools.Timing.synchronisation import get_lick_times
from accdatatools.ProcessFluorescence.data_cleaning import merge_rois

import accdatatools.ProcessFluorescence.df_on_f_calculations as df


class Recording:
    def __init__(self, path):
        '''
        

        Parameters
        ----------
        path : str
            The path to the experiment folder (which must already have been 
            processed by suite2p).

        '''
        self.exp_path = path
        suite2p_path = os.path.join(path,"suite2p","plane0")
        self.__path = suite2p_path
        cwd = os.getcwd()
        try:
            os.chdir(suite2p_path)
            
            #load in all the suite2p stuff
            self.stat = np.load("stat.npy", allow_pickle = True)
            self.ops  = np.load("ops.npy", allow_pickle = True).item()
            self._F    = np.load("F.npy")
            #drop ROIs that are everywhere zero (why do these exist?)
            valid_idxs = np.count_nonzero(self._F,axis=-1)!=0
            self.F = self._F[valid_idxs]
            # #Sometimes small values are negative which messes with division by F0
            # self.F    = np.abs(self.F)
            self._Fneu = np.load("Fneu.npy")
            self.Fneu = self._Fneu[valid_idxs]
            # #same here
            # self.Fneu = np.abs(self.Fneu)
    
            #Calculate deltaF/F
            self.Fcorr = df.subtract_neuropil_trace(self.F, self.Fneu)
            self.F0 = df.get_smoothed_running_minimum(self.Fcorr)
            self.dF_on_F = df.get_df_on_f0(self.Fcorr,self.F0)
            
            #Get the logged form
            self.logged_dF = df.log_transform(self.dF_on_F)
            
            #Get the deconvoluted spiking data and then binarise it
            self.spks_unbinarized = np.load('spks.npy')
            self.spks = (np.load('spks.npy') > 0)
            self.spks = self.spks[valid_idxs]
            self.iscell = np.load("iscell.npy")[:,0].astype(np.bool)
            self.iscell = self.iscell[valid_idxs]
            
            self.skew = np.fromiter((x["skew"] for x in self.stat),
                                    dtype = np.float32)
            
            #Merge together highly correlated ROIs
            self.dF_on_F, self.spks,self.iscell = merge_rois(self.dF_on_F,
                                                             self.spks,
                                                             self.iscell)
            
            timeline_path = get_timeline_path(self.exp_path)
            self.lick_times = get_lick_times(timeline_path)
            
            #Lastly we need to know in how many prior recordings this mouse
            #has appeared. To do this we need to go up a level to
            #the mouse's folder, then see which have dates that happened
            #before this one:
            mouse_dir,exp_id = os.path.split(self.exp_path)
            folders = sorted(os.listdir(mouse_dir))
            folders_preceding_this_one = folders[:folders.index(exp_id)]
            self.trial_number = len(folders_preceding_this_one) + 1
            
        finally:
            os.chdir(cwd)
    
    def gen_iscell(self):
        F_Fneu_ratio = np.fromiter((np.sum(x[0])/np.sum(x[1]) for x in zip(self._F,self._Fneu)),
                                        dtype = np.double)
        skew = np.fromiter((x["skew"] for x in self.stat),
                                dtype = np.float32)
        iscell_neuropil_criterion = F_Fneu_ratio > 1.05
        iscell_skew_criterion = skew > np.percentile(self.skew,10)
        
        #Both criteria must be met for ROI to be included in analysis
        iscell = np.logical_and(iscell_neuropil_criterion, 
                                iscell_skew_criterion)
        iscell = iscell.astype(np.int16)
        return np.stack((iscell,iscell)).transpose()
    
    def _overwrite_iscell(self):
        iscell = self.gen_iscell()
        np.save(
            os.path.join(self.__path,
                         "iscell.npy"),
            iscell)
    


    def plot_cell_pipeline(self,cell_id):
        seaborn.set()
        fig,ax = plt.subplots(nrows = 5, ncols = 1, tight_layout = True,
                              figsize = [6.4,8])
        ax[0].set_title("Raw F Trace")
        ax[0].plot(self.F[cell_id])
        ax[1].set_title("Raw Fneu Trace")
        ax[1].plot(self.Fneu[cell_id])
        ax[2].set_title("Fluor Trace corrected by Fneu")
        ax[2].plot(self.Fcorr[cell_id])
        ax[3].set_title("F0 with sliding window minimum")
        ax[3].plot(self.F0[cell_id])
        ax[4].set_title("dF/F0")
        ax[4].plot(self.dF_on_F[cell_id])
        return fig
    
    def plot_many_cells(self,condition = 'iscell'):
        seaborn.set()
        fig,ax = plt.subplots(nrows = 17, ncols = 1, constrained_layout=True,
                              figsize = [5,8.5])
        if condition in ("iscell","is cell","cell"):
            condition = self.iscell
            fig.suptitle("Random ROIs included under metric")
        elif condition in ("not cell", "notcell","ncell"):
            condition = np.logical_not(self.iscell)
            fig.suptitle("Random ROIs excluded under metric")
        series = (self.dF_on_F[condition])
        series = [series[x] for x in range(len(series))]
        for a in ax:
            idx = np.random.randint(len(series))
            val = series.pop(idx)
            a.plot(val)
            a.set_xticks([])
        return fig


if __name__=="__main__":
    from accdatatools.Utils.path import get_exp_path
    rec = Recording((r"H:\Local_Repository\CFEB026\2016-09-23_02_CFEB026"))

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 19:34:34 2020

@author: viviani
"""
import os
from math import ceil, sqrt
from itertools import zip_longest

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn

from accdatatools.Utils.path import get_mouse_path, get_exp_path
from accdatatools.Utils.map_across_dataset import no_of_planes


plt.ioff()
class AnimalMeanVideoFigure:
    def __init__(self, mouse_id = None, drive = "H://",
                  ls_of_exp_ids = None, ls_of_exp_paths=None):
        if sum((not mouse_id,not ls_of_exp_ids, not ls_of_exp_paths))!=2:
            raise ValueError("Supply exactly one of a mouse_id, a list of "
                             +"exp_ids or a list of exp_paths!")
        seaborn.set_style("dark")
        if mouse_id:
            experiment_paths = []
            animal_path = get_mouse_path(mouse_id, drive)
            
            for experiment in os.listdir(animal_path):
                try:
                    experiment_paths.append(
                        get_exp_path(experiment,drive)
                        )
                except FileNotFoundError: pass
        elif ls_of_exp_ids:
            experiment_paths = [get_exp_path(id,drive) for id in ls_of_exp_ids]
        elif ls_of_exp_paths:
            experiment_paths = ls_of_exp_paths
        mean_images = {}
        self.ls_of_exp_paths = []
        for experiment_path in experiment_paths:
            try:
                ops_path = os.path.join(experiment_path,
                                   "suite2p",
                                   "plane0",
                                   "ops.npy")
                ops = np.load(ops_path,allow_pickle=True).item()
                mean_image = ops["meanImgE"]
                mean_images[experiment_path] = mean_image
                self.ls_of_exp_paths.append(experiment_path)
            except FileNotFoundError:
                pass
        n_exps = len(mean_images)
        
        for width in range(ceil(sqrt(n_exps)),1,-1):
           if n_exps % width == 0: height = n_exps // width; break
        else:
            width = ceil(sqrt(n_exps))
            height= width
        
        self.fig = plt.figure(figsize=(2*width+1, 2*height+1)) 
        self.ax = np.empty((width,height),dtype=object)
        gs = gridspec.GridSpec(width, height,
                 wspace=0.0, hspace=0.0, 
                 top=1.-0.5/(height+1), bottom=0.5/(height+1), 
                 left=0.5/(width+1), right=1-0.5/(width+1)) 
        
        iterator = iter(mean_images.items())
        im_idx = 0
        for i in range(width):
            for j in range(height):
                try:
                    experiment, im = next(iterator)
                except StopIteration:
                    break
                
                self.ax[i][j]= plt.subplot(gs[i,j])
                self.ax[i][j].imshow(im)
                self.ax[i][j].text(0.5, 0.5,str(im_idx),
                                     horizontalalignment='center',
                                     verticalalignment='center',
                                     transform = self.ax[i][j].transAxes,
                                     size = 16,
                                     color = 'k')
                self.ax[i][j].set_xticklabels([])
                self.ax[i][j].set_yticklabels([])
                im_idx+=1
        self.canvas = self.fig.canvas
        self.show = self.fig.show
        self.draw = self.fig.draw


if __name__=="__main__":
    plt.ioff()
    testing_ls = [
                'D://Local_Repository\\CFEB037\\2017-01-27_01_CFEB037',
                'D://Local_Repository\\CFEB037\\2017-01-28_01_CFEB037',
                'D://Local_Repository\\CFEB037\\2017-01-30_01_CFEB037',
                'D://Local_Repository\\CFEB037\\2017-02-08_02_CFEB037']
    fig = AnimalMeanVideoFigure(ls_of_exp_paths = testing_ls)
    fig.show()
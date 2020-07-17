# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 19:03:37 2020

@author: Vivian Imbriotis
"""
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import numpy as np
from rastermap.mapping import Rastermap

class TrialKernelFigure:
    def __init__(self,csv):
        '''
        An encapsulation of a heatmap figure for all
        neurons' responses over the course of a trial.
        

        Parameters
        ----------
        exp_path : string
            The root path of an experiment.

        '''
        seaborn.set_style("dark")
        print(f"Loading data from {csv}")
        self.df = pd.read_csv(csv)
        self.licks = self.df.iloc[:,2:23].dropna().to_numpy()
        self.trials = self.df.iloc[:,23:49].dropna().to_numpy()
        self.r = Rastermap()

    def showtrials(self):
        self.r.fit(self.licks)
        low = np.percentile(self.trials ,5)
        high = np.percentile(self.trials, 99)
        plt.imshow(np.clip(self.licks,low,high)[self.r.isort])
        plt.show()
        
    def showlicks(self):
        self.r.fit(self.trials)
        low = np.percentile(self.licks ,5)
        high = np.percentile(self.licks, 99)
        plt.imshow(np.clip(self.trials,low,high)[self.r.isort])
        
    def showgo(self):
        fig, ax = plt.subplots(nrows = 3, ncols = 1)
        ax


if __name__=="__main__":
    fig = TrialKernelFigure("coefficientArray.csv")
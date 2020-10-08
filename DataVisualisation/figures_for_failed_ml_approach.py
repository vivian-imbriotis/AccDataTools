# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 18:36:36 2020

@author: Vivian Imbriotis
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage.filters import minimum_filter1d, uniform_filter1d
from scipy.stats import siegelslopes
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns

sns.set_style("dark")
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 11


class FailedMachineLearningFigureFactory():
    def __init__(self, suite2p_path, iscell_path):
        self.__path = suite2p_path
        cwd = os.getcwd()
        os.chdir(suite2p_path)

        self.stat = np.load("stat.npy", allow_pickle = True)
        self.ops  = np.load("ops.npy", allow_pickle = True).item()
        self.F    = np.load("F.npy")
        self.F    = np.abs(self.F)
        self.Fneu = np.load("Fneu.npy")
        self.Fneu = np.abs(self.Fneu)

        #Old way of calculating deltaF/F0
        self.Fcorr = np.maximum(self.F - 0.8*self.Fneu,1)
        self.F0 = self.running_min(self.Fcorr, 30, 100)
        self.dF_on_F = np.maximum((self.Fcorr - self.F0) / self.F0,0.01)
        

        self.iscell = np.load(iscell_path)[:,0].astype(np.bool)
        
        self.vcorr_map = self.ops['Vcorr']
        self.ROIs = self.stat_to_pixels()
        self.masks      = list(map(self.pixel_list_to_mask,self.ROIs))
        self.mean_corrs = np.fromiter(map(self.mask_to_mean_correlation,self.masks),
                                       dtype = np.double)
        self.F_Fneu_ratio = np.fromiter((np.sum(x[0])/np.sum(x[1]) for x in zip(self.F,self.Fneu)),
                                        dtype = np.double)
        self.stds =  np.fromiter((np.std(x) for x in self.dF_on_F),
                                  dtype = np.double)
        self.skew = np.fromiter((x["skew"] for x in self.stat),
                                dtype = np.float32)
        self.out_prime = np.stack((
            self.skew,
            self.stds,
            self.F_Fneu_ratio,
            self.mean_corrs,
            )).transpose()
        os.chdir(cwd)
        
    def stat_to_pixels(self):
        ROIs = []
        for roi in self.stat:
            ROIs.append(list(zip(roi['ypix'],roi['xpix'])))
        return ROIs
    
    def pixel_list_to_mask(self, pixel_list):
        mask = np.zeros(self.vcorr_map.shape, dtype = np.bool)
        for pixel in pixel_list:
            try:
                mask[pixel[0]][pixel[1]] = True
            except IndexError:
                pass
        return mask

    def mask_to_mean_correlation(self, mask):
        n_pixels = np.count_nonzero(mask)
        masked_corr_map = self.vcorr_map[mask] #This filters vcorr_map by mask
        if n_pixels==0:
            return 0
        return np.sum(masked_corr_map) / n_pixels
    
    @staticmethod
    def running_min(X,tau1,tau2):
        ###DEBUGGING IMPLEMENTATION###
        # return minimum_filter1d(X,tau2,mode = 'nearest')
        ###PREVIOUS IMPLEMENTATION###
        mode = 'nearest'
        result = minimum_filter1d(uniform_filter1d(X,tau1,mode=mode),
                                tau2,
                                mode = 'reflect')
        return result


        
    def produce_classifier(self, verbose=False, use_all = False):
        classifier = LogisticRegression(verbose=verbose,max_iter=1000)
        classifier.fit(self.out_prime,self.iscell) if not use_all else classifier.fit(self.all_features,self.iscell)
        return classifier
    



class PcaPlotFigure(FailedMachineLearningFigureFactory):
    def __init__(self):
        super().__init__()
        self.fig,ax = plt.subplots()
        pca = PCA(n_components = 2)
        res = pca.fit_transform(self.out_prime)
        is_cell = res[self.iscell]
        not_cell = res[np.logical_not(self.iscell)]
        ax.set_title("PCA of cell statistics for a single trial")
        ax.scatter(is_cell[:,0],is_cell[:,1],
                    color = 'green')
        ax.scatter(not_cell[:,0],not_cell[:,1],
                    color = 'red')
    def show(self):
        self.fig.show()

class ConfusionMatrixFigure(FailedMachineLearningFigureFactory):
    def __init__(self):
        super().__init__()
        self.classifier = self.produce_classifier()
    def show(self):
        plot_confusion_matrix(
                            self.classifier, 
                            self.out_prime,
                            self.iscell,
                            display_labels = [
                                "Not Bouton",
                                "Bouton"])


if __name__=='__main__':
    suite2p_path = ""
    iscell_path  = ""
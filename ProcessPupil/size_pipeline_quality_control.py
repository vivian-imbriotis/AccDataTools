# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:27:18 2020

@author: viviani
"""
import os
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from accdatatools.ProcessPupil.size import FittedEyeShape, FittedEllipse
from accdatatools.ProcessPupil.size import labelcsv_as_dataframe
from scipy.spatial.distance import directed_hausdorff

sns.set_style("darkgrid")

class ModifiedFittedEllipse(FittedEllipse):
    def __init__(self, n, pupil):
        self.min_points = n
        super().__init__(pupil)


def hausdorff(ellipse1, ellipse2):
    '''
    Calculate the Hausdorff Distance between the two ellipses.
    Parameters
    ----------
    ellipse1 : FittedEllipse Object
 
    ellipse2 : FittedEllipse Object

    Returns
    -------
    distance : int

    '''
    distance = directed_hausdorff(ellipse1.points, ellipse2.points)
    return distance



class SingleFrameFigure:
    def __init__(self,path):
        '''
        Display a figure of a training frame, and attempt
        to fit an eye and pupil to the training data.
    
        Parameters
        ----------
        path : str
            Path to a DeepLabCut labeled_data csv.
    
        Returns
        -------
        None.
    
        '''
        folder,_ = os.path.split(path)
        df = labelcsv_as_dataframe(path)
        row = df.sample()
        row = row.dropna(axis=1)
        row = row.to_numpy().reshape(-1)
        path = row[0]
        _,file = os.path.split(path)
        abs_path = os.path.join(folder,file)
        xy = row[1:].reshape(-1,2).astype(float)
        pupil = xy[:-4]
        eye = xy[-4:]
        self.fig,ax_rows = plt.subplots(ncols = 3, nrows = 4, figsize = (8,7),
                                   constrained_layout=True)
        for n_points_to_consider, ax in zip(range(8,4,-1),ax_rows):
            ellipses = []
            for pupil_mod in combinations(pupil,n_points_to_consider):
                try:
                    ellipses.append(ModifiedFittedEllipse(n_points_to_consider,pupil_mod))
                except np.linalg.LinAlgError:
                    ellipses.append(False)
            fittedeye = FittedEyeShape(eye)
            img = mpimg.imread(abs_path)
            ax[0].set_ylabel(f"N={n_points_to_consider}",
                             rotation='horizontal',
                             labelpad = 10)
            #Plot of labelled points
            ax[0].imshow(img)
            ax[0].plot(eye[:,0], eye[:,1], 'o', color = "red")
            ax[0].plot(pupil[:,0], pupil[:,1], 'o', color = "blue")
            #Plot of ellipse on image
            ax[1].imshow(img)
            for ellipse in ellipses:
                if ellipse:
                    ellipse.plot(ax[1],color='blue')
            fittedeye.plot(ax[1])
            #plot of ellipse alone
            fittedeye.plot(ax[2])
            for ellipse in ellipses:
                if ellipse:
                    ellipse.plot(ax[2])
            ax[2].set_xlim(ax[1].get_xlim())
            ax[2].set_ylim(ax[1].get_ylim())
            for axis in ax:
                axis.set_xticks([])
                axis.set_yticks([])
        ax_rows[0][0].set_title("DeepLabCut-labelled Points")
        ax_rows[0][1].set_title("Pupil extraction using N points")
        ax_rows[0][2].set_title("Algorithm's prediction")
    def show(self):
        self.fig.show()
    

# def least_squares_trend_for_one_frame(row, plotting = True):
#     if row.isnull().values.any():
#         raise ValueError("Row must have all 8 eye points placed!")
#     row = row.to_numpy().reshape(-1)
#     path = row[0]
#     _,file = os.path.split(path)
#     xy = row[1:].reshape(-1,2).astype(float)
#     pupil = xy[:-4]
#     eye = xy[-4:]
#     error_arrays = []
#     for n_points_to_consider in range(8,4,-1):
#         ellipses = []
#         for pupil_mod in combinations(pupil,n_points_to_consider):
#             try:
#                 ellipses.append(
#                     ModifiedFittedEllipse(n_points_to_consider,pupil_mod)
#                     )
#             except:
#                 pass
#         errors = np.array([(n_points_to_consider,x.get_mean_squared_error(pupil)) for x in ellipses])
#         error_arrays.append(errors)
#     errors = np.concatenate(error_arrays)
#     if plotting:
#         fig,ax = plt.subplots()
#         violin = ax.violinplot([i[:,1] for i in error_arrays],[8,7,6,5],showmeans=True)
#         violin["bodies"][0].set_label("Probability Density")
#         violin["cmeans"].set_color("black")
#         violin["cmeans"].set_label("Mean")
#         violin["cmins"].set_label("Minima and maxima")
#         ax.set_xticks([8,7,6,5])
#         ax.set_xlim((8.5,4.5));
#         ax.set_xlabel("Points considered in fitting ellipse")
#         ax.set_ylabel("Root Mean Squared Error from full set of 8 points (pixels)")
#         ax.legend(loc = 'upper left')
#         ax.set_ylim((0,ax.get_ylim()[1]))
#         fig.show()
#     return errors

# def least_squares_figure(path):
#     folder,_ = os.path.split(path)
#     df = labelcsv_as_dataframe(path)
#     errors = []
#     for idx,row in df.iterrows():
#         try:
#             errors.append(least_squares_trend_for_one_frame(row, False))
#         except: pass
#     errors = np.concatenate(errors) #get as a flat 2d np array
#     error_arrays = [errors[errors[:,0]==n][:,1] for n in range(8,4,-1)]
#     fig,ax = plt.subplots()
#     violin = ax.violinplot(error_arrays,[8,7,6,5],showmeans=True)
#     violin["bodies"][0].set_label("Probability Density")
#     violin["cmeans"].set_color("black")
#     violin["cmeans"].set_label("Mean")
#     violin["cmins"].set_label("Minima and maxima")
#     #ax.plot(distances[:,0],distances[:,1],'o', markersize=4, label = 'Ellipse from random permutation of points');
#     # ax.plot([8,7,6,5],[np.mean(i) for i in distance_arrays],
#     #          label='Mean',
#     #          color = 'black')
#     ax.set_xticks([8,7,6,5])
#     ax.set_xlim((8.5,4.5));
#     ax.set_xlabel("Points considered in fitting ellipse")
#     ax.set_ylabel("Root Mean Square Error from all 8 points (pixels)")
#     ax.legend(loc='upper left')
#     fig.show()
#     return errors

# def hausdorff_trend_for_one_frame(row, plotting = True):
#     if row.isnull().values.any():
#         raise ValueError("Row must have all 8 eye points placed!")
#     row = row.to_numpy().reshape(-1)
#     path = row[0]
#     _,file = os.path.split(path)
#     xy = row[1:].reshape(-1,2).astype(float)
#     pupil = xy[:-4]
#     eye = xy[-4:]
#     master_ellipse = FittedEllipse(pupil)
#     distance_arrays = []
#     for n_points_to_consider in range(8,4,-1):
#         ellipses = []
#         for pupil_mod in combinations(pupil,n_points_to_consider):
#             try:
#                 ellipses.append(
#                     ModifiedFittedEllipse(n_points_to_consider,pupil_mod)
#                     )
#             except:
#                 pass
#         distances = np.array([(n_points_to_consider,hausdorff(master_ellipse, x)[0]) for x in ellipses])
#         distance_arrays.append(distances)
#     distances = np.concatenate(distance_arrays)
#     if plotting:
#         plt.violinplot([i[:,1] for i in distance_arrays],[8,7,6,5],showmeans=True)
#         plt.plot(distances[:,0],distances[:,1],'o', markersize=4, label = 'Single ellipse from random combination of points');
#         plt.plot([8,7,6,5],[np.mean(i[:,1]) for i in distance_arrays],
#                  label='Mean',
#                  color = 'black')
#         plt.xticks([8,7,6,5])
#         plt.xlim((8.5,4.5));
#         plt.xlabel("Points considered in fitting ellipse")
#         plt.ylabel("Hausdorff Distance to 8-point fitted ellipse")
#         plt.legend()
#         plt.show()
#     return distances

# def hausdorff_trend_figure(path):
#     folder,_ = os.path.split(path)
#     df = labelcsv_as_dataframe(path)
#     distances = []
#     for idx,row in df.iterrows():
#         try:
#             distances.append(hausdorff_trend_for_one_frame(row, False))
#         except: pass
#     distances = np.concatenate(distances) #get as a flat 2d np array
#     distance_arrays = [distances[distances[:,0]==n][:,1] for n in range(8,4,-1)]
#     fig,ax = plt.subplots()
#     violin = ax.violinplot(distance_arrays,[8,7,6,5],showmeans=True)
#     violin["bodies"][0].set_label("Probability Density")
#     violin["cmeans"].set_color("black")
#     violin["cmeans"].set_label("Mean")
#     violin["cmins"].set_label("Minima and maxima")
#     #ax.plot(distances[:,0],distances[:,1],'o', markersize=4, label = 'Ellipse from random permutation of points');
#     # ax.plot([8,7,6,5],[np.mean(i) for i in distance_arrays],
#     #          label='Mean',
#     #          color = 'black')
#     ax.set_xticks([8,7,6,5])
#     ax.set_xlim((8.5,4.5));
#     ax.set_xlabel("Points considered in fitting ellipse")
#     ax.set_ylabel("Hausdorff Distance to 8-point fitted ellipse")
#     ax.legend(loc='upper left')
#     fig.show()
#     return distances
    
class LeastSquaresAndHausdorffEllipseFigure:
    def __init__(self,path):
        self.fig, ax = plt.subplots(figsize=(8,5),ncols = 2)
        self.draw_lst_squs_plot(ax[0],path)
        self.draw_hausdorff_plot(ax[1],path)
    def draw_lst_squs_plot(self,ax,path):
        folder,_ = os.path.split(path)
        df = labelcsv_as_dataframe(path)
        errors = []
        for idx,row in df.iterrows():
            try:
                errors.append(least_squares_trend_for_one_frame(row, False))
            except: pass
        errors = np.concatenate(errors) #get as a flat 2d np array
        error_arrays = [errors[errors[:,0]==n][:,1] for n in range(8,4,-1)]
        violin = ax.violinplot(error_arrays,[8,7,6,5],showmeans=True)
        violin["bodies"][0].set_label("Probability Density")
        violin["cmeans"].set_color("black")
        violin["cmeans"].set_label("Mean")
        violin["cmins"].set_label("Minima and maxima")
        ax.set_xticks([8,7,6,5])
        ax.set_xlim((8.5,4.5));
        ax.set_xlabel("Points considered in fitting ellipse")
        ax.set_ylabel("Root Mean Square Error from all 8 points (pixels)")
        ax.legend(loc='upper left')
    def draw_hausdorff_plot(self,ax,path):
        folder,_ = os.path.split(path)
        df = labelcsv_as_dataframe(path)
        distances = []
        for idx,row in df.iterrows():
            try:
                distances.append(hausdorff_trend_for_one_frame(row, False))
            except: pass
        distances = np.concatenate(distances) #get as a flat 2d np array
        distance_arrays = [distances[distances[:,0]==n][:,1] for n in range(8,4,-1)]
        violin = ax.violinplot(distance_arrays,[8,7,6,5],showmeans=True)
        violin["bodies"][0].set_label("Probability Density")
        violin["cmeans"].set_color("black")
        violin["cmeans"].set_label("Mean")
        violin["cmins"].set_label("Minima and maxima")
        ax.set_xticks([8,7,6,5])
        ax.set_xlim((8.5,4.5));
        ax.set_xlabel("Points considered in fitting ellipse")
        ax.set_ylabel("Hausdorff Distance to 8-point fitted ellipse")
        ax.legend(loc='upper left')
        
    @staticmethod
    def least_squares_trend_for_one_frame(row, plotting = True):
        if row.isnull().values.any():
            raise ValueError("Row must have all 8 eye points placed!")
        row = row.to_numpy().reshape(-1)
        path = row[0]
        _,file = os.path.split(path)
        xy = row[1:].reshape(-1,2).astype(float)
        pupil = xy[:-4]
        eye = xy[-4:]
        error_arrays = []
        for n_points_to_consider in range(8,4,-1):
            ellipses = []
            for pupil_mod in combinations(pupil,n_points_to_consider):
                try:
                    ellipses.append(
                        ModifiedFittedEllipse(n_points_to_consider,pupil_mod)
                        )
                except:
                    pass
            errors = np.array([(n_points_to_consider,x.get_mean_squared_error(pupil)) for x in ellipses])
            error_arrays.append(errors)
        errors = np.concatenate(error_arrays)
        if plotting:
            fig,ax = plt.subplots()
            violin = ax.violinplot([i[:,1] for i in error_arrays],[8,7,6,5],showmeans=True)
            violin["bodies"][0].set_label("Probability Density")
            violin["cmeans"].set_color("black")
            violin["cmeans"].set_label("Mean")
            violin["cmins"].set_label("Minima and maxima")
            ax.set_xticks([8,7,6,5])
            ax.set_xlim((8.5,4.5));
            ax.set_xlabel("Points considered in fitting ellipse")
            ax.set_ylabel("Root Mean Squared Error from full set of 8 points (pixels)")
            ax.legend(loc = 'upper left')
            ax.set_ylim((0,ax.get_ylim()[1]))
            fig.show()
        return errors
    @staticmethod
    def hausdorff_trend_for_one_frame(row, plotting = True):
        if row.isnull().values.any():
            raise ValueError("Row must have all 8 eye points placed!")
        row = row.to_numpy().reshape(-1)
        path = row[0]
        _,file = os.path.split(path)
        xy = row[1:].reshape(-1,2).astype(float)
        pupil = xy[:-4]
        eye = xy[-4:]
        master_ellipse = FittedEllipse(pupil)
        distance_arrays = []
        for n_points_to_consider in range(8,4,-1):
            ellipses = []
            for pupil_mod in combinations(pupil,n_points_to_consider):
                try:
                    ellipses.append(
                        ModifiedFittedEllipse(n_points_to_consider,pupil_mod)
                        )
                except:
                    pass
            distances = np.array([(n_points_to_consider,hausdorff(master_ellipse, x)[0]) for x in ellipses])
            distance_arrays.append(distances)
        distances = np.concatenate(distance_arrays)
        if plotting:
            plt.violinplot([i[:,1] for i in distance_arrays],[8,7,6,5],showmeans=True)
            plt.plot(distances[:,0],distances[:,1],'o', markersize=4, label = 'Single ellipse from random combination of points');
            plt.plot([8,7,6,5],[np.mean(i[:,1]) for i in distance_arrays],
                     label='Mean',
                     color = 'black')
            plt.xticks([8,7,6,5])
            plt.xlim((8.5,4.5));
            plt.xlabel("Points considered in fitting ellipse")
            plt.ylabel("Hausdorff Distance to 8-point fitted ellipse")
            plt.legend()
            plt.show()
        return distances
    def show(self):
        self.fig.show()

 
if __name__=="__main__":
    im = hausdorff_trend_figure(
        "C:/Users/viviani/Desktop/micepupils-viviani-2020-07-09/labeled-data/2017-03-30_01_CFEB045_eye/CollectedData_viviani.csv",
        )
    
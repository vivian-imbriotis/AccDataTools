# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 14:26:31 2020

@author: viviani
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.image as mpimg
import pandas as pd
import os

plt.ioff()

class FittedEyeShape:
    @staticmethod
    def calc_parabola_vertex(xy1, xy2, xy3):
        '''
        Find the coeffieients of a parabola that passes through three points.
		Adapted and modifed to get the unknowns for defining a parabola:
		http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
		'''
        (x1,y1),(x2,y2),(x3,y3) = xy1,xy2,xy3
        denom = (x1-x2) * (x1-x3) * (x2-x3);
        A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom;
        B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom;
        C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom;
        return A,B,C
    
    def __init__(self,xys,allow_invalid=False):
        '''
        An eye outline, based on parabolic curves.
        
        Parameters
        ----------
        xys : An arraylike of 4 points describing the lateral, superior,
            medial, and inferiormost points of an eye in pixel coordinates.
            Expects points to be passed in that order; lateral, superior, 
            medial, inferior.
        allow_invalid: bool
          If True, suppress errors if the eyeshape is found to be invalid.
          Instead, just set the self.valid attribue to False. Useful if
          you want to plot the invalid eye shape!
        
        Attributes
        -------
        lateral:  lateral  (x,y) point
        superior: superior (x,y) point
        medial:   medial   (x,y) point
        inferior: inferior (x,y) point
        upper_parabola: function mapping x -> y for the upper half of
            the eye shape, in pixel coordinates
        lower_parabola: function mapping x -> y for the lower half of
            the eye shape, in pixel coordinates
        valid:  True unless allow_invalid and the eye shape is poor (ie would
            otherwise raise an exception)

        '''

        xys = np.array(xys)
        xys = xys.reshape(-1,2)
        self.lateral,self.superior,self.medial,self.inferior = xys

        A,B,C = self.calc_parabola_vertex(self.lateral,self.superior,self.medial)
        a,b,c = self.calc_parabola_vertex(self.lateral,self.inferior,self.medial)
        self.upper_parabola = lambda x:A*x**2+B*x+C
        self.lower_parabola = lambda x:a*x**2+b*x+c
        
        if allow_invalid:
            try:
                self.throw_errors_if_absurd()
                self.valid=True
            except Exception as e:
                self.valid=False
        else:
            self.throw_errors_if_absurd()
            self.valid = True    
    def throw_errors_if_absurd(self):
        '''
        Basic quality control considerations

        Raises
        -------
        ValueError: raised if points violate basic assumptions about eye shape

        '''
        #The medial edge of the eye should be left of the lateral edge
        if self.medial[0] > self.lateral[0]:
            raise ValueError('Unacceptable Eye Shape: medial_x > lateral_x')
        #The inferior edge of the eye may be above the superior edge in a 
            #blink state, so we'll add a 150 pixel buffer
        if self.superior[1] < self.inferior[1] - 150:
            raise ValueError('Unacceptable Eye Shape: inferior_y > superior_y')
        #The eye also shouldn't bee too large
        if self.lateral[0] - self.medial[0] > 300:
            raise ValueError('Unacceptable Eye Shape: mediolateral aspect > 300')
        if self.superior[1] - self.inferior[1] > 300:
            raise ValueError('Unacceptable Eye Shape: superioinferior aspect>300')
        
        #Finally, the superior and inferior points should be between the medial
            #and lateral ones.
        for point in (self.lateral,self.medial):
            if point[0]<self.medial[0] or point[0]>self.lateral[0]:
                raise ValueError("Unacceptable Eye Shape.")
        #Thinking now about the parabolas we generate...if they go outside
                #the frame, something has probably gone wrong...
        x = np.linspace(self.medial[0],self.lateral[0],50)
        upper = self.upper_parabola(x)
        lower = self.lower_parabola(x)
        for parabola in (upper,lower):
            if np.any(parabola < 0) or np.any(parabola > 480):
                raise ValueError("Unacceptable Eye shape: shape not in frame")
    def contains(self,points):
        '''
        For each point in points, checks if that point is contained in 
        the eye sihlouette.
        
        Parameters
        ----------
        points : Arraylike of float
            Passed as either a flattened array, ie [x0,y0,x1,x2]
            or as a Nx2 array, ie [[x1,y1],[x2,y2]]
            
        Returns
        -------
        contains : Array of Bools
            If points is flat, this is the same shape as points.
            If points is an Nx2 array, this is a flat N element array
            (ie one bool for each point in points)
            eg:
            >>eye.contains([1,2,3,4])     -->[True,True,False,False]
            >>eye.contains([[1,2],[3,4]]) -->[True,False]

        '''
        points = np.array(points)
        original_shape = points.shape[0]
        points = points.reshape(-1,2)
        contains = np.logical_and.reduce((
               (points[:,0]>self.medial[0]),
               (points[:,0]<self.lateral[0]),
               (self.upper_parabola(points[:,0])<points[:,1]),
               (self.lower_parabola(points[:,0])>points[:,1])))
        if contains.shape!=original_shape:
            contains = np.repeat(contains,2)
        return contains
    def plot(self,axis=None,color='red'):
        '''
        Plot the eye shape, or add the eyeshape to an existing plot
        
        Parameters
        ----------
        axis : TYPE, optional
            An existing axis to add this artist to or None. 
            The default is None.
        color : str, optional
            The matplotlib colorcode for the artist. The default is 'red'.
            
        Returns
        -------
        (Artist,Artist) pair
        '''
        x = np.linspace(self.medial[0],self.lateral[0],50)
        upper = self.upper_parabola(x)
        lower = self.lower_parabola(x)
        if axis==None:
            artist1, = plt.plot(x,upper,color=color)
            artist2, = plt.plot(x,lower,color=color)
            plt.show()
        else:
            artist1, = axis.plot(x,upper,color=color)
            artist2, = axis.plot(x,lower,color=color)
        return (artist1,artist2)
    def curves(self):
        x = np.linspace(self.medial[0],self.lateral[0],50)
        upper = self.upper_parabola(x)
        lower = self.lower_parabola(x)
        return (x,upper,lower)
    def __bool__(self):
        return self.valid

class FittedEllipse:
    def __init__(self,*args):
        '''
        A least-squares fitted ellipse.
        
        Parameters
        ----------
        *args : An arraylike of points XY (shape Nx2), 
                 or two arraylikes of ordinates, X and Y (shape N)
            The points against which to fit the ellipse.
            
        Raises
        ------
        np.linalg.LinAlgError
            Raised when there are an insufficient number of points
            to fit the ellipse, or when the resultant matrix is singular.
            
        Attributes
        -------
        centre_x: x ordinate of ellipse centre
        centree_y: y ordinate of ellipse centre
        centre:   (centre_x,centre_y)
        angle:   rotation of the ellipse from horizontal, in degrees
                   counterclockwise
        axes:    major and minor axes (order not guaranteed)
        area:    ellipse area
        points:  a series of points on the ellipse, for plotting
        '''
        if len(args)==1:
            xy = np.array(args[0])
            xy = xy.reshape(-1,2)
            x = xy[:,0:1]
            y = xy[:,1:]
        elif len(args)==2:
            x = args[0]
            y = args[1]
        #Require at least 6 points#
        if x.shape[0]<6:
            raise np.linalg.LinAlgError("Insufficient data for fitting ellipse")
        
        #The general form of a conic is Ax**2 + By**2 + Cx + Dy + E == 0
        D=np.hstack([x*x,x*y,y*y,x,y,np.ones(x.shape)])
        #But we can't just least-squares solve this matrix equation, because
        #we could end up with any conic! We need to constrain to the ellipse
        #case, ie we have a constrained minimization problem.

        #The algorithm to do this is from here: 
        #https://github.com/ndvanforeest/fit_ellipse/blob/master/fitEllipse.pdf
        #With additional adaptions from here: 
        #https://stackoverflow.com/a/48002645/12488760
        S=np.dot(D.T,D)
        C=np.zeros([6,6])
        C[0,2]=C[2,0]=2
        C[1,1]=-1
        E,V=np.linalg.eig(np.dot(np.linalg.inv(S),C))
        n=np.argmax(E)
        a=V[:,n]
        b,c,d,f,g,a=a[1]/2., a[2], a[3]/2., a[4]/2., a[5], a[0]
        num=b*b-a*c
        if num==0:
            raise np.linalg.LinAlgError("Insufficient data to fit an ellipse")
            
        #Now we know the general form coefficients, so we can convert them
        #to useful quantities
        self.centre_x = (c*d-b*f)/num
        self.centre_y = (a*f-b*d)/num
        self.centre = (self.centre_x,self.centre_y)
        self.angle = 0.5*np.arctan(2*b/(a-c))*180/np.pi #In degrees
        
        up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
        down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        
        radius_horizontal=np.sqrt(abs(up/down1))
        radius_vertical  =np.sqrt(abs(up/down2))
        
        self.axes = (radius_horizontal,radius_vertical)
        self.area = np.pi*radius_horizontal*radius_vertical
        
        
        ell=Ellipse(self.centre,
                    radius_horizontal*2.,
                    radius_vertical*2.,
                    self.angle)
        self.points=ell.get_verts()
    
    def plot(self, axis = None, color='blue'):
        '''
        Plot the ellipse , or add the ellipse to an existing plot
        
        Parameters
        ----------
        axis : TYPE, optional
            An existing axis to add this artist to or None. 
            The default is None.
        color : str, optional
            The matplotlib colorcode for the artist. The default is 'red'.
            
        Returns
        -------
        None.
        '''
        if axis==None:
            plt.plot(self.points[:,0],
                     self.points[:,1],
                     color= color,
                     label = "Fitted Ellipse")
        else:
            artist, = axis.plot(self.points[:,0],
                      self.points[:,1],
                      color= color,
                      label = "Fitted Ellipse")
            return artist
    def __repr__(self):
        return f"Fitted ellipse: area={self.area:.2f} angle={self.angle:.0f}°"
    

def reject_outliers(group, stds=3):
    group = group.copy()
    group[np.abs(group - np.nanmean(group)) > stds * np.nanstd(group)] = np.nan
    return group


def unit_test_random():
    '''Randomly generate an ellipse, add points to it with
    some degree of error, and then attempt to fit an ellipse.
    '''
    N = 6
    DIM=2
    # Generate random points on the unit circle by sampling uniform angles
    theta = np.random.uniform(0, 2*np.pi, (N,1))
    eps_noise = 0.1 * np.random.normal(size=[N,1])
    circle = np.hstack([np.cos(theta), np.sin(theta)])

    # Stretch and rotate circle to an ellipse with random linear tranformation
    B = np.random.randint(-3, 3, (DIM, DIM))
    noisy_ellipse = circle.dot(B) + eps_noise
    X = noisy_ellipse[:,0:1]
    Y = noisy_ellipse[:,1:]
    plt.scatter(X, Y, label='Data Points')
    phi = np.linspace(0, 2*np.pi, 1000).reshape((1000,1))
    c = np.hstack([np.cos(phi), np.sin(phi)])
    ground_truth_ellipse = c.dot(B)
    plt.plot(ground_truth_ellipse[:,0], ground_truth_ellipse[:,1], 'k--', label='Generating Ellipse')
    fitted_ellipse = FittedEllipse(X,Y)
    try:
        fitted_ellipse.plot()
    except:
        pass
    plt.legend()
    print(fitted_ellipse)
 
def process_dataframe(df, csv= True):
    df.columns = [part+coord for part,coord in zip(df.loc[0],df.loc[1])]
    df = df.rename({"bodypartscoords":"path"},axis='columns')
    df = df.loc[2:]  
    return df
    
def labelcsv_as_dataframe(path):
    df = pd.read_csv(path)
    return process_dataframe(df)

def unit_test_data(path):
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
    try:
        ellipse = FittedEllipse(pupil)
    except np.linalg.LinAlgError:
        ellipse = False
    fittedeye = FittedEyeShape(eye)
    img = mpimg.imread(abs_path)
    fig,ax = plt.subplots(ncols = 3, figsize = (10,2.3))
    #Plot of labelled points
    ax[0].imshow(img)
    ax[0].plot(eye[:,0], eye[:,1], 'o', color = "red")
    ax[0].plot(pupil[:,0], pupil[:,1], 'o', color = "blue")
    #Plot of ellipse on image
    ax[1].imshow(img)
    if ellipse:
        ellipse.plot(ax[1],color='blue')
    fittedeye.plot(ax[1])
    #plot of ellipse alone
    fittedeye.plot(ax[2])
    if ellipse:
        ellipse.plot(ax[2])
    ax[2].set_xlim(ax[1].get_xlim())
    ax[2].set_ylim(ax[1].get_ylim())
    fig.show()
 
def get_plot_of_extracted_eye(row, fig = None, ax = None, artists = None):
    '''
    Produce a nice plot of the eye position inferable from a list of points.

    Parameters
    ----------
    row : pandas.Series
        An iterable of interlaced x and y coordinates for the placement of 
        eye markers, as output by DeepLabCut.
    fig : Figure, optional
        A pre-existing figure. The default is None.
    ax : tuple, optional
        A prexisting (axes, axes) pair returned from a previous
        call to this function. The default is None.
    artists : iterable, optional
        A preexisting iterable of artists, returned from a previous
        call to this function. The default is None.

    Returns
    -------
    fig : Figure
    ax : (axes,axes) pair
    artists : iterable of artists

    '''
    row = row[np.arange(row.shape[0])%3!=2] #Drop likelihoods
    pupil = row[:-8]
    eye = row[-8:]
    fittedeye = FittedEyeShape(eye,allow_invalid=True)
    pupil_okay_idxs = fittedeye.contains(pupil)
    pupil_okay = pupil[pupil_okay_idxs]
    pupil_bad = pupil[~pupil_okay_idxs]
    try:
        ellipse = FittedEllipse(pupil_okay)
    except np.linalg.LinAlgError:
        ellipse = False
    if (not fig) and (not ax):
        fig,ax = plt.subplots(ncols = 2,figsize = (10,3.5))
        ax[0].set_title("Network prediction")
        ax[1].set_title("After quality control")
        ax[0].set_xlim(0,640)
        ax[0].set_ylim(480,0)
        ax[1].set_xlim(0,640)
        ax[1].set_ylim(480,0)
        eyeline1, eyeline2 = fittedeye.plot(ax[0])
        scatter1, = ax[0].plot(
            pupil_okay[::2],pupil_okay[1::2],'o',color='green')
        scatter2, = ax[0].plot(
            pupil_bad[::2],pupil_bad[1::2],'o',color='red')
        if fittedeye:
            eyeline3, eyeline4 = fittedeye.plot(ax[1])
        else:
            eyeline3, = ax[1].plot([],[])
            eyeline4, = ax[1].plot([],[])
        if fittedeye and ellipse:
            ellipseline = ellipse.plot(ax[1])
        else:
            ellipseline, = ax[1].plot([],[])
    else:
        (eyeline1,eyeline2,eyeline3,eyeline4,
               scatter1,scatter2, ellipseline) = artists
        x, upper, lower = fittedeye.curves()
        eyeline1.set_data(x, upper)
        eyeline2.set_data(x,lower)
        if fittedeye:
            eyeline3.set_data(x,upper)
            eyeline4.set_data(x,lower)
        else:
            eyeline3.set_data([],[])
            eyeline4.set_data([],[])
        scatter1.set_data(pupil_okay[::2],pupil_okay[1::2])
        scatter2.set_data(pupil_bad[::2],pupil_bad[1::2])
        if fittedeye and ellipse:
            ellipseline.set_data(ellipse.points[:,0],ellipse.points[:,1])
        else:
            ellipseline.set_data([],[])
        for artist in (eyeline1,eyeline2,scatter1,scatter2):
            ax[0].draw_artist(artist)
        for artist in (eyeline3, eyeline4,ellipseline):
            ax[1].draw_artist(artist)
        fig.canvas.update()
        fig.canvas.flush_events()
    artists = [eyeline1,eyeline2,eyeline3,eyeline4,
               scatter1,scatter2, ellipseline]
    return fig, ax, artists

def create_video(h5_path, file_path, n = None):
    '''
    Create a video of the extracted eye shapes from CNN output
    of DeepLabCut. Be aware that this could take quite a long time.

    Parameters
    ----------
    h5_path : str
        path to an h5 file created with deeplabcut.analyze_video.
    file_path : str
        path of the output video.
    n : int
        maximum number of frames to animate.

    Returns
    -------
    None.

    '''
    try: 
        os.mkdir(file_path)
    except:
        for file in os.listdir(file_path):
          os.remove(os.path.join(file_path,file))
          
    df = pd.read_hdf(h5_path)
    for idx, row in df.iterrows():
        if idx==0:
            fig,ax,artists = get_plot_of_extracted_eye(row)
        else:
            get_plot_of_extracted_eye(row,fig,ax,artists)
        target_path = os.path.join(file_path,f'{idx}.png')
        fig.savefig(target_path)
        if n!=None and n>0 and idx>=n:
            break
    os.system(
          "C:\\Users\\viviani\\ffmpeg\\bin\\.\\ffmpeg.exe -r 30 -f image2 -i"+
          f" {file_path}\\%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p"+
          f" {file_path}.mp4")
    for file in os.listdir(file_path):
          os.remove(os.path.join(file_path,file))
    os.rmdir(file_path)
    plt.close('all')
        
def get_pupil_size_at_each_eyecam_frame(h5_path):
    df = pd.read_hdf(h5_path)
    results = np.empty(df.shape[0])
    for idx, row in df.iterrows():
        row = row[np.arange(row.shape[0])%3!=2] #Drop likelihoods
        pupil = row[:-8]
        eye = row[-8:]
        fittedeye = FittedEyeShape(eye,allow_invalid=True)
        pupil_okay_idxs = fittedeye.contains(pupil)
        pupil_okay = pupil[pupil_okay_idxs]
        try:
            ellipse = FittedEllipse(pupil_okay)
        except np.linalg.LinAlgError:
            ellipse = False
        if ellipse and fittedeye:
            results[idx] = ellipse.area
        else:
            results[idx] = np.nan
        if (idx%1000==0):
            print(f"{idx}/{df.shape[0]}")
    results = reject_outliers(results)
    return results
        



if __name__=="__main__":
    # im = unit_test_data(
    #     "C:/Users/viviani/Desktop/micepupils-viviani-2020-07-09/labeled-data/2017-03-30_01_CFEB045_eye/CollectedData_viviani.csv")
    
    res = get_pupil_size_over_time("C:/Users/viviani/Desktop/micepupils-viviani-2020-07-09/videos/"+
                                   "2016-05-28_02_CFEB014_eyeDLC_resnet50_micepupilsJul9shuffle1_1030000.h5")
    res_outlier_removed = reject_outliers(res)
    plt.plot(np.array(list(range(len(res_outlier_removed))))/30,res_outlier_removed)
    plt.show()
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import matplotlib.image as mpimg

seaborn.set_style("dark")

plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 11

class Line:
    '''
    Encapsulates a linear function
    and its inverse
    '''
    def __init__(self,m,c):
        self.m = m
        self.c = c
    def __call__(self,x):
        return self.m*x+self.c
    def inverse(self):
        return Line(1/self.m, -self.c/self.m)




class RetinalReceptiveField:
    gauss = staticmethod(
        lambda r2,a,b:a*np.exp(-b*(r2))
        )
    def __init__(self,xy = (100,150),sigma = 150,
                 shape = (1000,1000)):
        field = np.empty(shape)
        self.xy = xy
        self.sigma = sigma
        for (x,y),_ in np.ndenumerate(field):
            field[x][y] = self.field_fn(x,y)
        self.field = field
    def field_fn(self,x,y):
        _x = x - self.xy[0]
        _y = y - self.xy[1]
        r2 = (_x**2+_y**2) / ((self.sigma)**2)
        return (10/3)*(self.gauss(r2,1,4)-self.gauss(r2,0.5,2))
    
    
    def apply_to_axis(self, axis):

        axis.imshow(self.field,
                    cmap = "gray",
                    interpolation = "bicubic",
                    vmin=-1,
                    vmax= 1)
        axis.set_yticks([])
        axis.set_xticks([])
        return axis

class Image:
    def __init__(self,data):
        self.field = (data / 255)*2 - 1
    def apply_to_axis(self, axis):

        axis.imshow(self.field,
                    cmap = "gray",
                    interpolation = "bicubic")
        axis.set_yticks([])
        axis.set_xticks([])
        return axis


class HubelWeiselFigure:
    caption = ("Figure 1: A bipolar retinal cell's receptive field. "
               "On-centre lumiance or off-centre darkness increases "
               "neuron firing; on-centre\n darkness or off-centre luminance "
               "decreases it. Mathematically, this is equivalent to "
               "convolving the receptive field (a)\nwith the stimulus (b) "
               "and responding to the average brightness of the output (c).")
    def __init__(self):
        im = mpimg.imread("cat.jpg")[100:-100:,300:-300]
        im = im.mean(axis=-1)        #convert to grayscale
        im = np.minimum(im/140*256,np.ones(im.shape)*255)
        field0 = RetinalReceptiveField(shape = im.shape)
        field1 = RetinalReceptiveField(shape = im.shape, xy = (530,510))
        im = Image(im)
        self.fig, ax = plt.subplots(ncols=3,
                                    nrows=2,
                                    figsize=(8,6),
                                    tight_layout=True)
        (stim_ax, field_ax, conv_ax) = ax.transpose()
        field_ax[0].set_title("$\\bf{(B)}$ Bipolar cell receptive field")
        field0.apply_to_axis(field_ax[0])
        field1.apply_to_axis(field_ax[1])
        stim_ax[0].set_title("$\\bf{(A)}$ Visual stimulus")
        im.apply_to_axis(stim_ax[0])
        im.apply_to_axis(stim_ax[1])
        conv_ax[0].set_title("$\\bf{(C)}$ Convolved stimulus")
        conv_ax[0].imshow(field0.field*im.field,
                       cmap = 'gray',
                       interpolation = 'bicubic',
                       vmin=-1,
                       vmax = 1)
        conv_ax[1].imshow(field1.field*im.field,
                       cmap = 'gray',
                       interpolation = 'bicubic',
                       vmin=-1,
                       vmax = 1)
        conv_ax[0].set_yticks([])
        conv_ax[0].set_xticks([])
        conv_ax[1].set_yticks([])
        conv_ax[1].set_xticks([])
    def show(self):
        self.fig.show()
        

class TopDownIllusionFigure:
    caption = (
        "Figure 3: many optical illusions are thought to be caused\n"
        "by top-down signalling in the visual system. In the Ponzo\n"
        "illusion, two horizontal lines of equal length are perceived\n"
        "to be of different lengths because the other lines in the\n"
        "figure mimic a perspective situation with which we are familiar.\n"
        "Past knowledge is influencing the perception of simple elements\n"
        "of the image."
        )

    def __init__(self):
        self.fig, (self.ax1,self.ax2) = plt.subplots(ncols=2, figsize = (8,6))
        for axis in (self.ax1, self.ax2):
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xlim(0,10)
            axis.set_ylim(0,10)
            rect1 = patches.Rectangle((3.75,1.5),2.5,0.25,
                               color = "blue")
            rect2 = patches.Rectangle((3.75,7),2.5,0.25,
                               color = "blue")
            axis.add_patch(rect1)
            axis.add_patch(rect2)
        for line in self.generate_railroad_lines(3.7):
            self.ax1.add_line(line)
            
        dotted_line1 = lines.Line2D((3.75,3.75),(0,10),
                                    color = 'orange',
                                    linestyle = ":")
        dotted_line2 = lines.Line2D((6.25,6.25),(0,10),
                                    color = 'orange',
                                    linestyle = ":")
        self.ax2.add_line(dotted_line1)
        self.ax2.add_line(dotted_line2)
        self.ax1.set_title("(a)")
        self.ax2.set_title("(b)")
        # self.fig.text(0.1, 0.01, self.caption,
        #               horizontalalignment = 'left')
    def show(self):
        self.fig.show()
    def generate_railroad_lines(self,slope): #() -> [lines.Line2D]
        line1 = Line(10/slope,-5/slope)
        line2 = Line(-10/slope,9.5*10/slope)
        xs1 = (0.5, 4.2)
        xs2 = (9.5,5.8)
        rail1 = lines.Line2D(xs1,tuple(map(line1,xs1)),
                             color = 'black')
        rail2 = lines.Line2D(xs2,tuple(map(line2,xs2)),
                             color = 'black')
        lst = [rail1,rail2]
        for y in range(2,10,2):
            lst.append(lines.Line2D(
                (line1.inverse()(y),line2.inverse()(y)),(y,y),
                color = 'blue')
            )
        return lst
                             



class BayesPlot:
    xs = np.linspace(-5, 5, 200)
    @staticmethod
    def format_axis(axis, legend=True):
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_ylim((0,1.5))
        if legend: axis.legend()
        return axis
    def format_axis_prior(self, axis, prior, likelihood, legend = True):
        prior_line, = axis.plot(self.xs, prior.predict(self.xs))
        prior_line.set_label("Prior")
        likelihood_line, = axis.plot(self.xs,likelihood.predict(self.xs))
        likelihood_line.set_label("Likelihood")
        self.format_axis(axis, legend)
        return axis
    def format_axis_post(self,axis,posterior, legend = True):
        post_line, = axis.plot(self.xs,posterior.predict(self.xs),
                               color = 'green')
        post_line.set_label("Posterior")
        self.format_axis(axis, legend)
        return axis

class GaussianDistribution:
    def __init__(self, mu, sig):
        self.mu = mu
        self.sig = sig
    def predict(self, X):
        return self.gaussian(X,self.mu,self.sig)
    @staticmethod
    def gaussian(x, mu, sig):
        coeff = 1/(sig * np.sqrt(2*np.pi))
        return coeff * np.exp(-(x-mu)*(x-mu) / (2 * sig * sig))


class PosteriorGaussianDistribution(GaussianDistribution):
    def __init__(self,prior,likelihood):
        p = prior
        l = likelihood
        self.sig = np.sqrt(p.sig**2*l.sig**2)/(p.sig**2 + l.sig**2)
        self.mu = (p.sig**(-2)*p.mu + l.sig**(-2)*l.mu)/(p.sig**(-2)+l.sig**(-2))
        

        

def test():
    xs = np.linspace(-5, 5, 200)
    dist = GaussianDistribution(0,0.5)
    plt.plot(xs,dist.predict(xs))
    plt.show()
        

class PredictiveCodingPlot(BayesPlot):
    labels = ("(a)","(b)","(c)","(d)")
    caption = ("Figure 4: The bayesian formaulation of predictive coding.(a) A\n"
               "prior belief measures the probability of various environmental\n"
               "states based on prior knowlege. (b) When new information is received,\n"
               "the likelihood of seeing it for each environmental state comprises\n"
               "a likelihood function. (c) Together these can produce a new prediction\n"
               "of the probabilities of external states, a posterior distribution.\n"
               "(d) How much this differs from the prior is the prediction error.\n"
               "\n"
               )
    def __init__(self, prior_mu = 0, prior_sig = 1,
                 likeli_mu = 3, likeli_sig = 1.4):
        prior      = GaussianDistribution(prior_mu, prior_sig)
        likelihood = GaussianDistribution(likeli_mu, likeli_sig)
        posterior  = PosteriorGaussianDistribution(prior,likelihood)
        self.fig, self.axes = plt.subplots(nrows = 4, figsize = (8,6))
        self.axes = self.axes.flatten()
        self.fig.text(0.5, 0.05, 'Environmental State', ha='center', va='center')
        self.fig.text(0.03, 0.5, 'Probability Density', ha='center', va='center',
                 rotation='vertical')
        # self.fig.text(0.03, 0.01, self.caption, ha = 'left')
        for idx,axis in enumerate(self.axes):

            self.format_axis(axis, legend = False)
            prior_line, = axis.plot(self.xs, prior.predict(self.xs))
            if idx in (0,): prior_line.set_label("Prior")
            if idx in (1,2):
                likelihood_line, = axis.plot(self.xs,likelihood.predict(self.xs))
                if idx==1: likelihood_line.set_label("Likelihood")
            if idx>1:
                post_line, = axis.plot(self.xs,posterior.predict(self.xs),
                                       color = 'green')
                if idx==2: post_line.set_label("Posterior")
            if idx==3:
                axis.axvline(prior.mu,
                             linestyle = ':',
                             color = prior_line.get_color())
                axis.axvline(posterior.mu,
                             linestyle = ':',
                             color = post_line.get_color())
                arrow = axis.arrow(prior.mu, 1,
                                (posterior.mu - prior.mu),
                                0,
                                color = 'red',
                                length_includes_head = True,
                                head_width = 0.2,
                                head_length = 0.2
                            )
                axis.legend([arrow],['Prediction Error'])
            else:
                axis.legend()
            axis.set_ylabel(self.labels[idx],
                            rotation = "horizontal",
                            labelpad = 9)
    def show(self):
        self.fig.show()







class ContrastResponsePlot:
    caption = ("Figure 5: The effects of attention on the contrast response "
               "function of a V1 neuron.\n"
               "(a) Contrast gain, where the sensitivity of the neuron to its "
               "stimulus increases for all\n"
               "contrast levels, as though the stimulus had higher contrast. "
               "(b) Response gain, where \n"
               "the maximum response of the neuron is increased. (c) A "
               "combination of both contrast\n"
               "and response gain "
               "is sometimes observed. (adapted from Carrasco 2011)")
    xs = np.linspace(-5,5,50)
    log_fn = staticmethod(
        lambda limit,midpoint,X:limit/(1+np.exp(midpoint-X))
        )
    def __init__(self):
        self.fig,self.axes= plt.subplots(ncols=3, figsize = [8,2.5])
        for axis in self.axes:
            axis.set_yticks([])
            axis.set_xticks([])
            axis.set_xlabel("Stimulus Contrast")
            axis.set_ylabel("Neuron Response")
            axis.plot(self.log_fn(1,0,self.xs))
            axis.set_ylim((0,1.2))
        #First Axis
        self.axes[0].set_title("(a) Contrast Gain")
        self.axes[0].plot(self.log_fn(1,-1,self.xs),
                          linestyle = ":",
                          color = "blue")
        self.axes[1].set_title("(b) Response Gain")
        self.axes[1].plot(self.log_fn(1.15,0,self.xs),
                          linestyle = ":",
                          color = "blue")
        self.axes[2].set_title("(c) Contrast and Response Gain")
        self.axes[2].plot(self.log_fn(1.15,-1,self.xs),
                          linestyle = ":",
                          color = "blue")
        # self.fig.text(0.0625,0.1,self.caption)
    def show(self):
        self.fig.show()
    def __call__(self):
        self.fig.show()




class PsychosisPlot(BayesPlot):
    # caption = ("Figure 6: The predictive coding account of psychosis. Compare "
    #            "the healthy response to an unexpected \n"
    #            "stimulus (a) to the psychotic response characterised "
    #            "by a less certain prior and overestimation of the\n"
    #            "signal's precision (b). This leads to a larger shift in, "
    #            "and higher estimated precision of, the posterior. \n"
    #            "This results in a radically altered experience from noise in "
    #            "the environment.")
    def __init__(self):
        healthy_prior = GaussianDistribution(0, 0.3)
        psych_prior = GaussianDistribution(0,1.65)
        healthy_likelihood = GaussianDistribution(3, 0.7)
        psych_likelihood = GaussianDistribution(3, 0.5)
        healthy_posterior  = PosteriorGaussianDistribution(healthy_prior,
                                                           healthy_likelihood)
        
        psych_posterior  = PosteriorGaussianDistribution(psych_prior,
                                                           psych_likelihood)

        self.fig,ax =plt.subplots(nrows = 2, ncols = 2, figsize = (8,6))
        healthy,psych = ax.transpose()
        # fig.text(0.03, 0.1, self.caption, ha = 'left')

        self.format_axis_prior(healthy[0],healthy_prior,healthy_likelihood)
        healthy[0].set_title("(a) Healthy Response")
        healthy[0].set_ylabel("Probability Density")
        self.format_axis_prior(psych[0], psych_prior, psych_likelihood,
                               legend = False)
        
        psych[0].set_title("(b) Psychosis")

        
        self.format_axis_post(healthy[1], healthy_posterior)
        healthy[1].set_xlabel("Environmental State")
        healthy[1].set_ylabel("Probability Density")
        
        self.format_axis_post(psych[1], psych_posterior,
                              legend = False)
        psych[1].set_xlabel("Environmental State")

    def __call__(self):
        self.show()
    def show(self):
        self.fig.show()


if __name__=="__main__":
    plt.close("all")
    HubelWeiselFigure().show()
    # TopDownIllusionFigure().show()
    # PredictiveCodingPlot().show()
    # ContrastResponsePlot().show()
    # PsychosisPlot().show()

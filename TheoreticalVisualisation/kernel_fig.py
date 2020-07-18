import matplotlib.pyplot as plt
import numpy as np
import seaborn


class KernelExampleFigure:
    seaborn.set_style("dark")
    lick_kernel_fn = lambda x:100*x**2*(1-x)**7
    lick_kernel_series = lick_kernel_fn(np.linspace(0,1,30))
    reward_kernel_fn = lambda x:-200*x**4*(1-x)**5
    reward_kernel_series = reward_kernel_fn(np.linspace(0,1,30))
    licks = (1,5.7,8,8.2,8.4)
    rewards = (3,5,6)
    def __init__(self):
        fig = plt.figure()
        lick_kernel   = fig.add_axes((0.1,0.7,0.35,0.2))
        reward_kernel = fig.add_axes((0.55,0.7,0.35,0.2))
        event_train   = fig.add_axes((0.1,0.45,0.8,0.1))
        prediction    = fig.add_axes((0.1,0.1, 0.8,0.2))
        for ax in (lick_kernel,reward_kernel):
            ax.set_ylim((-0.5,1.2))
        for ax in (event_train,prediction):
            ax.set_xlim((0,12))
        lick_kernel.set_title("Licking kernel")
        lick_kernel.set_xlabel("∆t around a lick")
        lick_kernel.plot(np.linspace(-0.2,2,30),self.lick_kernel_series,
                         color = 'blue')
        reward_kernel.set_title("Reward kernel")
        reward_kernel.plot(np.linspace(-0.2,2,30),self.reward_kernel_series,
                           color = 'orange')
        reward_kernel.set_xlabel("∆t around a reward")
        event_train.set_title("Event Occurances")
        event_train.set_yticks([])
        event_train.vlines(self.licks,0,1,colors = 'blue',label='lick')
        event_train.vlines(self.rewards,0,1,colors='orange',label='rewards')
        event_train.legend()
        prediction.set_title("Predicted Fluoresence Response")
        prediction.set_xlabel("time")
        prediction.set_ylabel("∆f/f")
        prediction.plot(*self.predict(),color='k')
        for lick in self.licks:
            prediction.plot(np.linspace(lick-0.2,lick+2,30),
                            self.lick_kernel_series,
                            color = 'blue',
                            linestyle = '-',
                            alpha = 0.5)
        for reward in self.rewards:
            prediction.plot(np.linspace(reward-0.2,reward+2,30),
                            self.reward_kernel_series,
                            color = 'orange',
                            linestyle = '-',
                            alpha = 0.5)
        self.fig = fig
    def predict(self):
        time = np.linspace(0,12,180)
        prediction = np.zeros(time.shape)
        for lick in self.licks:
            start = np.searchsorted(time,lick-0.2)+1
            prediction[start:start+len(
                self.lick_kernel_series)]+=self.lick_kernel_series
        for reward in self.rewards:
            start = np.searchsorted(time,reward-0.2)+1
            prediction[start:start+len(
                self.reward_kernel_series)]+=self.reward_kernel_series
        return (time,prediction)
            
            
        

    def show(self):
        self.fig.show()
        
if __name__=="__main__":
    fig = KernelExampleFigure()
    fig.show()

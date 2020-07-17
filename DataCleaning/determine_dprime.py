##Stimulus feature encodings
GOLEFT    = 3
GORIGHT   = 4
NOGOLEFT  = 5
NOGORIGHT = 6
GO        = {GOLEFT, GORIGHT}
NOGO      = {NOGOLEFT, NOGORIGHT}


__debug = False
from accdatatools.Utils.deeploadmat import loadmat
import accdatatools.Utils.acc_path_tools as p
from scipy.stats import norm
from sys import float_info
import numpy as np
import os



##  Statistics helper functions

def restrict_to_open_interval(x, a, b):
    '''Rounds x to the nearest representable element of the open interval (a,b)
    ::rtype:: float
    '''
    if x >= b:
        x = b -  float_info.min
    elif x <= a:
        x = a + float_info.min
    return x

def d_prime(hit_rate, false_alarm_rate):
    hit_rate = restrict_to_open_interval(hit_rate, 0, 1)
    false_alarm_rate = restrict_to_open_interval(false_alarm_rate, 0, 1)
    return norm.ppf(hit_rate)  - norm.ppf(false_alarm_rate)

#--------------------------------------------------

def extract_stim_corr_resptype(matfile_path):
    '''Returns a list of 3-tuples of (stimIDs, correct, responseType) from
    a 'psychstim' matlab file. 
    StimID is a numeric code for the stimulus type,
    correct is true for a successful behaviour response else false,
    responseType is true if Go behavior else false.
    ::rtype:: [(int,int,int)]
    '''
    file = loadmat(matfile_path)
    trials = file['expData']['trialData']
    try:
        stimIDs   = [trial.stimID for trial in trials]
        corrects   = [trial.correct for trial in trials]
        responses = [trial.responseType for trial in trials]
        return zip(stimIDs,corrects,responses)
    except AttributeError:
        raise AttributeError(
                "That matlab file does not have behavioral structure")

def calc_d_prime(matfile_path):
    '''
        Calculates the Dprime statistic for a trial from that trial's psychstim
        matlab file.

    Parameters
    ----------
    matfile_path : str
        Path to a psychstim.mat matlab file.

    Raises
    ------
    AttributeError
        Raised when the trial referenced by matfile_path is either not a 
        formatted psychstim file or is not a behavioural experiment and 
        d-prime cannot be calculated from it.
        Specifically checks whether the file has the stimID, correct, 
        and responseType fields.

    Returns
    -------
    Int
        The dprime statistic for the experiment.

    '''
    hits         = 0
    false_alarms = 0
    n = 0
    try:
        for (stim, corr, resp) in extract_stim_corr_resptype(matfile_path):
            if stim in GO:
                hits += corr
                false_alarms += not corr
                n+=1
    except AttributeError:
        raise AttributeError(
            'That file does not refer to an experimental trial'
            )
    try:
        hit_rate, false_alarm_rate = (hits/n, 
                                  false_alarms/n)
        return d_prime(hit_rate,false_alarm_rate)
    except ZeroDivisionError:
        return np.nan

def get_dprimes_from_dirtree(path):
    '''
    Determines the Dprime statistic for each trial for each mouse in the
    directory rooted at path.


    Parameters
    ----------
    path : str
        The root path from which to fetch trials.

    Returns
    -------
    results : list
        A list of 3-tuples of 
        (str mouse_ID, str experiment_path, float dprime)

    '''
    performances = {}

    for root, dirs, files in os.walk(path):
        for file in files:
            if 'psychstim' in file:
                try:
                    performances[root] = calc_d_prime(os.path.join(root,file))
                except AttributeError:
                    if(__debug==True):
                        print(f"Non-behavioural exp located at {root}")
                    else:
                        pass
    results = []
    
    for root, dprime in performances.items():
        if np.isfinite(dprime):
            results.append((
                p.mouse_id(root),
                p.exp_id(root),
                dprime))

    return results


if __name__=='__main__':
    # import pickle
    # #For the purpose of graphing the dprime statistics, dump it as a pickle
    # #file.
    # results = get_dprimes_from_dirtree('D:\\Local_Repository')
    # # with open('C:\\Users\\uic\\Desktop\\resultsdump.p', 'wb') as file:
    #     results = get_dprimes_from_dirtree('D:\\Local_Repository')
    #     pickle.dump(results, file)
    
    print(sum(len([get_dprimes_from_dirtree('D:\\Local_Repository').items()][1])))




# -*- coding: utf-8 -*-

import scipy.io as io


def loadmat(matfile_path):
    '''
    Wrapper of scipy.io.loadmat that more deeply interrogates matlab objects.
    '''
    data = io.loadmat(matfile_path, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dic):
    '''
    Checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dic:
        if isinstance(dic[key], io.matlab.mio5_params.mat_struct):
            dic[key] = _todict(dic[key])
    return dic        

def _todict(matobj):
    '''
    Constructs nested dictionaries from matlab objects.
    '''
    dic = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, io.matlab.mio5_params.mat_struct):
            dic[strg] = _todict(elem)
        else:
            dic[strg] = elem
    return dic
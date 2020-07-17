# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:19:15 2020

@author: Vivian Imbriotis
"""
TESTLEFT  = 1
TESTRIGHT = 2
GOLEFT    = 3
GORIGHT   = 4
NOGOLEFT  = 5
NOGORIGHT = 6
GO        = {GOLEFT, GORIGHT}
NOGO      = {NOGOLEFT, NOGORIGHT}
LEFT      = {TESTLEFT, GOLEFT,NOGOLEFT}
RIGHT     = {TESTRIGHT, GORIGHT,NOGORIGHT}

TRIALTYPE = {
    1: 'TESTLEFT',
    2: 'TESTRIGHT',
    3: 'GOLEFT',
    4: 'GORIGHT',
    5: 'NOGOLEFT',
    6: 'NOGORIGHT',
    7: 'STIMCODE7',
    8: 'STIMCODE8'
    }

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:59:02 2020

@author: viviani
"""


def item(ls):
    '''As numpys ndarray.item method but takes list, list of list, etc'''
    if type(ls)==list and len(ls)==1:
        return ls[0] if type(ls[0])!=list else item(ls[0])
    else:
        raise ValueError("ls must be a list with a single element")
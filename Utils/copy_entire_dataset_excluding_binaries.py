# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 12:58:23 2020

@author: Vivian Imbriotis
"""

from shutil import copy_tree
from os.path import splitext

def is_binary(path):
    file,extension = splitext(path)
    if extension in (".bin",".dat",".tif",".tiff"):
        return True
    return False

def binaries_to_exclude(root,children):
    return [child for child in children if is_binary(child)]

def copy_whole_dataset_to_dir(root = "H:\\",
                              new_directory = "C:/Users/viviani/desktop/ACCDatasetCopy"
                              ):
    copy_tree(root,new_directory,ignore=binaries_to_exclude)
    

if __name__=="__main__":
    copy_whole_dataset_to_dir()
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 12:58:23 2020

@author: Vivian Imbriotis
"""

from shutil import copytree
import os.path as path

def is_binary(filepath):
    file,extension = path.splitext(filepath)
    if extension in (".bin",".dat",".tif",".tiff"):
        return True
    return False


def files_to_exclude(root,children,dst = "D:/vivian/DataSetCopy"):
    return [c for c in children if is_binary(c)]

def copy_whole_dataset_to_dir(root = "H:\\",
                              new_directory = "D:/vivian/DataSetCopy"
                              ):
    copytree(root,new_directory,ignore=files_to_exclude)
    

if __name__=="__main__":
    copy_whole_dataset_to_dir()
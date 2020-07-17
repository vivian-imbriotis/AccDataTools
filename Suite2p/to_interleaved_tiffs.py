# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:11:05 2020

@author: uic
"""
from os import listdir
from os.path import join, isfile
from acc_path_tools import get_all_files_with_name
from get_tiff_from_binary import extract_all_frames

def collect_planes_in(path):
    res = []
    val = join(path,'suite2p')
    for folder in listdir(val):
        if folder[:-1] == 'plane': 
            target = join(val,folder)
            raw = join(target,'raw_data.bin')
            nonraw = join(target,'data.bin')
            if isfile(raw):
                res.append(raw)
            elif isfile(nonraw):
                res.append(nonraw)
            else:
                raise FileNotFoundError(
                    f'No data found in {target}')
    return res
                

def collect_channels_in(path):
    res = []
    res += collect_planes_in(path)
    for folder in listdir(path):
        if folder[:-1]=='cd':
            res += collect_planes_in(folder)
    return res


def main(read_path,write_path,verbose=True):
    res = get_all_files_with_name(read_path, name='data.bin')
    exps = set(list(zip(*res))[1])
    for experiment in exps:
        print(experiment)
        paths = collect_channels_in(experiment)
        print(paths)
        extract_all_frames(paths, write_path, verbose=verbose)



if __name__=='__main__':
    read_path = 'G:\\Local_Repository'
    write_path = 'D:\\TIFF_FILES'
    main(read_path,write_path)
    
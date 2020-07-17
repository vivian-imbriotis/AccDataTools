# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:57:51 2020

@author: uic
"""
from acc_path_tools import get_all_files_with_name
from get_tiff_from_binary import extract_all_frames
from os.path import split,join

def get_path_after_str(path,strng):
    head, tail = split(path)
    if head == strng:
        return tail
    if head == "":
        raise AttributeError()
    else:
        try:
            return join(get_path_after_str(head,strng),tail)
        except AttributeError:
            raise AttributeError(f"Failed to split {strng} off {path}")

def main(read_dir, write_dir):
    all_files = get_all_files_with_name(read_dir, name='data.bin')
    _,_,files=zip(*all_files)

    for idx, file in enumerate(files):
        directory,justfile = split(file)
        relpath = get_path_after_str(directory,read_dir)
        new_path = join(write_dir,relpath)
        print(f"Extracting file {idx+1} of {len(files)}")
        extract_all_frames(
            read_path = file,
            target_directory = new_path,
            verbose = True)


if __name__=="__main__":
    READ_PATH = 'G:\\Local_Repository'
    WRITE_PATH = 'D:\\TIFF_FILES'
    main(READ_PATH,WRITE_PATH)
    
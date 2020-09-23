# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:59:17 2020

@author: Vivian Imbriotis

A small utility to create a text version of this entire repository to dump
to a txt file, so all the code can be put into an appendix.
"""

from os import listdir, walk 
from os.path import isdir, join, split, splitext
from contextlib import redirect_stdout

import accdatatools
root = accdatatools.__path__[0]



def tree(path, indent = 0):
    name = split(path)[1]
    print(f"{''.join([' |']*indent)}{name}")
    if isdir(path) and "git" not in path:
        [tree(join(path,f),
              indent=indent+1) for f in listdir(path)]

def dump_formatted_file_content(path):
    '''
    Cat the contents of a file to stdout, prepending each line with
    a nicely formatted line number.

    Parameters
    ----------
    path : str
        Path to the code file.

    '''
    newline = '\n'
    with open(path,'r') as file:
        code = file.readlines()
    for n, line in enumerate(code):
        print(f"{n:<3}| {line.replace(newline,'')}")

def is_text(file):
    '''Check if a file is a text file or a binary file'''
    name, ext = splitext(file)
    if ext in ("",".py",".R",".yaml",".txt"): 
        return True
    return False

def print_all_code(root):
    '''
    Cat formatted representations of the directory structure and the content
    of all text files in a projected rooted at root to stdout.

    Parameters
    ----------
    root : str
        The project's root directory.

    '''
    for root,dirs,files in walk(root):
        rname = root.split("accdatatools")[-1]
        if 'git' not in root:
            for file in files:
                if is_text(file) and file != "quine.txt":
                    print(join(rname,file))
                    dump_formatted_file_content(join(root,file))
                    print("\n")

def main():
    try:
        outfile = open("quine.txt",'w')
        #This is roughly similar to a unix forward pipe operator
        with redirect_stdout(outfile):
            print("Directory Structure\n")
            tree(root)
            print_all_code(root)
    finally:
        outfile.close()

if __name__=="__main__":
    main()
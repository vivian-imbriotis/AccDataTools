# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:16:10 2020

@author: Vivian Imbriotis
"""
from os.path import join, split, splitext, splitdrive, sep
from os import walk, listdir
from collections import defaultdict

from accdatatools.Utils.convienience import item



def as_nested_dict(three_tuple):
    '''
    Parameters
    ----------
    obj : list of 3-tuples

    Returns
    -------
    result : dictionary.
        1-elements of tuples become keys, with dictionaries as values.
        2-elemements of the tuples become keys of those dictionaries,
        with lists of 3-elements as values.
        
        So [(A,a,x),(A,a,y),(A,b,x),(B,a,x)] => 
            {
                A: {
                        a:[x,y],
                        b:[x]                        
                    },
                B: {
                        a:[x]
                        }
            }                                           

    '''
    result = defaultdict(lambda:defaultdict(list))
    for layer1, layer2, layer3 in three_tuple:
        result[layer1][layer2].append(layer3)
    for key,value in result.items():
        value.default_factory = None
    result.default_factory = None
    return result


def prune_path_to_experiment(path):
    head, tail = split(path)
    if tail[:2] == '20':
        return path
    elif head == splitdrive(head)[0]:
        raise AttributeError("Path could not be pruned to experiment")
    else:
        return prune_path_to_experiment(head)

def prune_path_to_mouse(path):
    head, tail = split(path)
    if tail[:4] == 'CFEB':
        return path
    elif head == splitdrive(head)[0]:
        raise AttributeError("Path could not be pruned to mouse")
    else:
        return prune_path_to_mouse(head)

def get_mouse_id(path):
    '''
    Convert a path including a mouse ID to just the mouse ID.
    '''
    path = prune_path_to_mouse(path)
    path = path.split(sep)[-1]
    return path
    

def get_exp_id(path):
    '''
    Convert a path including an experiment ID to just the experiment ID.
    '''
    path = prune_path_to_experiment(path)
    path = path.split(sep)[-1]
    return path

def get_mouse_path(id_str, root):
    '''
    Searches the directory tree rooted at root for a path corresponding to
    the Mouse ID id_str.
    eg 'CFEB015', 'D:\\' => 'D:\\Local_Directory\\CFEB015'
    '''
    for root,dirs,files in walk(root):
        for directory in dirs:
            try:
                if get_mouse_id(directory) == id_str:
                    return join(root,directory)
            except AttributeError: pass
    raise FileNotFoundError(
        "No directory corresponding to that mouse ID found.")
    
    

def get_exp_path(id_str, root):
    '''
    Searches the directory tree rooted at root for a path corresponding to
    the Experiment ID id_str. 
    '''
    for root,dirs,files in walk(root):
        for directory in dirs:
            try:
                if get_exp_id(directory) == id_str:
                    return join(root,directory)
            except AttributeError: pass
    raise FileNotFoundError(
        "No directory corresponding to that Experiment ID found.")

def get_timeline_path(exp_path):
    file = item([file for file in listdir(exp_path) if "Timeline" in file])
    return join(exp_path,file)

def get_psychstim_path(exp_path):
    file = item([file for file in listdir(exp_path) if "psychstim" in file])
    return join(exp_path,file)




def get_all_files_with_condition(path, condition, verbose=False):
    '''
    Searches all files in directories rooted at path, including files
    if they satisfy condition. Typical usage idiom similar to os.walk:
        
        > For mouse, exp, file in get_all_files_with_condition(PATH,cond):
            #do things
        
    If only the files are needed, discard the mouse and experiment information
    
        >for _,_,file in get_all_files_with_condition(PATH, cond)
            #do things

    Parameters
    ----------
    path : str
        The root path to search recursively.
    condition : callable object
        A function that accepts a string filename and returns a bool.
        eg > lambda file: ext==".tif" for _,ext in os.splitext(file)

    Returns
    -------
    result : list
        A list of 3-tuples of 
        (str mouse_path, str experiment_path, str file_path)

    '''
    all_files = set()
    experiments = set()
    mice = set()
    result= []
    for root, dirs, files in walk(path):
        for file in files:
            filename, file_extension = splitext(file)
            if condition(file):
                if verbose:
                    all_files.add(join(root,file))
                    experiments.add(prune_path_to_experiment(root))
                    mice.add(prune_path_to_mouse(root))
                result.append((prune_path_to_mouse(root),
                               prune_path_to_experiment(root),
                               join(root,file)))
    if verbose:
        print(f'\n-----{len(all_files)} Files-----')
        for i in all_files:
            print(i)
            print(f'\n-----{len(experiments)} Experiments-----')
        for i in experiments:
            print(i)
            print(f'\n-----{len(mice)} Mice-----')
        for i in mice:
            print(i)
    return result

def get_all_files_with_name(path, name):
    '''
    Gets all files from a directory tree rooted at path with filename name.

    Parameters
    ----------
    path : str
        The root path to search recursively.
    name : str
        The filename for which to search.

    Returns
    -------
    result : list
        A list of 3-tuples of 
        (str mouse_path, str experiment_path, str file_path)
    '''
    condition = lambda file:file==name
    return get_all_files_with_condition(path, condition)

def get_all_files_with_ext(path,ext):
    '''
    Gets all files from a directory tree rooted at path with file extention
    ext.

    Parameters
    ----------
    path : str
        The root path to search recursively.
    ext : str
        The file extention for which to search, including period, eg '.tif'

    Returns
    -------
    result : list
        A list of 3-tuples of 
        (str mouse_path, str experiment_path, str file_path)
    '''
    condition = lambda file:splitext(file)[1]==ext
    return get_all_files_with_condition(path,condition)

if __name__=="__main__":
    res = get_all_files_with_name('D:\\Local_Repository', ext='data.bin')
    mice = set()
    exps = set()
    files = set()
    for mouse, exp, file in res:
            mice.add(mouse)
            exps.add(exp)
            files.add(file)
    mice = list(mice); mice.sort()
    exps = list(exps); exps.sort()
    files = list(files); files.sort()
    for mouse in mice:
        print(mouse)
    print('\n\n----------\n\n')
    for exp in exps:
        print(exp)
        



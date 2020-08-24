# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 12:50:01 2020

@author: Vivian Imbriotis
"""
import subprocess
from os import path
from accdatatools.RScripts import __path__ as r_script_root

RSCRIPT_INTERPRETER = "c:\\Program Files\\R\\R-3.6.0\\bin\\Rscript.exe"


def execute_r_script(r_script_name, *args):
    try:
        r_script_root_path = r_script_root._path[0]
    except AttributeError:
        r_script_root_path = r_script_root[0]
    path_to_r_script = path.join(r_script_root_path,
                                 r_script_name)
    command = [RSCRIPT_INTERPRETER, path_to_r_script]
    if args:
        command.extend(args)
    print(command)
    result = subprocess.check_output(command)
    return result.replace(b"\r\n",b"\n").decode("utf-8")

if __name__=="__main__":
    a = execute_r_script(
        "pupil_size_mixed_linear_model.R")
    print(a)
    

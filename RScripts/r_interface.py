# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 12:50:01 2020

@author: Vivian Imbriotis
"""
import subprocess
from os import path
from accdatatools.RScripts import __path__ as r_script_root

RSCRIPT_INTERPRETER = "C:\\Program Files\\R\\R-4.0.1\\bin\\Rscript.exe"


def execute_r_script(r_script_name, *args):
    path_to_r_script = path.join(r_script_root._path[0],
                                 r_script_name)
    command = [RSCRIPT_INTERPRETER, path_to_r_script]
    if args:
        command.extend(args)
    result = subprocess.check_output(command)
    return result.decode("utf-8")

if __name__=="__main__":
    a = execute_r_script(
        "example_r_script.r")
    print(a)

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 12:50:01 2020

@author: Vivian Imbriotis
"""
import subprocess
from os import path

RSCRIPT = "C:\\Program Files\\R\\R-4.0.1\\bin\\Rscript.exe"

def execute_r_script(path_to_r_script, *args):
    command = [RSCRIPT, path_to_r_script]
    if args:
        command.extend(args)
    result = subprocess.check_output(command)
    return result.decode("utf-8")

if __name__=="__main__":
    a = execute_r_script(
        "C:/Users/Vivian Imbriotis/Desktop/example_r_script.r")
    print(a)

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 04:20:10 2020

@author: AAKRITI
"""

#Project 1: Iris Dataset - Classification Problem 
#Main script

#Import modules
# =============================================================================
import pandas as pd 
import importlib
import sys


#Import scripts
def f_runScripts(script_name, script_path):
    
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    print(spec)
    imp_script = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = imp_script 
    spec.loader.exec_module(imp_script)
    imp_script
    print(df_raw.head())


if __name__ == '__main__':
    
    print('Main function')
    #Get raw dataframe
    script_name = "make_dataset"
    script_path = "E:/Data Science Projects/1. Iris Dataset - Classification/src/data/make_dataset.py"
    f_runScripts(script_name, script_path)
    print(df_raw.head())
  
    #Get tidy dataframe
    script_name = "clean_data"
    script_path = "E:/Data Science Projects/1. Iris Dataset - Classification/src/data/clean_data.py"
    f_runScripts(script_name, script_path)
    print(df_tidy.head())
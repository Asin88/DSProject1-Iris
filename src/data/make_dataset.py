 # -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 00:09:13 2020

@author: AAKRITI
"""

#Project 1: Iris Dataset - Classification Problem 
#Identify the type of Iris flower from flower characteristics: length and width of sepal and petals.

#Script to load data and make dataset

#Import module
import pandas as pd 
import os

print('Make_Dataset script started')
script_path = os.path.abspath(__file__) # i.e. /path/to/dir/foobar.py
script_dir1 = os.path.split(script_path)[0] #i.e. /path/to/dir/
script_dir2 = os.path.split(script_dir1)[0] #i.e. /path/to/dir/
cwd_dir = os.path.split(script_dir2)[0] #i.e. /path/to/
rel_path = "data\\raw\\iris.csv"
abs_file_path = os.path.join(cwd_dir, rel_path)
df_iris = pd.read_csv(abs_file_path, header='infer', index_col=None)    

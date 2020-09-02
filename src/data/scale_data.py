# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 21:22:05 2020

@author: AAKRITI
"""

#Import modules
from sklearn import preprocessing
from data.clean_data import df_iris
import os

#Function to centre data
def f_centreData():
    df_iris['sepal_length_ctr'] = df_iris.sepal_length - df_iris.sepal_length.mean()
    df_iris['sepal_width_ctr'] = df_iris.sepal_width - df_iris.sepal_width.mean()
    df_iris['petal_length_ctr'] = df_iris.petal_length - df_iris.petal_length.mean()
    df_iris['petal_width_ctr'] = df_iris.petal_width - df_iris.petal_width.mean()
    df_iris.drop(df_iris.columns[[0, 1, 2, 3]], axis = 1, inplace = True)
    
#Function to standardize data
def f_standardizeData():
    df_iris['sepal_length_std'] = df_iris.sepal_length_ctr/df_iris.sepal_length_ctr.std()
    df_iris['sepal_width_std'] = df_iris.sepal_width_ctr/df_iris.sepal_width_ctr.std()
    df_iris['petal_length_std'] = df_iris.petal_length_ctr/df_iris.petal_length_ctr.std()
    df_iris['petal_width_std'] = df_iris.petal_width_ctr/df_iris.petal_width_ctr.std()
    df_iris.drop(df_iris.columns[[1, 2, 3, 4]], axis = 1, inplace = True)    

#Main function
print('Scale_data script started')
#Centre data 
f_centreData()
#Standardize data
f_standardizeData()
#Normalize data
preprocessing.normalize(df_iris)

#Saving scaled data to new csv file
script_path = os.path.abspath(__file__) # i.e. /path/to/dir/foobar.py
script_dir1 = os.path.split(script_path)[0] #i.e. /path/to/dir/
script_dir2 = os.path.split(script_dir1)[0] #i.e. /path/to/dir/
cwd_dir = os.path.split(script_dir2)[0] #i.e. /path/to/
rel_path = "data\\interim\\iris_scaled.csv"
abs_file_path = os.path.join(cwd_dir, rel_path)
df_iris.to_csv(abs_file_path, index = False, encoding='utf-8')
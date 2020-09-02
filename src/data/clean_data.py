# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 04:12:16 2020

@author: AAKRITI
"""

#Project 1: Iris Dataset - Classification Problem 
#Script to clean data

#Import modules
import pandas as pd
import os

#Encode categorical variable
def f_categoryEncoder():
    #Integer encoding
    df_iris["species"] = df_iris["species"].astype('category')
    df_iris["species"] = df_iris["species"].cat.codes #Replace column data with numerical codes
#    #One-hot encoding
#    df_iris = pd.get_dummies(df_iris, columns = ['species'])

#Handle missing data
def f_missingData():
    missing_values = df_iris.isnull()
    if missing_values is None:
        missing_message = "**MISSING VALUES FOUND AND IMPUTED WITH COLUMN MEAN**"
        df_iris.fillna(value={'sepal_length':df_iris.sepal_length.mean(), 'sepal_width':df_iris.sepal_width.mean(), 'petal_length':df_iris.petal_length.mean(), 'petal_width':df_iris.petal_width.mean(), 'species':round(df_iris.species.mean())})
    else:
        missing_message = "No Missing values found"
    return(missing_message)
    
 #Main function   
print('Clean_data script started')
from data.make_dataset import df_iris

#Encode Categories as 0/1
f_categoryEncoder()

#Handle missing data
missing_message = f_missingData()

#Saving clean data to new csv file
script_path = os.path.abspath(__file__) # i.e. /path/to/dir/foobar.py
script_dir1 = os.path.split(script_path)[0] #i.e. /path/to/dir/
script_dir2 = os.path.split(script_dir1)[0] #i.e. /path/to/dir/
cwd_dir = os.path.split(script_dir2)[0] #i.e. /path/to/
rel_path = "data\\interim\\iris_clean.csv"
abs_file_path = os.path.join(cwd_dir, rel_path)
df_iris.to_csv(abs_file_path, index = False, encoding='utf-8')


# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 00:09:13 2020

@author: AAKRITI
"""

#Project 1: Iris Dataset - Classification Problem 
#Script to load data and make dataset

#Import module
import pandas as pd 

#Load data from csv file and create a pandas dataframe



def f_loadData():
    filepath = 'E:/Data Science Projects/1. Iris Dataset - Classification/data/raw/iris.csv'
    global df_raw
    df_raw = pd.read_csv(filepath, header='infer', index_col=None)

if __name__ == '__main__':
    print('Make_Dataset script started')
    f_loadData()
    print(df_raw.head())
    
    
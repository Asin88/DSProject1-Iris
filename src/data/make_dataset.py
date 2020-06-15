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

filepath = 'E:\Data Science Projects\1. Iris Dataset - Classification\data\raw\iris.csv'
df_table = pd.read_csv(filepath, header=0, index_col=None)

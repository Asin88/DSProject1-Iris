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

print('Make_Dataset script started')
filepath = 'E:/Data Science Projects/1. Iris Dataset - Classification/data/raw/iris.csv'
df_iris = pd.read_csv(filepath, header='infer', index_col=None)    

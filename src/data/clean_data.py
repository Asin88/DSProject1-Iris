# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 04:12:16 2020

@author: AAKRITI
"""

#Project 1: Iris Dataset - Classification Problem 
#Script to clean data

#Encode categorical variable
def f_categoryEncoder():
    df_iris["species"] = df_iris["species"].astype('category')
    df_iris["species"] = df_iris["species"].cat.codes #Replace column data with numerical codes

print('Clean_data script started')
from data.make_dataset import df_iris
f_categoryEncoder()
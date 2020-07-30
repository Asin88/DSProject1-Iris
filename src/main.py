# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 04:20:10 2020

@author: AAKRITI
"""

#Project 1: Iris Dataset - Classification Problem 
#Identify the type of Iris flower from flower characteristics: length and width of sepal and petals.

#Main script

#Import modules
# =============================================================================
import pandas as pd 

if __name__ == '__main__':
        
    print('Main function')
    
    #Get raw dataframe
    import data.make_dataset
  
    #Get tidy dataframe
    from data.clean_data import df_iris
    print(df_iris.head())
    df_iris.to_csv('E:/Data Science Projects/1. Iris Dataset - Classification/data/processed/iris_clean.csv', encoding='utf-8')
    
    #Handle missing data 
    print(df_iris.describe())
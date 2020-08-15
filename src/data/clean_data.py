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

#Handle missing data
def f_missingData():
    df_iris.fillna(value={'sepal_length':df_iris.sepal_length.mean(), 'sepal_width':df_iris.sepal_width.mean(), 'petal_length':df_iris.petal_length.mean(), 'petal_width':df_iris.petal_width.mean(), 'species':round(df_iris.species.mean())})
    
 #Main function   
print('Clean_data script started')
from data.make_dataset import df_iris

#Encode Categories as 0/1
f_categoryEncoder()

#Handle missing data
f_missingData()

#Saving clean data to new csv file
df_iris.to_csv('E:/Data Science Projects/1. Iris Dataset - Classification/data/processed/iris_clean.csv', encoding='utf-8')

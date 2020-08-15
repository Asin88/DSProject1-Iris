# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 04:11:24 2020

@author: AAKRITI
"""

#Script to perform factor analysis


        
#Function to identify number of factors
def f_noOfFactors(scree_image_name):
    # Create factor analysis object and perform factor analysis
    factor = FactorAnalysis(n_components=len(df_indep.columns), random_state=101).fit(df_indep)
    eigen_values = pd.DataFrame(factor.components_,columns=df_indep.columns)
    return eigen_values
    
#Function to perform factor analysis
def f_factorAnalysis(num_factors):
    factor = FactorAnalysis(n_components=num_factors, random_state=101).fit(df_indep)
    covariances = factor.get_covariance()
    df_iris_scores = pd.DataFrame(factor.transform(df_indep))
    print(df_iris_scores.columns)
    df_iris_scores.rename(columns={0:'Factor1',1:'Factor2'},inplace=True)
    #Saving scores to new csv file
    df_iris_scores.to_csv('E:/Data Science Projects/1. Iris Dataset - Classification/data/processed/iris_scores.csv', encoding='utf-8')
    return(covariances, df_iris_scores)
    
#Main function
print('factor_analysis script started')

#IMport modules
import pandas as pd
from statsmodels.stats.power import TTestIndPower
from sklearn.decomposition import FactorAnalysis

filepath = 'E:/Data Science Projects/1. Iris Dataset - Classification/data/processed/iris_scaled.csv'
df_iris_scaled = pd.read_csv(filepath, header='infer', index_col=None)

#Identify number of factors
df_indep = df_iris_scaled[[i for i in list(df_iris_scaled.columns) if i != 'species']]
scree_image_name = 'E:/Data Science Projects/1. Iris Dataset - Classification/reports/figures/Scree_Plot.png'
eigen_values = f_noOfFactors(scree_image_name)
print('Number of factors considered: ')
num_factors = int(input())
#Perform factor analysis
covariances, df_iris_scores = f_factorAnalysis(num_factors)
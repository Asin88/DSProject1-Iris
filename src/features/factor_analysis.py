# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 04:11:24 2020

@author: AAKRITI
"""

#Script to perform factor analysis

#Import modules
import pandas as pd
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
#from sklearn.decomposition import FactorAnalysis
import os
import matplotlib.pyplot as plt
import numpy as np

#Define functions

#Function to get absolute file path
def f_getFilePath(rel_path):
    script_path = os.path.abspath(__file__) # i.e. /path/to/dir/foobar.py
    script_dir1 = os.path.split(script_path)[0] #i.e. /path/to/dir/
    script_dir2 = os.path.split(script_dir1)[0] #i.e. /path/to/dir/
    cwd_dir = os.path.split(script_dir2)[0] #i.e. /path/to/
    abs_file_path = os.path.join(cwd_dir, rel_path)
    return abs_file_path

#Function to identify number of factors
def f_noOfFactors(scree_image_name):
    # Create factor analysis object and perform factor analysis
    fa = FactorAnalyzer(rotation='varimax')
    fafit = fa.fit(df_indep)
    eigen_values = pd.DataFrame(fa.get_eigenvalues(), columns = df_indep.columns)
    print(eigen_values)
    
    #using module from sklearn.decomposition
#    factor = FactorAnalysis(n_components=len(df_indep.columns), random_state=101).fit(df_indep)
#    eigen_values = pd.DataFrame(factor.components_,columns=df_indep.columns)
    
    #Scree Plot
    plt.ylabel('Eigenvalues')
    plt.xlabel('# of Features')
    plt.title('Scree Plot')
    plt.style.context('seaborn-whitegrid')
    plt.axhline(y=1, color='r', linestyle='--')
    plt.plot(fa.get_eigenvalues())
    plt.show()
    abs_file_path = f_getFilePath(scree_image_name)
    plt.savefig(abs_file_path)
    
#Function to perform factor analysis
def f_factorAnalysis(num_factors):
    abs_file_path = f_getFilePath('reports\\factor_analysis.txt')
    fafile = open(abs_file_path, 'w')
    print('Factor Analysis Report\n\n', file = fafile)
    #Test assumptions
    #Bartlett's test
    statistic, p_value = calculate_bartlett_sphericity(df_iris_scaled)
    print('Bartlett’s test of sphericity:\nChi-square value = ', round(statistic, 3), ', p-value = ',round(p_value, 3), file = fafile)
    if p_value <= 0.05:
        print('Statistically significant. Dataset is not an identity matrix.\n',file = fafile)
    else:
        print('**Test is Insignificant. Dataset cannot be used for factor analysis**\n', file = fafile)
        exit()
    #KMO test
    kmo_peritem, kmo_overall = calculate_kmo(df_iris_scaled)
    print('Kaiser-Meyer-Olkin (KMO) test: \nKMO = ',round(kmo_overall, 3), file = fafile)
    if kmo_overall >= 0.60:
        print('KMO is adequate. Data is suitable for factor analysis.\n',file = fafile)
    else:
        print('**KMO is inadequate. Dataset cannot be used for factor analysis**\n', file = fafile)
        exit()
    #Get factors
    factor = FactorAnalyzer(n_factors = num_factors,rotation='varimax')
    factorfit = factor.fit(df_indep)
    eigen_values = pd.DataFrame(factor.get_eigenvalues()).round(decimals = 2)
    loadings = pd.DataFrame(factor.loadings_, columns = ['Factor 1', 'Factor 2']).round(decimals = 2)
    communalities = pd.DataFrame(np.round(factor.get_communalities(), 2), columns = ['Communalities']).round(decimals = 2)
    covariances = pd.DataFrame(factor.get_factor_variance(), columns = ['Factor 1', 'Factor 2']).round(decimals = 2)
    df_iris_scores = pd.DataFrame(factor.transform(df_indep))
    #using module from sklearn.decomposition
#    factor = FactorAnalysis(n_components=num_factors, random_state=101).fit(df_indep)
#    covariances = factor.get_covariance()
#    df_iris_scores = pd.DataFrame(factor.transform(df_indep))
    df_iris_scores.rename(columns={0:'Factor1',1:'Factor2'},inplace=True)
    
    #Save factor analysis report
    print('Configuration\n', file = fafile)
    print(factorfit, '\n', file = fafile)
    print('Eigen Values', file = fafile)
    print(eigen_values, '\n', file = fafile)
    print('Factor Loadings', file = fafile)
    print(loadings, '\n', file = fafile)
    print(communalities, '\n', file = fafile)
    print("Variances \n1. Sum of squared loadings (variance) \n2. Proportional variance \n3. Cumulative variance)", file = fafile)
    print(covariances, file = fafile)
    fafile.close()
    return df_iris_scores
    
    
#Main function
print('factor_analysis script started')

abs_file_path = f_getFilePath("data\\interim\\iris_scaled.csv")
df_iris_scaled = pd.read_csv(abs_file_path, header='infer', index_col=None)

#Identify number of factors
df_indep = df_iris_scaled[[i for i in list(df_iris_scaled.columns) if i != 'species']]
eigen_values = f_noOfFactors('reports\\figures\\Scree_Plot.png')
print('Number of factors considered: ')
num_factors = int(input())

#Perform factor analysis
df_iris_scores = f_factorAnalysis(num_factors)

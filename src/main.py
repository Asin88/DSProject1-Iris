# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 04:20:10 2020

@author: AAKRITI
"""

#Project 1: Iris Dataset - Classification Problem 
#Identify the type of Iris flower from flower characteristics: length and width of sepal and petals.

# =============================================================================
# #Import modules
# =============================================================================
from datetime import datetime
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sklearn.model_selection as moses
from statsmodels.stats import power as pwr
import statsmodels.discrete.discrete_model as dm
import math 

# =============================================================================
# #Define Functions
# =============================================================================
#Function to test power of dataset
def f_powerTest():
    #parameters for power analysis
    effect = 0.8
    alpha = 0.05
    power = 0.8
    #perform power analysis
    analysis = pwr.TTestIndPower()
    result = analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)
    print('Minimum Sample Size: %.3f' % result, file = outfile)
    if len(df_iris) >= result:
        print('Sample size is sufficient for effect size of 0.8 and power of 0.8.', file = outfile)
    else:
        printf('**LOW POWER OF DATASET: SAMPLE SIZE TOO SMALL**', file = outfile)
        
#------------------------------------------------------------------------------
        
#Function to build collinearity matrix and heatmap
def f_correlation(df_name,corr_csv_name, corr_image_name):
    print('\nCorrelation Matrix',file = outfile)
    correlation = df_name.corr(method='pearson')
    #Saving correlation matrix to new csv file
    correlation.to_csv(corr_csv_name, encoding='utf-8')
    print(correlation,file = outfile)
    corr_matrix = plt.matshow(correlation)
    plt.yticks(range(len(correlation.columns)), correlation.columns)
    plt.savefig(corr_image_name)
    print('\nCorrelation Heatmap image file saved',file = outfile)
    plt.show()
    return correlation

#------------------------------------------------------------------------------
    
def f_scatterplot(df_name,scplt_image_name):
    scatterplot = pd.plotting.scatter_matrix(df_name,alpha=0.2,figsize=(6,6),diagonal='hist')
    [s.xaxis.label.set_rotation(45) for s in scatterplot.reshape(-1)]
    [s.yaxis.label.set_rotation(0) for s in scatterplot.reshape(-1)]
    #May need to offset label when rotating to prevent overlap of figure
    [s.get_yaxis().set_label_coords(-0.5,0.5) for s in scatterplot.reshape(-1)]
    [s.set_yticks(()) for s in scatterplot.reshape(-1)]    
    plt.savefig(scplt_image_name)
    print('\nScatterplot image file saved',file=outfile)
    plt.show()
    
#------------------------------------------------------------------------------
        
#Function to test factorability of data
def f_testFactorability(correlation):
    #Bartlett's test
#    chi_square_value,p_value = ss.bartlett(df_iris)
#    print('Bartlett’s test of sphericity:\nChi-square value = ', chi_square_value, ', p-value = ',p_value, file = outfile)
#    if p_value <= 0.05:
#        print('Statistically significant. Dataset is not an identity matrix.',file = outfile)
#    else:
#        print('**Test is Insignificant. Dataset cannot be used for factor analysis**', file = outfile)
#        exit()
    
    #Kaiser-Meyer-Olkin test
    corr_inv = np.linalg.inv(correlation)
    nrow_inv_corr, ncol_inv_corr = correlation.shape
    A = np.ones((nrow_inv_corr, ncol_inv_corr))
    for i in range(0,nrow_inv_corr,1):
        for j in range(i,ncol_inv_corr,1):
            A[i,j] =  -(corr_inv[i,j])/(math.sqrt(corr_inv[i,i] * corr_inv[j,j]))
            A[j,i] = A[i,j]
    corr = np.asarray(correlation)
    kmo_num = np.sum(np.square(corr))-np.sum(np.square(np.diagonal(corr)))
    kmo_denom = kmo_num + np.sum(np.square(A)) - np.sum(np.square(np.diagonal(A)))
    kmo_value = kmo_num/kmo_denom
    print('Kaiser-Meyer-Olkin (KMO) test: \nKMO = ',kmo_value, file = outfile)
    if kmo_value >= 0.60:
        print('KMO is adequate. Data is suitable for factor analysis.',file = outfile)
    else:
        print('**KMO is inadequate. Dataset cannot be used for factor analysis**', file = outfile)
        exit()
    
# =============================================================================
# #Main function
# =============================================================================
def main():
    
      #Initialize project    
       
      #Intialize Project
        print('Project: Iris Classification', file = outfile)
        print('Author: Aakriti Sinha', file = outfile)
        print('Last run on ', datetime.now(), file = outfile)
        
    #------------------------------------------------------------------------------
      #Raw Data    
      #Get raw dataframe
        from data.make_dataset import df_iris
            
      #Describe raw data
        print('\nRaw Dataset Snapshot', file = outfile)
        print(df_iris.head(),'\n', file = outfile)
        print('\nRaw Data Description', file = outfile)
        print(df_iris.describe(), '\n',file = outfile)
        print('List of categories in categorical variable',file = outfile)
        print(df_iris['species'].unique(),'\n',file = outfile)
    
    #------------------------------------------------------------------------------
        
      #Data Cleaning  
      #Get tidy dataframe
        from data.clean_data import df_iris, missing_message
        print(missing_message)
        
      #Describe clean data
        print('\n\nClean Dataset Snapshot', file = outfile)
        print(df_iris.head(),'\n', file = outfile)
        print('\nClean Data Description', file = outfile)
        data_desc = df_iris.describe()
        print(data_desc,'\n', file = outfile)
        data_desc.to_csv('E:/Data Science Projects/1. Iris Dataset - Classification/reports/iris_clean_description.csv', index = False, encoding='utf-8')
        print('List of categories in categorical variable',file = outfile)
        print(df_iris['species'].unique(),'\n',file = outfile)
        
    #------------------------------------------------------------------------------
        
      #Test power of dataset
      
        f_powerTest()
    
    #------------------------------------------------------------------------------
            
      #Feature Scaling
        print('\n\nFeature Scaling: Centering, Standardizing and Normalizing', file = outfile)
        from data.scale_data import df_iris
        #Describe scaled data
        print('\nScaled Dataset Snapshot', file = outfile)
        print(df_iris.head(),'\n', file = outfile)
        print('\nScaled Data Description', file = outfile)
        data_desc = df_iris.describe()
        print(data_desc,'\n', file = outfile)
        data_desc.to_csv('E:/Data Science Projects/1. Iris Dataset - Classification/reports/iris_scaled_description.csv', encoding='utf-8')
        print('List of categories in categorical variable',file = outfile)
        print(df_iris['species'].unique(),'\n',file = outfile)
        
    #------------------------------------------------------------------------------
        
      #Check Correlation
        corr_csv_name = 'E:/Data Science Projects/1. Iris Dataset - Classification/reports/correlation.csv'
        corr_image_name = 'E:/Data Science Projects/1. Iris Dataset - Classification/reports/figures/Correlation_Heatmap.png'
        correlation = f_correlation(df_iris,corr_csv_name,corr_image_name)
      #Scatterplot Matrix    
        scplt_image_name = 'E:/Data Science Projects/1. Iris Dataset - Classification/reports/figures/Scatterplot_Matrix.png'
        f_scatterplot(df_iris,scplt_image_name)    
        
        
        print('\n**MULTICOLLINEARITY FOUND**', file = outfile)
    
    #------------------------------------------------------------------------------
        
      #Factor Analysis
        print('\nFACTOR ANALYSIS\n', file = outfile)
       
      #Testing factorability
        f_testFactorability(correlation)
        from features.factor_analysis import eigen_values, covariances, df_iris_scores
        print('\nEigen values: \n',eigen_values.transpose(), file = outfile)
        print('\nFactor Covariance: \n', covariances, file = outfile)
        df_iris_scores['species'] = df_iris['species']
        df_iris_scores.to_csv('E:/Data Science Projects/1. Iris Dataset - Classification/data/processed/iris_scores.csv', index = False, encoding='utf-8')
      
      #Describe selected features
        print('\nSelected Features Snapshot', file = outfile)
        print(df_iris_scores.head(),'\n', file = outfile)
        print('\nSelected Features Description', file = outfile)
        data_desc = df_iris.describe()
        print(data_desc,'\n', file = outfile)
        data_desc.to_csv('E:/Data Science Projects/1. Iris Dataset - Classification/reports/iris_scored_description.csv', index = False, encoding='utf-8')
        print('List of categories in categorical variable',file = outfile)
        print(df_iris_scores['species'].unique(),'\n',file = outfile)
            
      #Check Correlation
        corr_csv_name = 'E:/Data Science Projects/1. Iris Dataset - Classification/reports/correlation_factors.csv'
        corr_image_name = 'E:/Data Science Projects/1. Iris Dataset - Classification/reports/figures/Correlation_Heatmap_Factors.png'
        correlation = f_correlation(df_iris_scores,corr_csv_name,corr_image_name)
      #Scatterplot Matrix    
        scplt_image_name = 'E:/Data Science Projects/1. Iris Dataset - Classification/reports/figures/Scatterplot_Matrix_Factors.png'
        f_scatterplot(df_iris_scores,scplt_image_name)    
        
    #------------------------------------------------------------------------------
        
      #Model Development
        outfile.close()
     
      #Train-Test Split
        train_x, test_x, train_y, test_y = moses.train_test_split(df_iris_scores.iloc[:,:-1], df_iris_scores.iloc[:,-1], train_size=0.7, test_size=0.3)
        train_x.to_csv('E:/Data Science Projects/1. Iris Dataset - Classification/data/processed/iris_train_x.csv', index = False, encoding='utf-8')
        train_y.to_csv('E:/Data Science Projects/1. Iris Dataset - Classification/data/processed/iris_train_y.csv', header = ['species'], index = False, encoding='utf-8')
        test_x.to_csv('E:/Data Science Projects/1. Iris Dataset - Classification/data/processed/iris_test_x.csv', index = False, encoding='utf-8')
        test_y.to_csv('E:/Data Science Projects/1. Iris Dataset - Classification/data/processed/iris_test_y.csv', header = ['species'],  index = False, encoding='utf-8')
        
        #Train model
        import models.train_model
         
        
      
    # =============================================================================
    #     #End of Main Function
    # =============================================================================
    

if __name__ == '__main__':
        
    print('Main function')
    #Output file
    outfile = open('E:/Data Science Projects/1. Iris Dataset - Classification/reports/outfile.txt','w')
    main()
    #Close output file
    outfile.close()
#------------------------------------------------------------------------------


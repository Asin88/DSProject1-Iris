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
#from sklearn import preprocessing
from scipy.stats import chi2
import sklearn.model_selection as moses
from statsmodels.stats import power as pwr
#import statsmodels.discrete.discrete_model as dm
import math 
import os
import pingouin

# =============================================================================
# #Define Functions
# =============================================================================

#Function to get file path
def f_getFilePath(rel_path):
    script_path = os.path.abspath(__file__) # i.e. /path/to/dir/foobar.py
    script_dir = os.path.split(script_path)[0] #i.e. /path/to/dir/
    cwd_dir = os.path.split(script_dir)[0] #i.e. /path/to/
    abs_file_path = os.path.join(cwd_dir, rel_path)
    return(abs_file_path)
    
#Function to test power of dataset
def f_powerTest(df_iris):
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
    
    #Get file path
    abs_file_path = f_getFilePath(corr_csv_name)
    
    print('\nCorrelation Matrix',file = outfile)
    correlation = df_name.corr(method='pearson')
    #Saving correlation matrix to new csv file
    correlation.to_csv(abs_file_path, encoding='utf-8')
    print(correlation,file = outfile)
    
    #Correlation heat map
    #Get file path
    abs_file_path = f_getFilePath(corr_image_name)
    
    corr_matrix = plt.matshow(correlation)
    plt.yticks(range(len(correlation.columns)), correlation.columns)
    plt.savefig(abs_file_path)
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
    abs_file_path = f_getFilePath(scplt_image_name)
    plt.savefig(abs_file_path)
    print('\nScatterplot image file saved',file=outfile)
    plt.show()
    
#------------------------------------------------------------------------------
        
#Function to test factorability of data
def f_testFactorability(df_iris, correlation):
    #Bartlett's test
    #not working
#    chi_square_value,p_value = ss.bartlett(df_iris)
    #might work
    n, p = df_iris.shape
    corr_det = np.linalg.det(correlation)
    statistic = -np.log(corr_det) * (n - 1 - (2 * p + 5) / 6)
    degrees_of_freedom = p * (p - 1) / 2
    p_value = chi2.pdf(statistic, degrees_of_freedom)
#    
    print('Bartlett’s test of sphericity:\nChi-square value = ', statistic, ', p-value = ',p_value, file = outfile)
    if p_value <= 0.05:
        print('Statistically significant. Dataset is not an identity matrix.',file = outfile)
    else:
        print('**Test is Insignificant. Dataset cannot be used for factor analysis**', file = outfile)
        exit()
    
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
        print('List of categories in categorical variable',file = outfile)
        cat_list = df_iris['species'].unique()
        print(cat_list,'\n',file = outfile)
        print('Distribution of categories',file = outfile)
        cat_dist = df_iris.groupby('species').count()
        print(cat_dist, file = outfile)
        
        #Save clean data description report
        abs_file_path = f_getFilePath("reports\\iris_clean_description.txt")
        cleandescfile = open(abs_file_path,'w')
        print('\nClean Data Description', file = cleandescfile)
        print(data_desc,'\n', file = cleandescfile)
        print('List of categories in categorical variable',file = cleandescfile)
        print(cat_list,'\n',file = cleandescfile)
        print('Distribution of categories',file = cleandescfile)
        print(cat_dist, file = cleandescfile)
        
        cleandescfile.close()
                
    #------------------------------------------------------------------------------
        
      #Test power of dataset
      
        f_powerTest(df_iris)
    
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
        print('List of categories in categorical variable',file = outfile)
        cat_list = df_iris['species'].unique()
        print(cat_list,'\n',file = outfile)
        print('Distribution of categories',file = outfile)
        cat_dist = df_iris.groupby('species').count()
        print(cat_dist, file = outfile)
        
        #Save scaled data description report
        abs_file_path = f_getFilePath("reports\\iris_scaled_description.txt")
        scaledescfile = open(abs_file_path,'w')
        print('\nClean Data Description', file = scaledescfile)
        print(data_desc,'\n', file = scaledescfile)
        print('List of categories in categorical variable',file = scaledescfile)
        print(cat_list,'\n',file = scaledescfile)
        print('Distribution of categories',file = scaledescfile)
        print(cat_dist, file = scaledescfile)
        
        scaledescfile.close()
        
        
    #------------------------------------------------------------------------------
        
      #Check Correlation
        corr_csv_name = 'reports\\correlation.csv'
        corr_image_name = 'reports\\figures\\Correlation_Heatmap.png'
        correlation = f_correlation(df_iris,corr_csv_name,corr_image_name)
      #Scatterplot Matrix    
        scplt_image_name = 'reports\\figures\\Scatterplot_Matrix.png'
        f_scatterplot(df_iris,scplt_image_name)    
        
        
        print('\n**MULTICOLLINEARITY FOUND**', file = outfile)
    
    #------------------------------------------------------------------------------
        
      #Factor Analysis
        print('\nFACTOR ANALYSIS\n', file = outfile)
       
      #Testing factorability
    #        f_testFactorability(df_iris, correlation)
        from features.factor_analysis import df_iris_scores
        df_iris_scores['species'] = df_iris['species']
        
        #Check Correlation
        corr_csv_name = 'reports\\correlation_factors.csv'
        corr_image_name = 'reports\\figures\\Correlation_Heatmap_Factors.png'
        correlation = f_correlation(df_iris_scores,corr_csv_name,corr_image_name)
        #Scatterplot Matrix    
        scplt_image_name = 'reports\\figures\\Scatterplot_Matrix_Factors.png'
        f_scatterplot(df_iris_scores,scplt_image_name)    
        
        print('\n**Factor 2 has low correlation with Species. So dropping Factor 2**\n', file = outfile)
        df_iris_scores.drop('Factor2', axis = 1)
        
        #Save selected feature scores
        abs_file_path = f_getFilePath("data\\processed\\iris_scores.csv")
        df_iris_scores.to_csv(abs_file_path, index = False, encoding='utf-8')
        
        #Describe selected features
        print('\nSelected Features Snapshot', file = outfile)
        print(df_iris_scores.head(),'\n', file = outfile)
        print('\nSelected Features Description', file = outfile)
        data_desc = df_iris_scores.describe()
        print(data_desc, file = outfile)
        print('List of categories in categorical variable',file = outfile)
        cat_list = df_iris['species'].unique()
        print(cat_list,'\n',file = outfile)
        print('Distribution of categories',file = outfile)
        cat_dist = df_iris.groupby('species').count()
        print(cat_dist, file = outfile)
        
        #Save selected factors description report
        abs_file_path = f_getFilePath('reports\\iris_factors_description.txt')
        fadescfile = open(abs_file_path, 'w')
        print(data_desc, file = fadescfile)
        print('\nList of categories in categorical variable\n',cat_list,file = fadescfile)
        print('\nDistribution of categories\n', cat_dist,file = fadescfile)
        print('\nCronbach Alpha: ', pingouin.cronbach_alpha(df_iris_scores), file = fadescfile)
        fadescfile.close()
              
    #------------------------------------------------------------------------------
        
      #Model Development
        
      #Train-Test Split
        print(df_iris_scores.iloc[:,:-1].shape)
        print(df_iris_scores.iloc[:,-1].shape)
        train_x, test_x, train_y, test_y = moses.train_test_split(df_iris_scores.iloc[:,:-1], df_iris_scores.iloc[:,-1], train_size=0.7, test_size=0.3, random_state = 42, stratify = cat_list)
        abs_file_path = f_getFilePath("data\\processed\\iris_train_x.csv")
        train_x.to_csv(abs_file_path, index = False, encoding='utf-8')
        abs_file_path = f_getFilePath("data\\processed\\iris_train_y.csv")
        train_y.to_csv(abs_file_path, header = ['species'], index = False, encoding='utf-8')
        abs_file_path = f_getFilePath("data\\processed\\iris_test_x.csv")
        test_x.to_csv(abs_file_path, index = False, encoding='utf-8')
        abs_file_path = f_getFilePath("data\\processed\\iris_test_y.csv")
        test_y.to_csv(abs_file_path, header = ['species'],  index = False, encoding='utf-8')
        
        #Train model
        print('\n\nMultinomial Logistic Model\n', file = outfile)
        import models.train_model
             
            
          
        # =============================================================================
        #     #End of Main Function
        # =============================================================================


if __name__ == '__main__':
        
    print('Main function')
    #Output file
    abs_file_path = f_getFilePath("reports\\outfile.txt")
    outfile = open(abs_file_path,'w')
    #Call main function
    main()
    #Close output file
    outfile.close()
#------------------------------------------------------------------------------


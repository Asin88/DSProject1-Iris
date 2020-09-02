# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 22:58:04 2020

@author: AAKRITI
"""

#Script to perform regression

#Import modules
import pandas as pd
import statsmodels.discrete.discrete_model as dm
from statsmodels.tools.tools import add_constant
from sklearn import linear_model as lm
from sklearn import metrics
from sklearn import preprocessing 
import scikitplot as skplt
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import stats
import numpy as np
from datetime import datetime


#Define Functions

#Function to get absolute file path
def f_getFilePath(rel_path):
    script_path = os.path.abspath(__file__) # i.e. /path/to/dir/foobar.py
    script_dir1 = os.path.split(script_path)[0] #i.e. /path/to/dir/
    script_dir2 = os.path.split(script_dir1)[0] #i.e. /path/to/dir/
    cwd_dir = os.path.split(script_dir2)[0] #i.e. /path/to/
    abs_file_path = os.path.join(cwd_dir, rel_path)
    return(abs_file_path)

#Function to calculte Mean Square Error
def f_calcMSE(train_y, train_pred):
    
    difference = train_y.apply(lambda s: s - train_pred)
    sq_diff = difference.mul(difference)
    summation = sq_diff.sum(axis = 0)
    MSE = summation/len(train_y)
    MSE = np.array(MSE)
    return(MSE)
    
#Function to print summary of multinomial logistic regression model (scikit)
def f_printSummary(train_y, train_pred):
    
    #Summary file header
    print('Multinomial Logistic Regression Model Summary', file = sumfile)
    print('Author: Aakriti Sinha', file = sumfile)
    print('Project: Iris Classification', file = sumfile)
    print('Last run on: ', datetime.now(), '\n\n', file = sumfile)
    
    intercepts = reg_func.intercept_
    coeffs = reg_func.coef_
    print(intercepts)
    print(coeffs)
    
    for i in range(len(intercepts)):
        params = np.empty(train_y.nunique())
        params = np.append(reg_func.intercept_[i],reg_func.coef_[i])
        newX = pd.DataFrame({"Constant":np.ones(len(train_x))}).join(pd.DataFrame(train_x.reset_index(drop=True)))
        MSE = f_calcMSE(train_y, train_pred)
        var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
        sd_b = np.sqrt(var_b)
        ts_b = params/ sd_b
        p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX.iloc[:1,:])))) for i in ts_b]
        
        sd_b = np.round(sd_b,3)
        ts_b = np.round(ts_b,3)
        p_values = np.round(p_values,3)
        params = np.round(params,4)
        summary = pd.DataFrame()
        summary["Coefficients"],summary["Standard Errors"],summary["t values"],summary["Probabilities"] = [params,sd_b,ts_b,p_values]
        print('\n',summary, '\n\n', file = sumfile)
        
    
    
#Function to generate classification report
def f_classificationReport(train_y, train_pred):
#    #Binarize labels
    lb = preprocessing.LabelBinarizer()
    train_y = lb.fit_transform(train_y)
    train_y = pd.DataFrame(train_y, columns = ['Species 0','Species 1','Species 2'])
#    train_pred = lb.fit_transform(train_pred)
#    print('train_y: \n',train_y.head())
    train_pred = train_pred.applymap(int)
     #Confusion Matrices
    confusionmatrix = metrics.multilabel_confusion_matrix(train_y,train_pred)
    print('\nConfusion Matrices: \n', confusionmatrix,file= sumfile)
    
    #Find precision, recall, f score, support and accuracy
    class_report = metrics.classification_report(train_y,train_pred)
    #Print to summary file
    print('Saving detailed classification report to folder')
    print(class_report, file = sumfile)
    print('Accuracy: ', metrics.accuracy_score(train_y,train_pred), file = sumfile)
    

def f_rocAUC(train_y, train_pred):
    #Binarize labels
    lb = preprocessing.LabelBinarizer()
    train_y = lb.fit_transform(train_y)
#    train_pred = lb.fit_transform(train_pred)
    
    #Adding only unique labels of predited Y to remove warning of ill-defined precision and fscore.
    #There are some labels in train_y, which dont appear in train_pred and hence it is ill-defined
    #Selecting non-zero columns of train_pred
    m2 = (train_pred != 0).any()
    a = m2.index[m2]
    #Find AUC of ROC
    print('\nROC AUC: ', metrics.roc_auc_score(train_y, train_pred, labels = a, multi_class = 'ovr'), file = outfile)
    print('\nROC AUC: ', metrics.roc_auc_score(train_y, train_pred, labels = a, multi_class = 'ovr'), file = sumfile)
    
    #Plot ROC curves for the multilabel problem
    
    # Compute ROC curve and ROC area for each class
    n_classes = train_y.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(train_y[:, i], train_pred.iloc[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
#    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(train_y, train_pred)
#    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(train_y, train_pred)
#    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
#    plt.plot(fpr["micro"], tpr["micro"],
#             label='micro-average ROC curve (area = {0:0.2f})'
#                   ''.format(roc_auc["micro"]),
#             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(f_getFilePath('reports\\figures\\ROC_Curve.png'))
    print('\nROC Curve plot saved')
    plt.show()
    
# =============================================================================
#     skplt.metrics.plot_roc_curve(train_y, train_pred)
#     plt.savefig('E:/Data Science Projects/1. Iris Dataset - Classification/reports/figures/roc_curve.csv')
#     plt.show()
# 
# =============================================================================
    

    
#Main function
print('train_model script started')

#Output file
abs_file_path = f_getFilePath('data\\processed\\iris_train_x.csv')
train_x = pd.read_csv(filepath, header='infer', index_col=None)
print(train_x.head())
abs_file_path = f_getFilePath('data\\processed\\iris_train_y.csv')
train_y = pd.read_csv(filepath, header='infer', index_col=None)
print(train_y.head())

#Build logit function
reg_func = dm.MNLogit(train_y, add_constant(train_x))
reg_model = reg_func.fit(method = 'powell', maxiter = 200)
##Build logistic function
#reg_func = lm.LogisticRegression(solver='newton-cg', multi_class='multinomial')
#reg_model = reg_func.fit(train_x, train_y.values.ravel())

#Model Summary
stats.chisqprob = lambda chisq, reg_model: stats.chi2.sf(chisq, reg_model) #fix for summary()
#abs_file_path = f_getFilePath('reports\\reg_model_summary.txt')
sumfile = open(f_getFilePath('reports\\reg_model_summary.txt'),'w')
print(reg_model.summary2(), file = sumfile)

#Metrics
train_pred = reg_model.predict() #Predicted values of y
train_pred = pd.DataFrame(train_pred,columns=['LogitofSpecies0','LogitofSpecies1','LogitofSpecies2'])
print(train_pred.head())

##Model Summary
#sumfile = open('E:/Data Science Projects/1. Iris Dataset - Classification/reports/reg_model_summary.txt','w')
#f_printSummary(train_y, train_pred)

##Classification Report
classfile = open(f_getFilePath('reports\\classification_report.txt'),'w')
f_classificationReport(train_y, train_pred)
#ROC Curve
f_rocAUC(train_y, train_pred)

sumfile.close()
classfile.close()


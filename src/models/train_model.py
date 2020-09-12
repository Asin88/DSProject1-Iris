# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 22:58:04 2020

@author: AAKRITI

Script to train models and select the best model
"""

#Import modules
import os
import pandas as pd
import statsmodels.discrete.discrete_model as dm
from statsmodels.tools.tools import add_constant
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn import preprocessing 
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import stats
import numpy as np

#Define Functions

#Function to get absolute file path
def f_getFilePath(rel_path):
    script_path = os.path.abspath(__file__) # i.e. /path/to/dir/foobar.py
    script_dir1 = os.path.split(script_path)[0] #i.e. /path/to/dir/
    script_dir2 = os.path.split(script_dir1)[0] #i.e. /path/to/dir/
    cwd_dir = os.path.split(script_dir2)[0] #i.e. /path/to/
    abs_file_path = os.path.join(cwd_dir, rel_path)
    return(abs_file_path)
    
#Function to evaluate all training models
def f_modelEvaluation(model_name, reg_model, train_x, train_y):
    
    if model_name != 'MNLogit':
        
        # define the evaluation method
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        # evaluate the model on the dataset
        n_scores_acc = cross_val_score(reg_model, train_x, train_y, scoring='accuracy', cv=cv, n_jobs=-1)
        n_scores_roc = cross_val_score(reg_model, train_x, train_y, scoring='roc_auc_ovr', cv=cv, n_jobs=-1)
        # report performance
        print(model_name, '\t', round(mean(n_scores_acc), 3), '\t\t',round(mean(n_scores_roc),3), file = modelevalfile)
    else:    
        lb = preprocessing.LabelBinarizer()
        train_y = lb.fit_transform(train_y)
        train_y = pd.DataFrame(train_y, columns = ['Species 0','Species 1','Species 2'])
        
        if model_name == 'MNLogit':
            train_pred = reg_model.predict()   #Predicted values of y
            train_pred = pd.DataFrame(train_pred,columns=['LogitofSpecies0','LogitofSpecies1','LogitofSpecies2'])
            
        else:
            train_pred = reg_model.predict(train_x) #Predicted values of y
            train_pred = lb.fit_transform(train_pred)
        
        train_pred = pd.DataFrame(train_pred,columns=['LogitofSpecies0','LogitofSpecies1','LogitofSpecies2'])
        train_pred = train_pred.applymap(int)        
      
        accuracy = metrics.accuracy_score(train_y,train_pred)
        #Adding only unique labels of predited Y to remove warning of ill-defined precision and fscore.
        #There are some labels in train_y, which dont appear in train_pred and hence it is ill-defined
        #Selecting non-zero columns of train_pred
        m2 = (train_pred != 0).any()
        a = m2.index[m2]
        #Find AUC of ROC
        rocAUC = metrics.roc_auc_score(train_y, train_pred, labels = a, multi_class = 'ovr')
        
        print(model_name,'\t',round(accuracy,3), '\t\t', round(rocAUC,3), file = modelevalfile)
            
#Function to print summary of final model
def f_modelSummary(final_model_name, final_model):
    print('Final Model: ', final_model_name, '\n\n', file = sumfile)
    if final_model_name == 'MNLogit':
        stats.chisqprob = lambda chisq, final_model: stats.chi2.sf(chisq, final_model) #fix for summary()
        print(final_model.summary2(), file = sumfile)
        ##Model Summary for scikit
        #sumfile = open('E:/Data Science Projects/1. Iris Dataset - Classification/reports/reg_model_summary.txt','w')
        #f_printSummary(train_y, train_pred)
    elif final_model_name == 'Unpruned CART' or final_model_name == 'Pruned CART' or final_model_name == 'Random Forest' or final_model_name == 'Pruned Random Forest' or final_model_name == 'Gradient Boosting':
        # Get numerical feature importances
        importances = list(final_model.feature_importances_)
        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(list(train_x.columns), importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        # Print out the feature and importances 
        [print('Variable: {:20} Importance: {}'.format(*pair), file = sumfile) for pair in feature_importances];
    else:
        print('\n')
        
#Function to generate classification report
def f_classificationReport(train_y, train_pred):
    
    #Confusion Matrices
    confusionmatrix = metrics.multilabel_confusion_matrix(train_y,train_pred)
    
    #Find precision, recall, f score, support and accuracy
    class_report = metrics.classification_report(train_y,train_pred)
    accuracy = metrics.accuracy_score(train_y,train_pred)
    
    #Print to summary file
    print('Saving detailed classification report in summary file')
    print('\n\nCLassification Report',file = sumfile)
    print('\nConfusion Matrices: \n', confusionmatrix,file= sumfile)
    print(class_report, file = sumfile)
    print('Accuracy: ',accuracy, file = sumfile)
    
#Funtion to plot ROC        
def f_rocAUC(train_y, train_pred):
    
    #Adding only unique labels of predited Y to remove warning of ill-defined precision and fscore.
    #There are some labels in train_y, which dont appear in train_pred and hence it is ill-defined
    #Selecting non-zero columns of train_pred
    m2 = (train_pred != 0).any()
    a = m2.index[m2]
    #Find AUC of ROC
    rocAUC = metrics.roc_auc_score(train_y, train_pred, labels = a, multi_class = 'ovr')
    print('\nROC AUC: ',rocAUC , file = sumfile)
    
    #Plot ROC curves for the multilabel problem
    
    # Compute ROC curve and ROC area for each class
    n_classes = train_y.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(train_y.iloc[:, i], train_pred.iloc[:, i])
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
    
#Function to calculate model performance metrics
def f_calcMetrics(final_model_name, train_y):
    #Binarize labels
    lb = preprocessing.LabelBinarizer()
    train_y = lb.fit_transform(train_y)
    train_y = pd.DataFrame(train_y, columns = ['Species 0','Species 1','Species 2'])
    
    if final_model_name == 'MNLogit':
        train_pred = final_model.predict()   #Predicted values of y
        train_pred = pd.DataFrame(train_pred,columns=['LogitofSpecies0','LogitofSpecies1','LogitofSpecies2'])
        
    else:
        train_pred = final_model.predict(train_x) #Predicted values of y
        train_pred = lb.fit_transform(train_pred)
    
    train_pred = pd.DataFrame(train_pred,columns=['LogitofSpecies0','LogitofSpecies1','LogitofSpecies2'])
    train_pred = train_pred.applymap(int)  
    
    f_classificationReport(train_y, train_pred)
    f_rocAUC(train_y, train_pred)
        
    
#Main function
print('train_model script started')

#Training data files
abs_file_path = f_getFilePath('data\\processed\\iris_train_x.csv')
train_x = pd.read_csv(abs_file_path, header='infer', index_col=None)
abs_file_path = f_getFilePath('data\\processed\\iris_train_y.csv')
train_y = pd.read_csv(abs_file_path, header='infer', index_col=None)

#Build training models
reg_model =[]
model_name=[]

#Model 1: Multinomial Logit function
model_name.append('MNLogit')
mnlr_func = dm.MNLogit(train_y, add_constant(train_x))
mnlr_model = mnlr_func.fit(method = 'powell', maxiter = 200)
##Build logistic function (scikit) --Not Working
#reg_func = lm.LogisticRegression(solver='newton-cg', multi_class='multinomial')
#reg_model = reg_func.fit(train_x, train_y.values.ravel())
reg_model.append(mnlr_model)

#Model 2: K-Nearest Neighbours
#Model 2a: K = 3
model_name.append('KNN 3')
knn3_func = KNeighborsClassifier(n_neighbors=3)
knn3_model = knn3_func.fit(train_x, train_y)
reg_model.append(knn3_model)

#Model 2b: K = 5
model_name.append('KNN 5')
knn5_func = KNeighborsClassifier(n_neighbors=5)
knn5_model = knn5_func.fit(train_x, train_y)
reg_model.append(knn5_model)

#Model 2c: K = 7
model_name.append('KNN 7')
knn7_func = KNeighborsClassifier(n_neighbors=7)
knn7_model = knn7_func.fit(train_x, train_y)
reg_model.append(knn7_model)

#Model 3: Gaussian Naive Bayesian
model_name.append('Gaussian NB')
gnb_func = GaussianNB() 
gnb_model = gnb_func.fit(train_x, train_y)
reg_model.append(gnb_model)

#Model 4: Decision Trees / Classification And Regression Trees (CART)
#Model 4a: Unpruned CART
model_name.append('Unpruned CART')
unpcart_func = DecisionTreeClassifier()
unpcart_model = unpcart_func.fit(train_x, train_y)
reg_model.append(unpcart_model)

#Model 4b: Pruned CART
model_name.append('Pruned CART')
pcart_func = DecisionTreeClassifier(criterion="entropy",max_depth = 3)
pcart_model = pcart_func.fit(train_x, train_y)
reg_model.append(pcart_model)

#Model 4c: Random Forest Classifier
model_name.append('Random Forest')
rfc_func = RandomForestClassifier()
rfc_model = rfc_func.fit(train_x, train_y)
reg_model.append(rfc_model)

#Model 4d: Pruned Random Forest Classifier
model_name.append('Pruned Random Forest')
prfc_func = RandomForestClassifier(n_estimators=10, max_depth=3)
prfc_model = prfc_func.fit(train_x, train_y)
reg_model.append(prfc_model)

#Model 4: Gradient Boosting Classifier
model_name.append('Gradient Boosting')
gbc_func = GradientBoostingClassifier()
gbc_model = gbc_func.fit(train_x, train_y)
reg_model.append(gbc_model)

#Model Evaluation
modelevalfile = open(f_getFilePath('reports\\model_comparison.txt'),'w')
print('Model Comparison\n\n',file = modelevalfile)
print('Model Name\tAccuracy\tROC AUC', file = modelevalfile)
for i in range(len(reg_model)):
    f_modelEvaluation(model_name[i], reg_model[i], train_x, train_y)
modelevalfile.close()

print('Select the best model out of: \n', model_name)
loopagain = 1
while (loopagain == 1):
    final_model_name = input()
    if final_model_name in model_name:
        idx = model_name.index(final_model_name)
        final_model = reg_model[idx]
        loopagain = 0
    else:
        print('Enter correct input')
        loopagain = 1

#Final model
sumfile = open(f_getFilePath('reports\\final_model_summary.txt'),'w')
f_modelSummary(final_model_name, final_model)
#Metrics
f_calcMetrics(final_model_name, train_y)

sumfile.close()



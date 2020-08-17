# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 22:58:04 2020

@author: AAKRITI
"""

#Script to perform multinomial logit 

#Import modules
import pandas as pd
import statsmodels.discrete.discrete_model as dm
from sklearn import metrics
#import scikitplot as skplt
import matplotlib.pyplot as plt
from scipy import stats


#Define Functions

#Function to generate classification report
def f_classificationReport():
    
    #Confusion Matrices
    print('\nConfusion Matrices: \n',metrics.multilabel_confusion_matrix(train_y,train_pred_round),file= outfile)
    confusionmatrix = reg_model.pred_table()
    print(confusionmatrix,file=outfile)
    print('Precision: ', metrics.precision_recall_fscore_support(train_y, train_pred_round, average = 'micro')[0], file = outfile)
    print('Recall: ', metrics.precision_recall_fscore_support(train_y, train_pred_round, average = 'micro')[1],file=outfile)
    print('f-score: ', metrics.precision_recall_fscore_support(train_y, train_pred_round, average = 'micro')[2], file = outfile)
    print('Support: ', metrics.precision_recall_fscore_support(train_y, train_pred_round, average = 'micro')[3], file = outfile)
    print('Accuracy: ', metrics.accuracy_score(train_y,train_pred_round))
    print('Saving detailed classification report to folder', file = outfile)
    class_report = metrics.classification_report(train_y,train_pred_round)
    class_report.to_csv('E:/Data Science Projects/1. Iris Dataset - Classification/reports/classification_report.csv', index = False, encoding='utf-8')

def f_rocAUC():
    print('\nROC AUC: ', metrics.roc_auc_score(train_y, train_pred, multi_class = 'ovr'), file = outfile)
   
# =============================================================================
#     #Plot ROC curves for the multilabel problem
#     
#     # First aggregate all false positive rates
#     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#     
#     # Then interpolate all ROC curves at this points
#     mean_tpr = np.zeros_like(all_fpr)
#     for i in range(n_classes):
#         mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#     
#     # Finally average it and compute AUC
#     mean_tpr /= n_classes
#     
#     fpr["macro"] = all_fpr
#     tpr["macro"] = mean_tpr
#     roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
#     
#     # Plot all ROC curves
#     plt.figure()
#     plt.plot(fpr["micro"], tpr["micro"],
#              label='micro-average ROC curve (area = {0:0.2f})'
#                    ''.format(roc_auc["micro"]),
#              color='deeppink', linestyle=':', linewidth=4)
#     
#     plt.plot(fpr["macro"], tpr["macro"],
#              label='macro-average ROC curve (area = {0:0.2f})'
#                    ''.format(roc_auc["macro"]),
#              color='navy', linestyle=':', linewidth=4)
#     
#     colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
#     for i, color in zip(range(n_classes), colors):
#         plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#                  label='ROC curve of class {0} (area = {1:0.2f})'
#                  ''.format(i, roc_auc[i]))
#     
#     plt.plot([0, 1], [0, 1], 'k--', lw=lw)
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Some extension of Receiver operating characteristic to multi-class')
#     plt.legend(loc="lower right")
#     plt.show()
# =============================================================================
    
    skplt.metrics.plot_roc_curve(train_y, train_pred)
    plt.savefig('E:/Data Science Projects/1. Iris Dataset - Classification/reports/figures/roc_curve.csv')
    plt.show()

#Main function
print('train_model script started')
#Output file
outfile = open('E:/Data Science Projects/1. Iris Dataset - Classification/reports/outfile.txt','a')
filepath = 'E:/Data Science Projects/1. Iris Dataset - Classification/data/processed/iris_train_x.csv'
train_x = pd.read_csv(filepath, header='infer', index_col=None, skiprows=[0])
#train_x = pd.DataFrame(train_x)
filepath = 'E:/Data Science Projects/1. Iris Dataset - Classification/data/processed/iris_train_y.csv'
train_y = pd.read_csv(filepath, header='infer', index_col=None, skiprows=[0])
train_y = pd.DataFrame(train_y)

#Build logit function
reg_func = dm.MNLogit(train_y, train_x)
reg_model = reg_func.fit()
stats.chisqprob = lambda chisq, reg_model: stats.chi2.sf(chisq, reg_model) #fix for sumarry()
sumfile = open('E:/Data Science Projects/1. Iris Dataset - Classification/reports/reg_model_summary.txt','w')
print(reg_model.summary(), file = sumfile)
sumfile.close()

#Metrics
train_pred = reg_model.predict(train_x) #Predicted values of y
print(train_pred.head())
train_pred = train_pred.apply(int)

##Classification Report
f_classificationReport()
##ROC Curve
#f_rocAUC()

##Predict Y
#y_logit = pd.DataFrame(result.predict(x_train))
#y_logit.columns = ['yLogit']
#print(y_logit.head(5))

outfile.close()


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.metrics as sklm
import imblearn.metrics as imbm

#Import any classifier clf, than apply the same procedures.
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

X, y = make_classification(n_samples=500, n_features=25,
                           n_clusters_per_class=1, n_informative=15,
                           random_state=RANDOM_STATE)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=66)


clf =RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
clf.fit(X_train , y_train)
clf_predict = clf.predict(X_test)


accuracy_score =  sklm.accuracy_score(y_test,clf_predict)
average_precision =  sklm.average_precision_score(y_test,clf_predict)
brier_score_loss =  sklm.brier_score_loss(y_test,clf_predict)
precision_score =  sklm.precision_score(y_test,clf_predict)
recall_score =  sklm.recall_score(y_test,clf_predict)
f1_score =  sklm.f1_score(y_test,clf_predict)
roc_auc_score =  sklm.roc_auc_score(y_test,clf_predict)
classification_report = sklm.classification_report(y_test,clf_predict)

print("Accuracy Score : ", accuracy_score)
print("Average Precision Score : ", average_precision)
print("brier_score_loss : ", brier_score_loss)
print("precision_score : ", precision_score)
print("recall_score : ", recall_score)
print("f1_score : ", f1_score)
print("roc_auc_score : ", roc_auc_score)
print("classification_report : \n", classification_report)










"""imb
metrics.classification_report_imbalanced(…)	Build a classification report based on metrics used with imbalanced dataset
metrics.sensitivity_specificity_support(…)	Compute sensitivity, specificity, and support for each class
metrics.sensitivity_score(y_true, y_pred[, …])	Compute the sensitivity
metrics.specificity_score(y_true, y_pred[, …])	Compute the specificity
metrics.geometric_mean_score(y_true, y_pred)	Compute the geometric mean.
metrics.make_index_balanced_accuracy([…])	Balance any scoring function using the index balanced accuracy
"""
"""sklearn
‘accuracy’	metrics.accuracy_score	 
‘balanced_accuracy’	metrics.balanced_accuracy_score	for binary targets
‘average_precision’	metrics.average_precision_score	 
‘brier_score_loss’	metrics.brier_score_loss	 
‘f1’	metrics.f1_score	for binary targets
‘f1_micro’	metrics.f1_score	micro-averaged
‘f1_macro’	metrics.f1_score	macro-averaged
‘f1_weighted’	metrics.f1_score	weighted average
‘f1_samples’	metrics.f1_score	by multilabel sample
‘neg_log_loss’	metrics.log_loss	requires predict_proba support
‘precision’ etc.	metrics.precision_score	suffixes apply as with ‘f1’
‘recall’ etc.	metrics.recall_score	suffixes apply as with ‘f1’
‘roc_auc’	metrics.roc_auc_score
"""


import pandas as pd
import numpy as np
features = pd.read_csv("heart.csv")
#print features.head(5)
#print features.describe()
#print features.shape
labels = np.array(features["target"])
features = features.drop("target", axis = 1)
feature_list = list(features.columns)
features = np.array(features)
from sklearn.model_selection import train_test_split
# implementing train-test-split
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state = 42)
features_oversampled , labels_oversampled = ros.fit_resample(features, labels)
features_oversampled_train , features_oversampled_test , labels_oversampled_train, labels_oversampled_test = train_test_split(
    features_oversampled, 
    labels_oversampled, 
    test_size = 0.33, 
    random_state = 66)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=66)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 200 ,
                            random_state = 42 , 
                            oob_score = True ,
                            class_weight=None, 
                            criterion='gini', 
                            max_depth=None,
                            max_features='log2', 
                            max_leaf_nodes=None,
                            min_impurity_decrease=0.0, 
                            min_impurity_split=None,
                            min_samples_leaf=1, 
                            min_samples_split=2,
                            min_weight_fraction_leaf=0.0)

rf.fit(features_train, labels_train)


rf_predict = rf.predict(features_test)


rf_oversamp = RandomForestClassifier(n_estimators = 200 ,
                            random_state = 42 , 
                            oob_score = True ,
                            class_weight=None, 
                            criterion='gini', 
                            max_depth=None,
                            max_features='log2', 
                            max_leaf_nodes=None,
                            min_impurity_decrease=0.0, 
                            min_impurity_split=None,
                            min_samples_leaf=1, 
                            min_samples_split=2,
                            min_weight_fraction_leaf=0.0)
rf_oversamp.fit(features_oversampled_train , labels_oversampled_train)
rf_oversamp_predict = rf_oversamp.predict(features_oversampled_test)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
rf_cv_score = cross_val_score(rf, features, labels, cv=10, scoring="roc_auc")

print("=== Confusion Matrix ===")
print(confusion_matrix(labels_test, rf_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(labels_test, rf_predict))
print('\n')
print("=== All AUC Scores ===")
print(rf_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rf_cv_score.mean())
print("oob_score:" , rf.oob_score_)
#print("oob_decision_function:" ,rf.oob_decision_function_)
#print(rf.estimators_)
#print("Feature Importances: ", rf.feature_importances_)
#rf_predict_proba = rf.predict_proba(features_test)
#print("Predict Class Probabilities", rf_predict_proba)
print("Score : " , rf.score( features_test,labels_test ,))

print("Oversampled Metrics")
print classification_report(labels_oversampled_test , rf_oversamp_predict)
print("OOB")
print rf_oversamp.oob_score_
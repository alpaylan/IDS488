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
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=66)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100 , random_state = 42 , oob_score = True)

rf.fit(features_train, labels_train)

rf_predict = rf.predict(features_test)

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
print("Feature importances:" ,rf.feature_importances_)
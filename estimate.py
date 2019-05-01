import pandas as pd
import utilities2 as ut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
datasetName = "mergedDataset.csv"

features = ut.loadCSV(datasetName, lm=True)
features = features.dropna()

labels = features["Label"]
features = features.drop("Label", axis = 1)

def errorCheck():
        featureNames = list(features)
        for i in  featureNames:
                col = list(features[i])
                maks = col[0]
                for j in range(1, len(col)):
                        if col[j] == "Infinity":
                                print i + " row " + str(j)
                        if col[j] > maks:
                                maks = col[j]
                print i + " : " + str(maks)

features_train , features_test , labels_train , labels_test = train_test_split(features,labels, test_size = 0.33,random_state = 42)

rf = RandomForestClassifier(n_estimators=100,oob_score=True,max_features="log2",random_state=42)

rf.fit(features_train, labels_train)
rf_pred = rf.predict(features_test)

importances = rf.feature_importances_
featureNames = list(features)

featureImportance = []

for i in range(0, len(importances)):
        featureImportance.append((featureNames[i], importances[i]))

featureImportance.sort(key = lambda i : i[1], reverse = True)
for i in featureImportance:
        print i[0] + ": " + str(100 * i[1])


print classification_report(labels_test, rf_pred)
print "OOB_Score :" , rf.oob_score_
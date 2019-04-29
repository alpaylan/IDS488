import pandas as pd
import utilities as ut
from sklearn.ensemble import RandomForestClassifier

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

rf = RandomForestClassifier(n_estimators=100)
rf.fit(features, labels)

importances = rf.feature_importances_
featureNames = list(features)

featureImportance = []

for i in range(0, len(importances)):
        featureImportance.append((featureNames[i], importances[i]))

featureImportance.sort(key = lambda i : i[1], reverse = True)
for i in featureImportance:
        print i[0] + ": " + str(100 * i[1])
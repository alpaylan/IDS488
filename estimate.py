import pandas as pd
import time
import utilities as ut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

datasetName = "mergedDataset.csv"
testSample = "testSample.csv"

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
print "Fitting RF"
fitStart = time.time()
rf.fit(features, labels)
fitEnd = time.time()
print "Fitted. Time passed: " + str(fitEnd - fitStart)

def printImportance():
        importances = rf.feature_importances_
        featureNames = list(features)

        featureImportance = []

        for i in range(0, len(importances)):
                featureImportance.append((featureNames[i], importances[i]))

        featureImportance.sort(key = lambda i : i[1], reverse = True)
        for i in featureImportance:
                print i[0] + ": " + str(100 * i[1])

def firstTest():
        testFeatures = ut.loadCSV(testSample, lm=True)
        testLabels = testFeatures["Label"]
        testFeatures = testFeatures.drop("Label", axis = 1)

        predictions = rf.predict(testFeatures)

        print "Accuracy = " + str(accuracy_score(testLabels, predictions))

def compareClasses():
        global labels
        global rf
        labelTypes = ut.labelTypes(labels)
        benignFeatures = ut.loadCSV("classCompare/BENIGN.csv", True)
        benignLabels = benignFeatures["Label"]
        benignFeatures = benignFeatures.drop("Label", axis=1)
        for i in labelTypes:
                if i == "BENIGN":
                        continue
                attackFeatures = ut.loadCSV("classCompare/" + i + ".csv", True)
                attackLabels = attackFeatures["Label"]
                attackFeatures = attackFeatures.drop("Label", axis=1)
                mergedFeatures = benignFeatures.append(attackFeatures)
                mergedLabels = benignLabels.append(attackLabels)
                print "Testing BENIGN vs " + i
                predictions = rf.predict(mergedFeatures)
                print "Accuracy = " + str(accuracy_score(mergedLabels, predictions))
                print ""

compareClasses()
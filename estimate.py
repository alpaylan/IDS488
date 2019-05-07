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

def fitRF():
        rf = RandomForestClassifier(n_estimators=100)
        print "Fitting RF"
        fitStart = time.time()
        rf.fit(features, labels)
        fitEnd = time.time()
        print "Fitted. Time passed: " + str(fitEnd - fitStart)
        return rf

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

rf = None

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

def compareBinary():
        global labels
        labelNames = ut.labelTypes(labels)
        benignTrain = ut.loadCSV("binaryCompare/BENIGN_train.csv", lm=True)
        benignTrainLabels = benignTrain["Label"]
        benignTrain = benignTrain.drop("Label", axis = 1)
        benignTest = ut.loadCSV("binaryCompare/BENIGN_test.csv", lm=True)
        benignReal = benignTest["Label"]
        benignTest = benignTest.drop("Label", axis = 1)
        for i in labelNames:
                if i == "BENIGN":
                        continue
                print ""
                print "--------------------------------------------------------"
                print "BENIGN vs " + i
                print "--------------------------------------------------------"
                attackTrain = ut.loadCSV("binaryCompare/" + i + "_train.csv", lm=True)
                attackLabels = attackTrain["Label"]
                attackTrain = attackTrain.drop("Label", axis = 1)
                forest = RandomForestClassifier(n_estimators=100)
                mergedTrainFeatures = benignTrain.append(attackTrain)
                mergedTrainLabels = benignTrainLabels.append(attackLabels)
                print "Fitting RF for BENIGN vs " + i
                fitStart = time.time()
                forest.fit(mergedTrainFeatures, mergedTrainLabels)
                fitEnd = time.time()
                print "Fitted. Time passed: " + str(fitEnd - fitStart)
                attackTest = ut.loadCSV("binaryCompare/" + i + "_test.csv", lm=True)
                attackReal = attackTest["Label"]
                attackTest = attackTest.drop("Label", axis = 1)
                mergedTestFeatures = benignTest.append(attackTest)
                mergedReal = benignReal.append(attackReal)
                predictions = forest.predict(mergedTestFeatures)
                print "Accuracy for BENIGN vs " + i + "= " + str(accuracy_score(predictions,mergedReal))
                print ""
                print "Variable Importances:"
                importances = forest.feature_importances_
                featureNames = list(attackTest)
                importantPairs = []
                for i in range(0, len(featureNames)):
                        importantPairs.append((featureNames[i], importances[i]))
                importantPairs.sort(key = lambda i : i[1], reverse = True)
                for i in importantPairs:
                        print i[0] + ": " + str(100*i[1]) + "%"
                print "--------------------------------------------------------"

compareBinary()
import imblearn.metrics as imbm
import imblearn.datasets as imbd
import imblearn.utils as imbu
import imblearn.ensemble as imbe
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import *
"""
RandomUnderSampler, InstanceHardnessThreshold,  \
CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours \
AllKNN, NearMiss, NeighbourhoodCleaningRule, OneSidedSelection, TomekLinks
"""
from copy import deepcopy


RANDOM_STATE = 42

rf = RandomForestClassifier( n_estimators=100,
                        oob_score=True,
                        max_features="log2",
                        random_state=RANDOM_STATE)

X_samp = {}
X_train = {}
X_test = {}
X = {
    "sample"    : X_samp,
    "train"     : X_train,
    "test"      : X_test
}
y_samp = {}
y_train = {}
y_test = {}
y = {
    "sample"    : y_samp,
    "train"     : y_train,
    "test"      : y_test
}
predicts = {
}
clfs = {
    "ROS"           :deepcopy(rf),
    "SMOTE"         :deepcopy(rf),
    "RUS"           :deepcopy(rf),
    "IHT"           :deepcopy(rf),
    "CNN"           :deepcopy(rf),
    "ENN"           :deepcopy(rf),
    "RENN"          :deepcopy(rf),
    "AKNN"          :deepcopy(rf),
    "NM"            :deepcopy(rf),
    "NCR"           :deepcopy(rf),
    "OSS"           :deepcopy(rf),
    "TL"            :deepcopy(rf),
    "imb"           :imbe.BalancedRandomForestClassifier(   n_estimators = 100,
                                                            oob_score = True,
                                                            max_features = "log2",
                                                            random_state = RANDOM_STATE)
}
samplers = {
    "ROS"           :RandomOverSampler(random_state = RANDOM_STATE),
    "SMOTE"         :SMOTE(random_state = RANDOM_STATE),
    "RUS"           :RandomUnderSampler(random_state = RANDOM_STATE),
    "IHT"           :InstanceHardnessThreshold(random_state = RANDOM_STATE),
    "CNN"           :CondensedNearestNeighbour(random_state = RANDOM_STATE),
    "ENN"           :EditedNearestNeighbours(random_state = RANDOM_STATE),
    "RENN"          :RepeatedEditedNearestNeighbours(random_state = RANDOM_STATE),
    "AKNN"          :AllKNN(random_state = RANDOM_STATE),
    "NM"            :NearMiss(random_state = RANDOM_STATE),
    "NCR"           :NeighbourhoodCleaningRule(random_state = RANDOM_STATE),
    "OSS"           :OneSidedSelection(random_state = RANDOM_STATE),
    "TL"            :TomekLinks(random_state = RANDOM_STATE)
}
X["sample"]["NULL"] , y["sample"]["NULL"]= make_classification(n_samples=5000, n_features=25,
                           n_clusters_per_class=1, n_informative=15,
                           random_state=RANDOM_STATE)
#print format(Counter(y))

X["sample"]["imb"] , y["sample"]["imb"] = imbd.make_imbalance(X["sample"]["NULL"], y["sample"]["NULL"] , sampling_strategy = {0 : 100 , 1 : 2501 })
X["train"]["imb"], X["test"]["imb"], y["train"]["imb"], y["train"]["imb"]= train_test_split(X["sample"]["imb"] , y["sample"]["imb"],test_size = 0.33 , random_state = 66 )







for clf in samplers:
    X["sample"][clf], y["sample"][clf] = samplers[clf].fit_resample(X["sample"]["imb"], y["sample"]["imb"])
    X["train"][clf],X["test"][clf],y["train"][clf],y["test"][clf]\
            =train_test_split(X["sample"][clf],y["sample"][clf], test_size = 0.33, random_state = 66) 
    clfs[clf].fit(X["train"][clf], y["train"][clf])
    predicts[clf] = clfs[clf].predict(X["test"][clf])
    print "RF with " , clf, " and no optimization"
    print clfs[clf].oob_score_
    print classification_report(y["test"][clf], predicts[clf])
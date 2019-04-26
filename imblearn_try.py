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
from imblearn.under_sampling import RandomUnderSampler, InstanceHardnessThreshold
from copy import deepcopy


RANDOM_STATE = 42


X, y = make_classification(n_samples=5000, n_features=25,
                           n_clusters_per_class=1, n_informative=15,
                           random_state=RANDOM_STATE)
#print format(Counter(y))

X_imb, y_imb = imbd.make_imbalance(X ,y , sampling_strategy = {0 : 100 , 1 : 2501 })
X_imb_train, X_imb_test, y_imb_train, y_imb_test = train_test_split(X_imb, y_imb, test_size=0.33, random_state=66)

rf = RandomForestClassifier( n_estimators=100,
                        oob_score=True,
                        max_features="log2",
                        random_state=RANDOM_STATE)

rf_imb =  imbe.BalancedRandomForestClassifier(  n_estimators = 100,
                                                oob_score = True,
                                                max_features = "log2",
                                                random_state = RANDOM_STATE)


rf_oversamp     = deepcopy(rf)
rf_SMOTE        = deepcopy(rf)
rf_undersamp    = deepcopy(rf)
rf_IHT          = deepcopy(rf)


random_over_sampler     = RandomOverSampler(random_state = RANDOM_STATE)
SMOTE_sampler           = SMOTE(random_state = RANDOM_STATE)
random_under_sampler    = RandomUnderSampler(random_state = RANDOM_STATE)
IHT_sampler             = InstanceHardnessThreshold(random_state = RANDOM_STATE)


X_oversamp , y_oversamp = random_over_sampler.fit_resample(X_imb , y_imb)
X_oversamp_train, X_oversamp_test, y_oversamp_train, y_oversamp_test = train_test_split(X_oversamp, y_oversamp, test_size=0.33, random_state=66)

X_SMOTE , y_SMOTE = SMOTE_sampler.fit_resample(X_imb , y_imb)
X_SMOTE_train, X_SMOTE_test, y_SMOTE_train, y_SMOTE_test = train_test_split(X_SMOTE, y_SMOTE, test_size=0.33, random_state=66)

X_undersamp , y_undersamp = random_under_sampler.fit_resample(X_imb , y_imb)
X_undersamp_train, X_undersamp_test, y_undersamp_train, y_undersamp_test = train_test_split(X_undersamp, y_undersamp, test_size=0.33, random_state=66)

X_IHT , y_IHT = IHT_sampler.fit_resample(X_imb , y_imb)
X_IHT_train, X_IHT_test, y_IHT_train, y_IHT_test = train_test_split(X_IHT, y_IHT, test_size=0.33, random_state=66)


rf_IHT.fit(X_IHT_train, y_IHT_train)
rf_IHT_predict = rf_IHT.predict(X_IHT_test)

rf_undersamp.fit(X_undersamp_train, y_undersamp_train)
rf_undersamp_predict = rf_undersamp.predict(X_undersamp_test)

rf_SMOTE.fit(X_SMOTE_train, y_SMOTE_train)
rf_SMOTE_predict = rf_SMOTE.predict(X_SMOTE_test)

rf_oversamp.fit(X_oversamp_train, y_oversamp_train)
rf_oversamp_predict = rf_oversamp.predict(X_oversamp_test)

rf.fit(X_imb_train, y_imb_train)
rf_predict =  rf.predict(X_imb_test)

rf_imb.fit(X_imb_train , y_imb_train)
rf_imb_predict = rf_imb.predict(X_imb_test)


print "RF with Imbalanced Data and no optimization"
print rf.oob_score_
print classification_report(y_imb_test, rf_predict)

print "RF with Imbalanced Data and BalancedRF"
print rf_imb.oob_score_
print classification_report(y_imb_test, rf_imb_predict)

print "RF with Oversampled Data(Random) and no optimization"
print rf_oversamp.oob_score_
print classification_report(y_oversamp_test, rf_oversamp_predict)

print "RF with Oversampled Data(SMOTE) and no optimization"
print rf_SMOTE.oob_score_
print classification_report(y_SMOTE_test, rf_SMOTE_predict)

print "RF with Undersample Data(Random) and no optimization"
print rf_undersamp.oob_score_
print classification_report(y_undersamp_test, rf_undersamp_predict)

print "RF with Undersampled Data(InstanceHardnessThreshold) and no optimization"
print rf_IHT.oob_score_
print classification_report(y_IHT_test, rf_IHT_predict)

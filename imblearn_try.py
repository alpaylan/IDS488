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
from imblearn.over_sampling import RandomOverSampler
RANDOM_STATE = 42

X, y = make_classification(n_samples=5000, n_features=25,
                           n_clusters_per_class=1, n_informative=15,
                           random_state=RANDOM_STATE)
print format(Counter(y))

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

rf_oversamp = RandomForestClassifier( n_estimators=100,
                        oob_score=True,
                        max_features="log2",
                        random_state=RANDOM_STATE)

random_over_sampler = RandomOverSampler(random_state = RANDOM_STATE)

X_oversamp , y_oversamp = random_over_sampler.fit_resample(X_imb , y_imb)
X_oversamp_train, X_oversamp_test, y_oversamp_train, y_oversamp_test = train_test_split(X_oversamp, y_oversamp, test_size=0.33, random_state=66)

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

print "RF with Oversampled Data and no optimization"
print rf_oversamp.oob_score_
print classification_report(y_oversamp_test, rf_oversamp_predict)
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

RANDOM_STATE = 42



_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(n_estimators=100,
                               oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(n_estimators=100,
                               max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features=None",
        RandomForestClassifier(n_estimators=100,
                               max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE))
]

imb_clfs = [
    ("BalancedRandomForestClassifier, max_features='sqrt'",
        BalancedRandomForestClassifier(n_estimators=100,
                               oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE)),
    ("BalancedRandomForestClassifier, max_features='log2'",
        BalancedRandomForestClassifier(n_estimators=100,
                               max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("BalancedRandomForestClassifier, max_features=None",
        BalancedRandomForestClassifier(n_estimators=100,
                               max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE))
] 
import numpy as np
from sklearn.dummy import DummyClassifier

import spilearn.utils as utils


reports = []
utils.train.report = reports.append

X = np.arange(180).reshape(60, 3)
y = np.tile(np.arange(2), 30)
groups = np.repeat(np.arange(3), 20)


def check_group_splits(estimator, X, y, cv, groups, **_):
    scores = []
    for train, test in cv:
        assert set(groups[train]).isdisjoint(groups[test])
        scores.append(0.5)
    return np.asarray(scores)


utils.cross_val_score = check_group_splits
utils.run_cv(
    {},
    X,
    y,
    {},
    lambda model, params: DummyClassifier(strategy="most_frequent"),
    groups=groups,
    subset_fractions=(1.0,),
    n_splits=3,
    n_repeats_final=1,
    n_jobs=1,
)

assert reports[-1]["mean_accuracy"] == 0.5

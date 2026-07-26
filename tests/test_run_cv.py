from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier

import spilearn.utils as utils


reports = []
utils.train.report = reports.append

X, y = load_iris(return_X_y=True)
utils.run_cv(
    {},
    X,
    y,
    {},
    lambda model, params: DummyClassifier(strategy='most_frequent'),
    subset_fractions=(0.5, 1.0),
    n_splits=3,
    n_repeats_final=2,
    n_jobs=1,
)

assert [report['subset_fraction'] for report in reports] == [0.5, 1.0]
assert all('mean_accuracy' in report for report in reports)

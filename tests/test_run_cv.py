from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier

import spilearn.experiments._tuning as utils


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
assert all('mean_score' in report and 'std_score' in report for report in reports)
assert all('mean_accuracy' not in report and 'std_accuracy' not in report for report in reports)


def negative_score(estimator, X_val, y_val):
    return -float((estimator.predict(X_val) == y_val).mean()) - float(X_val[:, 0].mean()) / 1000


custom_reports = []
utils.train.report = custom_reports.append
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
    scorer=negative_score,
)

assert len(custom_reports) == 2
assert [report['subset_fraction'] for report in custom_reports] == [0.5, 1.0]
assert all('mean_score' in report and 'std_score' in report for report in custom_reports)
assert all('mean_accuracy' not in report and 'std_accuracy' not in report for report in custom_reports)
assert all(report['mean_score'] < 0 and report['std_score'] > 0 for report in custom_reports)
assert all(report['metric'] == report['mean_score'] - .5 * report['std_score'] for report in custom_reports)


def failing_score(estimator, X_val, y_val):
    raise RuntimeError('scorer failed')


failed_reports = []
utils.train.report = failed_reports.append
try:
    utils.run_cv(
        {}, X, y, {}, lambda model, params: DummyClassifier(),
        subset_fractions=(1.0,), n_splits=3, n_repeats_final=1, n_jobs=1,
        scorer=failing_score,
    )
except RuntimeError as exc:
    assert str(exc) == 'scorer failed'
else:
    raise AssertionError('custom scorer failure must propagate')
assert failed_reports == []


for nonfinite_score in (float('nan'), float('inf')):
    def invalid_score(estimator, X_val, y_val, score=nonfinite_score):
        return score
    try:
        utils.run_cv(
            {}, X, y, {}, lambda model, params: DummyClassifier(),
            subset_fractions=(1.0,), n_splits=3, n_repeats_final=1, n_jobs=1,
            scorer=invalid_score,
        )
    except ValueError:
        pass
    else:
        raise AssertionError('non-finite custom scorer output must be rejected')
    assert failed_reports == []


def failing_pipe(model, params):
    raise RuntimeError('pipe failed')


try:
    utils.run_cv(
        {}, X, y, {}, failing_pipe,
        subset_fractions=(1.0,), n_splits=3, n_repeats_final=1, n_jobs=1,
        scorer=negative_score,
    )
except RuntimeError as exc:
    assert str(exc) == 'pipe failed'
else:
    raise AssertionError('pipeline creation failure must propagate for custom scoring')
assert failed_reports == []

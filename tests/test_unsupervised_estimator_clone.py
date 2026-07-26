from sklearn.base import clone

from examples.iris_unsupervised import build_pipeline
from spilearn.estimators import UnsupervisedTemporalTransformer


template = build_pipeline().steps[-2][1]
estimator = UnsupervisedTemporalTransformer(
    template.settings,
    template.model,
    num_threads=3,
    num_procs=2,
    use_inhibition=False,
    test_with_inhibition=True,
    threshold=7,
    high_threshold_teacher=True,
)
cloned = clone(estimator)

assert cloned._network.n_threads == 3
assert cloned._network.n_procs == 2
assert cloned._network.use_inhibition is False
assert cloned._network.test_with_inhibition is True
assert cloned._network.learning_threshold == 7
assert cloned._network.high_threshold_teacher is True

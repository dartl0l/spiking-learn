from sklearn.base import clone

from spilearn.estimators import UnsupervisedTemporalTransformer


settings = {"topology": {}, "network": {}, "learning": {}}
model = {"syn_dict_exc": {"synapse_model": "static_synapse"}}
estimator = UnsupervisedTemporalTransformer(
    settings,
    model,
    num_threads=3,
    num_procs=2,
    use_inhibition=False,
    test_with_inhibition=True,
    threshold=7,
    high_threshold_teacher=True,
    random_state=42,
)
cloned = clone(estimator)

assert cloned.random_state == 42
assert cloned._network.random_state == 42
assert cloned._network.n_threads == 3
assert cloned._network.n_procs == 2
assert cloned._network.use_inhibition is False
assert cloned._network.test_with_inhibition is True
assert cloned._network.learning_threshold == 7
assert cloned._network.high_threshold_teacher is True

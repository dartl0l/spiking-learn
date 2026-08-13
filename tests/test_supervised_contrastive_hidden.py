# coding: utf-8

import copy

import nest
import numpy as np

from spilearn.estimators import (
    FirstSpikeVotingClassifier,
    SupervisedContrastiveMultilayerRLClassifier,
)


settings = {'learning': {}, 'network': {}, 'topology': {}}
model = {
    'neuron_out': {'C_m': 1.0, 'V_m': 0.0, 'V_th': 1.0, 'tau_m': 10.0},
    'neuron_out_model': 'iaf_psc_delta',
    'neuron_hid': {'C_m': 1.0, 'V_m': 0.0, 'V_th': 1.0, 'tau_m': 10.0},
    'neuron_hid_model': 'iaf_psc_delta',
    'syn_dict_exc': {
        'weight': 10.0,
        'lambda': 0.01,
        'mu_plus': 0.0,
        'mu_minus': 0.0,
        'synapse_model': 'stdp_synapse',
        'tau_plus': 20.0,
        'Wmax': 20.0,
    },
    'syn_dict_exc_hid': {
        'weight': 10.0,
        'lambda': 0.01,
        'mu_plus': 0.0,
        'mu_minus': 0.0,
        'synapse_model': 'stdp_synapse',
        'tau_plus': 20.0,
        'Wmax': 20.0,
    },
}
X = np.tile([1.0, 2.0] * 32, (4, 1))[..., None]
y = np.array([0, 1, 0, 1])

classifier = SupervisedContrastiveMultilayerRLClassifier(
    settings,
    model,
    pool_size=1,
    learning_rate=0.01,
    learning_rate_hid=0.01,
    n_layer_hid=12,
    n_layer_out=10,
    n_input=64,
    epochs=1,
    h_time=20.0,
    start_delta=5.0,
    use_inhibition=False,
    random_state=42,
)
classifier.fit(X, y)
weights_before = copy.deepcopy(classifier._weights)

classifier._network.test(
    X, classifier._weights, threshold_state=classifier._threshold_state
)
hidden_events = nest.GetStatus(
    classifier._network.spike_detector_hidden, keys='events'
)[0]
hidden_ids = classifier._network.layer_hid.tolist()
assert hidden_events['senders'].min() == min(hidden_ids)
assert hidden_events['senders'].max() == max(hidden_ids)
assert hidden_events['senders'].max() > classifier.n_layer_hid

hidden_first = classifier.transform_hidden(X)
hidden_second = classifier.transform_hidden(X)

assert hidden_first.shape == (len(X), classifier.n_layer_hid)
np.testing.assert_allclose(hidden_first, hidden_second, equal_nan=True)
assert classifier._weights == weights_before
assert FirstSpikeVotingClassifier().fit(hidden_first, y).predict(hidden_second).shape == y.shape

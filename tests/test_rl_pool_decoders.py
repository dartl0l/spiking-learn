"""Small standalone check; run in the project's NEST environment."""

import numpy as np
from sklearn.base import clone

from spilearn.estimators import SupervisedTemporalRLPoolClassifier
from spilearn.network.rl_pool import LiteRlPoolNetwork
from spilearn.utils import predict_from_class_pool


def check_decoders():
    three_winners = np.array([[[2., 20., 20.], [4., 5., 6.],
                              [1., np.nan, np.nan]]])
    for decoder, winner in {'vote': 0, 'mean_latency': 1, 'min_latency': 2}.items():
        assert predict_from_class_pool(three_winners, decoder).tolist() == [winner]
    # Vote prefers class 0, while minimum prefers class 2.
    latency = np.array([[[5., 6., 7.], [2., 3., np.nan],
                         [1., np.nan, np.nan]]])
    expected = {"vote": 0, "mean_latency": 0, "min_latency": 2}
    # With an infinite latency for every silent neuron, the only fully
    # firing pool wins the mean. A separate dense sample isolates mean.
    dense = np.array([[[1., 20., 20.], [4., 5., 6.], [8., 9., 10.]]])
    for decoder, prediction in expected.items():
        assert predict_from_class_pool(latency, decoder).tolist() == [prediction]
    assert predict_from_class_pool(dense, "mean_latency").tolist() == [1]
    assert predict_from_class_pool(dense, "min_latency").tolist() == [0]

    sparse = np.array([[[np.nan, 5.], [2., np.nan], [np.nan, np.nan]],
                       [[np.nan, np.nan], [np.nan, np.nan],
                        [np.nan, np.nan]]])
    for decoder in expected:
        assert predict_from_class_pool(sparse, decoder).tolist() == [1, -1]
        tied = np.array([[[2., np.nan], [2., np.nan]]])
        assert predict_from_class_pool(tied, decoder).tolist() == [0]
        network = LiteRlPoolNetwork.__new__(LiteRlPoolNetwork)
        network.decoder = decoder
        assert network.predict_from_latency_pool(latency).tolist() == [expected[decoder]]

    try:
        predict_from_class_pool(latency, "unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown decoder was accepted")

    for decoder in expected:
        estimator = SupervisedTemporalRLPoolClassifier(
            settings={"topology": {}, "network": {}, "learning": {}},
            model={"syn_dict_exc": {"synapse_model": "static_synapse"}},
            pool_size=3, learning_rate=.1, decoder=decoder, random_state=42)
        copied = clone(estimator)
        assert copied.get_params()["decoder"] == decoder
        assert copied.get_params()["random_state"] == 42
        assert copied._network.decoder == decoder
        assert copied._network.random_state == 42

    try:
        SupervisedTemporalRLPoolClassifier(
            settings={"topology": {}, "network": {}, "learning": {}},
            model={"syn_dict_exc": {"synapse_model": "static_synapse"}},
            pool_size=3, learning_rate=.1, decoder="unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown estimator decoder was accepted")


if __name__ == "__main__":
    check_decoders()
    print("RL pool decoder checks passed")

import numpy as np

from spilearn.network import LiteRlPoolNetwork, MultilayerRLNetwork
from spilearn.utils import predict_from_earliest_latency_pool


latency = np.array(
    [
        [[1.0, 9.0, 9.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
        [[5.0, np.nan, np.nan], [2.0, np.nan, np.nan], [3.0, np.nan, np.nan]],
        np.full((3, 3), np.nan),
    ]
)

np.testing.assert_array_equal(
    predict_from_earliest_latency_pool(latency),
    [0, 1, -1],
)
pool_network = LiteRlPoolNetwork.__new__(LiteRlPoolNetwork)
pool_network.decoder = 'min_latency'
np.testing.assert_array_equal(
    pool_network.predict_from_latency_pool(latency),
    [0, 1, -1],
)
np.testing.assert_array_equal(
    MultilayerRLNetwork.predict_from_latency_pool(None, latency),
    [1, -1, -1],
)

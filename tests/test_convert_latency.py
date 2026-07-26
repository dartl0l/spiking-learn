import numpy as np

from spilearn.utils import convert_latency


latency = [{'senders': [4, 3, 4], 'spikes': [2.0, 1.0, 3.0]}]
converted = convert_latency(latency, n_neurons=2, offset=3)

np.testing.assert_array_equal(converted, [[1.0, 2.0]])

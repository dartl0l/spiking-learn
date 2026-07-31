import numpy as np

from spilearn.network.base import Network


network = object.__new__(Network)
network.random_state = 42

first_seed = network._seed_rng()
first_value = np.random.random()
second_seed = network._seed_rng()
second_value = np.random.random()

assert first_seed == second_seed
assert first_value == second_value
assert 0 < first_seed < 2**32

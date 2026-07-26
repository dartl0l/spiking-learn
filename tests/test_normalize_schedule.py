from spilearn.network.epoch import EpochNetwork


assert [
    i for i in range(205) if EpochNetwork.should_normalize(i, 100)
] == [99, 199]
assert all(EpochNetwork.should_normalize(i, None) for i in range(3))

from examples.iris_supervised import build_pipeline as build_supervised
from examples.iris_unsupervised import build_pipeline as build_unsupervised


assert [type(step).__name__ for _, step in build_supervised().steps] == [
    "MinMaxScaler",
    "ReceptiveFieldsTransformer",
    "SupervisedTemporalClassifier",
]
assert [type(step).__name__ for _, step in build_unsupervised().steps] == [
    "MinMaxScaler",
    "ReceptiveFieldsTransformer",
    "UnsupervisedTemporalTransformer",
    "FirstSpikeVotingClassifier",
]

import numpy as np

from spilearn.estimators.encoders.dog import DoGTransformer


X = np.array(
    [
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
    ]
)
encoded = DoGTransformer(on_off=True).fit_transform(X)

assert encoded.shape == (2, 8)
assert np.all((0.0 <= encoded) & (encoded <= 1.0))

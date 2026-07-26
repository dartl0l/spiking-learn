# coding: utf-8

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.base import BaseEstimator, TransformerMixin


class DoGTransformer(BaseEstimator, TransformerMixin):
    """Difference-of-Gaussians (center-surround) image encoder.

    Applies an ON-center / OFF-center DoG filter to each (flattened, square)
    image and returns rectified contrast maps, as in Kheradpisheh et al. 2018.
    Meant to sit *before* a temporal encoder (e.g. ``TemporalPatternTransformer``)
    in an sklearn ``Pipeline``: stronger local contrast -> larger value ->
    earlier spike after TTFS encoding.

    Parameters
    ----------
    sigma_center : float
        Std of the narrow (center) Gaussian.
    sigma_surround : float
        Std of the wide (surround) Gaussian. Must be > ``sigma_center``.
    on_off : bool
        If True, output the ON channel (center brighter than surround) and the
        OFF channel (surround brighter) concatenated -> feature dim doubles.
        If False, output only the rectified ON channel (keeps a square dim,
        e.g. for ``ConvolutionNetwork``).
    image_shape : tuple[int, int] or None
        (H, W) of each sample. If None, a square image is inferred from the
        feature count (``sqrt(n_features)``), matching ``ConvolutionNetwork``.
    normalize : bool
        Per-sample scale each output channel to [0, 1] by its max response.
    threshold : float
        Responses with magnitude below this (after rectification, before
        normalization) are set to 0 to sparsify.
    mode : str
        Boundary mode passed to ``scipy.ndimage.gaussian_filter``.
    reshape : bool
        If True, return shape (n_samples, n_out_features, 1) like the other
        encoders; otherwise (n_samples, n_out_features).
    """

    def __init__(
        self,
        sigma_center=1.0,
        sigma_surround=2.0,
        on_off=True,
        image_shape=None,
        normalize=True,
        threshold=0.0,
        mode='reflect',
        reshape=False,
    ) -> None:
        self.sigma_center = sigma_center
        self.sigma_surround = sigma_surround
        self.on_off = on_off
        self.image_shape = image_shape
        self.normalize = normalize
        self.threshold = threshold
        self.mode = mode
        self.reshape = reshape

    def _get_shape(self, n_features):
        if self.image_shape is not None:
            return self.image_shape
        side = int(round(np.sqrt(n_features)))
        if side * side != n_features:
            raise ValueError(
                f'Cannot infer a square image from {n_features} features; '
                f'pass image_shape=(H, W).'
            )
        return side, side

    def _rectify(self, channel):
        if self.threshold > 0.0:
            channel = np.where(channel >= self.threshold, channel, 0.0)
        if self.normalize:
            peak = channel.max()
            if peak > 0.0:
                channel = channel / peak
        return channel

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = np.asarray(X, dtype=float)
        h, w = self._get_shape(X.shape[1])

        out = []
        for sample in X:
            image = sample.reshape(h, w)
            dog = gaussian_filter(image, self.sigma_center, mode=self.mode) - \
                gaussian_filter(image, self.sigma_surround, mode=self.mode)
            on = self._rectify(np.maximum(dog, 0.0)).ravel()
            if self.on_off:
                off = self._rectify(np.maximum(-dog, 0.0)).ravel()
                out.append(np.concatenate([on, off]))
            else:
                out.append(on)

        out = np.asarray(out)
        return out.reshape(out.shape[0], out.shape[1], 1) if self.reshape else out

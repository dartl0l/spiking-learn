# coding: utf-8

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ReceptiveFieldsTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_fields,
        sigma2=None,
        k_round=2,
        max_x=1.0,
        scale=1.0,
        reshape=True,
        reverse=False,
        no_last=False,
    ) -> None:
        self.sigma2 = sigma2
        self.max_x = max_x
        self.n_fields = n_fields
        self.k_round = k_round
        self.scale = scale
        self.reshape = reshape
        self.reverse = reverse
        self.no_last = no_last

    def _get_sigma_squared(self, min_x, max_x, n_fields):
        # as in [Yu et al. 2014]
        return (2 / 3 * (max_x - min_x) / (n_fields - 2)) ** 2

    def _get_gaussian(self, x, sigma2, mu):
        return (1 / np.sqrt(2 * sigma2 * np.pi)) * np.e ** (
            -((x - mu) ** 2) / (2 * sigma2)
        )

    def fit(self, X, y=None):
        h_mu = self.max_x / (self.n_fields - 1)
        if self.sigma2 is None:
            self.sigma2 = self._get_sigma_squared(0, self.max_x, self.n_fields)
        self.max_y = np.round(self._get_gaussian(h_mu, self.sigma2, h_mu), 0)
        self.mu = np.tile(np.linspace(0, self.max_x, self.n_fields), len(X[0]))
        return self

    def transform(self, X, y=None):
        X = np.repeat(X, self.n_fields, axis=1)
        assert len(self.mu) == len(X[0])

        X = np.round(self._get_gaussian(X, self.sigma2, self.mu), self.k_round)
        if self.no_last:
            X[X == 0] = np.nan
        X = X if self.reverse else self.max_y - X
        X *= self.scale
        return (
            X.reshape(X.shape[0], X.shape[1], 1)
            if self.reshape
            else X.reshape(X.shape[0], X.shape[1])
        )

# coding: utf-8

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TemporalPatternTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        pattern_length,
        k_round=2,
        max_x=1.0,
        reshape=True,
        reverse=False,
        no_last=False,
    ) -> None:
        self.pattern_length = pattern_length
        self.k_round = k_round
        self.max_x = max_x
        self.reshape = reshape
        self.reverse = reverse
        self.no_last = no_last

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.no_last:
            X[X == 0] = np.nan

        X = X if self.reverse else self.max_x - X
        X = np.round(self.pattern_length * X, self.k_round)
        return (
            X.reshape(X.shape[0], X.shape[1], 1)
            if self.reshape
            else X.reshape(X.shape[0], X.shape[1])
        )

# coding: utf-8

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TemporalPatternReversedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, pattern_length, k_round=2, max_x=1.0, reshape=True) -> None:
        self.pattern_length = pattern_length
        self.k_round = k_round
        self.max_x = max_x
        self.reshape = reshape

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[X == 0] = self.max_x
        X = np.round(self.pattern_length * X, self.k_round)
        return (
            X.reshape(X.shape[0], X.shape[1], 1)
            if self.reshape
            else X.reshape(X.shape[0], X.shape[1])
        )

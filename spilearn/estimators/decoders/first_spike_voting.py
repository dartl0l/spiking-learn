# coding: utf-8

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class FirstSpikeVotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, h_time=None) -> None:
        super().__init__()
        self.h_time = h_time

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.assignments_ = np.full(X.shape[1], None, dtype=object)
        best_latency = np.full(X.shape[1], np.inf)

        for current_class in self.classes_:
            class_latency = X[y == current_class]
            finite = np.isfinite(class_latency)
            count = finite.sum(axis=0)
            mean_latency = np.full(X.shape[1], np.inf)
            np.divide(
                np.where(finite, class_latency, 0.0).sum(axis=0),
                count,
                out=mean_latency,
                where=count > 0,
            )
            mask = mean_latency < best_latency
            self.assignments_[mask] = current_class
            best_latency[mask] = mean_latency[mask]

        self.classes = self.classes_
        self.assignments = self.assignments_
        return self

    def predict(self, X):
        predictions = []
        for latency in np.asarray(X, dtype=float):
            scores = []
            for current_class in self.classes_:
                values = latency[self.assignments_ == current_class]
                values = values[np.isfinite(values)]
                scores.append(values.mean() if len(values) else np.inf)
            predictions.append(self.classes_[int(np.argmin(scores))])
        return np.asarray(predictions)

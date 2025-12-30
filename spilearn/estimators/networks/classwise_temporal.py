# coding: utf-8
from typing import Optional

import numpy as np

from ...utils import (
    convert_latency,
    predict_from_latency,
    split_spikes_and_senders,
)
from .base import BaseTemporalEstimator


class ClasswiseTemporalClassifier(BaseTemporalEstimator):
    def __init__(
        self,
        settings,
        model,
        reshape=True,
        n_layer_out: Optional[int] = None,
        n_input: Optional[int] = None,
        epochs: Optional[int] = None,
        h_time: Optional[float] = None,
        start_delta: Optional[float] = None,
        h: Optional[float] = None,
        normalize_weights: bool = False,
        normalize_step: Optional[int] = None,
        w_target: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            settings,
            model,
            reshape,
            n_layer_out,
            n_input,
            epochs,
            h_time,
            start_delta,
            h,
            normalize_weights,
            normalize_step,
            w_target,
            **kwargs,
        )

    def fit(self, X, y):
        self.n_input = len(X[0])
        self._weights = []
        self._devices_fit = []
        self._network.n_input = self.n_input
        self._network.spike_generator.n_input = self.n_input
        for current_class in set(y):
            mask = y == current_class
            weights, _, devices_fit = self._network.train(X[mask], y[mask])
            self._weights.append(weights)
            self._devices_fit.append(devices_fit)
        return self

    def predict(self, X):
        full_output = []
        for weights in self._weights:
            output, self._devices_predict = self._network.test(X, weights)
            all_latency = split_spikes_and_senders(
                output, len(X), self.start_delta, self.h_time
            )
            out_latency = convert_latency(all_latency, self.n_layer_out)
            full_output.append(out_latency)
        y_pred = predict_from_latency(np.concatenate(full_output, axis=1))
        return y_pred.astype(int)

    def transform(self, X, y=None):
        full_output = []
        for weights in self._weights:
            output, self._devices_predict = self._network.test(X, weights)
            all_latency = split_spikes_and_senders(
                output, len(X), self.start_delta, self.h_time
            )
            out_latency = convert_latency(all_latency, self.n_layer_out)
            full_output.append(out_latency)
        out_latency = np.concatenate(full_output, axis=1)
        return (
            out_latency.reshape(out_latency.shape[0], out_latency.shape[1], 1)
            if self.reshape
            else out_latency
        )

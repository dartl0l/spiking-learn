# coding: utf-8
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from ...network import EpochNetwork
from ...spike_generator import TemporalSpikeGenerator
from ...utils import (
    convert_latency,
    predict_from_latency,
    split_spikes_and_senders,
)


class BaseTemporalEstimator(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(
        self,
        settings,  # deprecated
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
        self.model = model
        self.settings = settings

        self.epochs = epochs or settings['learning'].get('epochs', 1)
        self.n_input = n_input or settings['topology'].get('n_input', 2)
        self.n_layer_out = n_layer_out or settings['topology'].get('n_layer_out', 2)
        self.start_delta = start_delta or settings['network'].get('start_delta', 50)
        self.h_time = h_time or settings['network'].get('h_time', 50)
        self.h = h or settings['network'].get('h', 0.01)
        self.reshape = reshape
        self.normalize_weights = normalize_weights
        self.normalize_step = normalize_step
        self.w_target = w_target

        self._network = self._init_network(
            settings,
            model,
            normalize_weights=self.normalize_weights,
            normalize_step=self.normalize_step,
            w_target=self.w_target,
            **kwargs,
        )
        self._devices_fit = None
        self._weights = None

    def _init_network(self, settings, model, **kwargs):
        return EpochNetwork(
            settings,
            model,
            n_layer_out=self.n_layer_out,
            n_input=self.n_input,
            epochs=self.epochs,
            spike_generator=TemporalSpikeGenerator(
                self.n_input, self.epochs, self.h_time
            ),
            h_time=self.h_time,
            start_delta=self.start_delta,
            h=self.h,
            **kwargs,
        )

    def fit(self, X, y):
        self.n_input = len(X[0])
        self._network.n_input = self.n_input
        self._network.spike_generator.n_input = self.n_input
        self._weights, _, self._devices_fit = self._network.train(X, y)
        return self

    def predict(self, X):
        output, self._devices_predict = self._network.test(X, self._weights)

        all_latency = split_spikes_and_senders(
            output, len(X), self.start_delta, self.h_time
        )
        out_latency = convert_latency(all_latency, self.n_layer_out)
        y_pred = predict_from_latency(out_latency)
        return y_pred.astype(int)

    def transform(self, X, y=None):
        output, self._devices_predict = self._network.test(X, self._weights)

        all_latency = split_spikes_and_senders(
            output, len(X), self.start_delta, self.h_time
        )
        out_latency = np.array(convert_latency(all_latency, self.n_layer_out))
        return (
            out_latency.reshape(out_latency.shape[0], out_latency.shape[1], 1)
            if self.reshape
            else out_latency
        )

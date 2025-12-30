# coding: utf-8
from typing import Optional

from ...network import LiteRlPoolNetwork
from ...spike_generator import TemporalSpikeGenerator
from ...utils import (
    convert_latency_pool,
    predict_from_latency_pool,
    split_spikes_and_senders,
)
from .supervised_temporal_rl import SupervisedTemporalRLClassifier


class SupervisedTemporalRLPoolClassifier(SupervisedTemporalRLClassifier):
    def __init__(
        self,
        settings,
        model,
        pool_size,
        learning_rate,
        reshape=True,
        n_layer_out: Optional[int] = None,
        n_input: Optional[int] = None,
        epochs: Optional[int] = None,
        h_time: Optional[float] = None,
        start_delta: Optional[float] = None,
        h: Optional[float] = None,
        **kwargs,
    ) -> None:
        self.pool_size = pool_size
        super().__init__(
            settings,
            model,
            learning_rate,
            reshape,
            n_layer_out,
            n_input,
            epochs,
            h_time,
            start_delta,
            h,
            **kwargs,
        )

    def _init_network(self, settings, model, **kwargs):
        return LiteRlPoolNetwork(
            settings,
            model,
            pool_size=self.pool_size,
            learning_rate=self.learning_rate,
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

    def predict(self, X):
        output, self._devices_predict = self._network.test(X, self._weights)

        all_latency = split_spikes_and_senders(
            output, len(X), self.start_delta, self.h_time
        )
        out_latency = convert_latency_pool(
            all_latency, self.n_layer_out, self.pool_size
        )
        y_pred = predict_from_latency_pool(out_latency)
        return y_pred.astype(int)

# coding: utf-8
from typing import Optional

from ...network import LiteRlNetwork
from ...spike_generator import TemporalSpikeGenerator
from .base import BaseTemporalEstimator


class SupervisedTemporalRLClassifier(BaseTemporalEstimator):
    def __init__(
        self,
        settings,
        model,
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
        self.learning_rate = learning_rate
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
            **kwargs,
        )

    def _init_network(self, settings, model, **kwargs):
        return LiteRlNetwork(
            settings,
            model,
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

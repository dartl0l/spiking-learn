# coding: utf-8
from typing import Optional

from ...network import EpochNetwork
from ...noise import NoiseGenerator
from ...spike_generator import TemporalSpikeGenerator
from .base import BaseTemporalEstimator


class UnsupervisedTemporalNoiseTransformer(BaseTemporalEstimator):
    def __init__(
        self,
        settings,
        model,
        reshape=True,
        noise: Optional[float] = None,
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
        self.noise = noise
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
            noise=NoiseGenerator(noise_freq=self.noise, n_input=self.n_input),
            h_time=self.h_time,
            start_delta=self.start_delta,
            h=self.h,
            **kwargs,
        )

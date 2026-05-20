# coding: utf-8
from typing import Optional

from ...network import MultilayerRLNetwork
from ...spike_generator import TemporalSpikeGenerator
from .supervised_temporal_rl_pool import SupervisedTemporalRLPoolClassifier


class SupervisedMultilayerRLClassifier(SupervisedTemporalRLPoolClassifier):
    def __init__(
        self,
        settings,
        model,
        pool_size,
        learning_rate,
        learning_rate_hid,
        n_layer_hid,
        goodness_threshold_hid=None,
        goodness_tau_hid=None,
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
        w_target_hid: Optional[float] = None,
        **kwargs,
    ) -> None:
        self.learning_rate_hid = learning_rate_hid
        self.goodness_threshold_hid = goodness_threshold_hid
        self.goodness_tau_hid = goodness_tau_hid
        self.n_layer_hid = n_layer_hid
        self.w_target_hid = w_target_hid
        super().__init__(
            settings,
            model,
            pool_size,
            learning_rate,
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
            w_target_hid=w_target_hid,
            **kwargs,
        )

    def _init_network(self, settings, model, **kwargs):
        return MultilayerRLNetwork(
            settings,
            model,
            pool_size=self.pool_size,
            learning_rate=self.learning_rate,
            learning_rate_hid=self.learning_rate_hid,
            goodness_threshold_hid=self.goodness_threshold_hid,
            goodness_tau_hid=self.goodness_tau_hid,
            n_layer_hid=self.n_layer_hid,
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

# coding: utf-8
from typing import Optional

import nest
import numpy as np

from ...network import ContrastiveMultilayerRLNetwork
from ...spike_generator import TemporalSpikeGenerator
from ...utils import convert_latency, split_spikes_and_senders
from .supervised_temporal_rl_pool import SupervisedTemporalRLPoolClassifier


class SupervisedContrastiveMultilayerRLClassifier(SupervisedTemporalRLPoolClassifier):
    """Estimator wrapper around :class:`ContrastiveMultilayerRLNetwork`.

    Same interface as :class:`SupervisedMultilayerRLClassifier`, plus the
    contrastive negative-sample controls ``negative_mode`` / ``negative_fn``.
    """

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
        negative_mode='shuffle',
        negative_fn=None,
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
        random_state: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.learning_rate_hid = learning_rate_hid
        self.goodness_threshold_hid = goodness_threshold_hid
        self.goodness_tau_hid = goodness_tau_hid
        self.negative_mode = negative_mode
        self.negative_fn = negative_fn
        self.n_layer_hid = n_layer_hid
        self.w_target_hid = w_target_hid
        self.random_state = random_state
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
            random_state=random_state,
            **kwargs,
        )

    def _init_network(self, settings, model, **kwargs):
        return ContrastiveMultilayerRLNetwork(
            settings,
            model,
            pool_size=self.pool_size,
            learning_rate=self.learning_rate,
            learning_rate_hid=self.learning_rate_hid,
            goodness_threshold_hid=self.goodness_threshold_hid,
            goodness_tau_hid=self.goodness_tau_hid,
            negative_mode=self.negative_mode,
            negative_fn=self.negative_fn,
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

    def transform_hidden(self, X):
        self._network.test(
            X, self._weights, threshold_state=self._threshold_state
        )
        events = nest.GetStatus(
            self._network.spike_detector_hidden, keys='events'
        )[0]
        hidden_offset = self._network.layer_hid.tolist()[0]
        all_latency = split_spikes_and_senders(
            {
                'spikes': events['times'].copy(),
                'senders': events['senders'].copy() - hidden_offset + 1,
            },
            len(X),
            self.start_delta,
            self.h_time,
        )
        return np.asarray(convert_latency(all_latency, self.n_layer_hid))

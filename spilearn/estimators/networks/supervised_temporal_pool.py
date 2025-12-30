# coding: utf-8
from typing import Optional

from ...network import EpochNetwork
from ...spike_generator import TemporalSpikeGenerator
from ...teacher import TeacherPool
from ...utils import (
    convert_latency_pool,
    predict_from_latency_pool,
    split_spikes_and_senders,
)
from .supervised_temporal import SupervisedTemporalClassifier


class SupervisedTemporalPoolClassifier(SupervisedTemporalClassifier):
    def __init__(
        self,
        settings,
        model,
        pool_size,
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
        teacher_amplitude: Optional[float] = None,
        reinforce_delta: Optional[float] = None,
        reinforce_time: Optional[float] = None,
        use_min_teacher=True,
        **kwargs,
    ) -> None:
        self.pool_size = pool_size
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
            teacher_amplitude,
            reinforce_delta,
            reinforce_time,
            use_min_teacher,
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
            teacher=TeacherPool(
                n_layer_out=self.n_layer_out,
                pool_size=self.pool_size,
                teacher_amplitude=self.teacher_amplitude,
                reinforce_delta=self.reinforce_delta,
                reinforce_time=self.reinforce_time,
                start=self.start_delta,
                h_time=self.h_time,
                h=self.h,
                use_min=self.use_min_teacher,
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

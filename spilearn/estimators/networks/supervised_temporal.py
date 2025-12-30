# coding: utf-8
from typing import Optional

from ...network import EpochNetwork
from ...spike_generator import TemporalSpikeGenerator
from ...teacher import Teacher
from .base import BaseTemporalEstimator


class SupervisedTemporalClassifier(BaseTemporalEstimator):
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
        teacher_amplitude: Optional[float] = None,
        reinforce_delta: Optional[float] = None,
        reinforce_time: Optional[float] = None,
        use_min_teacher=True,
        **kwargs,
    ) -> None:
        self.teacher_amplitude = teacher_amplitude or settings['learning'].get(
            'teacher_amplitude', 1000
        )
        self.reinforce_delta = reinforce_delta or settings['learning'].get(
            'reinforce_delta', 0
        )
        self.reinforce_time = reinforce_time or settings['learning'].get(
            'reinforce_time', 0
        )
        self.use_min_teacher = use_min_teacher
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
        return EpochNetwork(
            settings,
            model,
            n_layer_out=self.n_layer_out,
            n_input=self.n_input,
            epochs=self.epochs,
            spike_generator=TemporalSpikeGenerator(
                self.n_input, self.epochs, self.h_time
            ),
            teacher=Teacher(
                n_layer_out=self.n_layer_out,
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

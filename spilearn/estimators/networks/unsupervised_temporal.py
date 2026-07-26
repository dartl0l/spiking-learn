# coding: utf-8
from typing import Optional

from .base import BaseTemporalEstimator


class UnsupervisedTemporalTransformer(BaseTemporalEstimator):
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
        num_threads: Optional[int] = None,
        num_procs: Optional[int] = None,
        use_inhibition: Optional[bool] = None,
        test_with_inhibition: Optional[bool] = None,
        threshold: Optional[float] = None,
        high_threshold_teacher: Optional[bool] = None,
        **kwargs,
    ) -> None:
        self.num_threads = num_threads
        self.num_procs = num_procs
        self.use_inhibition = use_inhibition
        self.test_with_inhibition = test_with_inhibition
        self.threshold = threshold
        self.high_threshold_teacher = high_threshold_teacher
        network_kwargs = dict(kwargs)
        network_kwargs.update(
            {
                name: value
                for name, value in (
                    ('num_threads', num_threads),
                    ('num_procs', num_procs),
                    ('use_inhibition', use_inhibition),
                    ('test_with_inhibition', test_with_inhibition),
                    ('threshold', threshold),
                    ('high_threshold_teacher', high_threshold_teacher),
                )
                if value is not None
            }
        )
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
            **network_kwargs,
        )

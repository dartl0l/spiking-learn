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
            **kwargs,
        )

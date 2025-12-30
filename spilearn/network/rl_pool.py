# coding: utf-8

import nest

from .rl import LiteRlNetwork
from ..utils import (
    convert_latency_pool,
    predict_from_latency_pool,
)


class LiteRlPoolNetwork(LiteRlNetwork):
    def __init__(self, settings, model, pool_size, learning_rate, **kwargs):
        super().__init__(settings, model, learning_rate, **kwargs)

        self.pool_size = pool_size

    def update_learning_rate_separate(self, learning_rate, action):
        neurons = self.layer_out[
            action * self.pool_size : action * self.pool_size + self.pool_size
        ]
        connection = nest.GetConnections(self.input_layer, target=neurons)
        nest.SetStatus(connection, 'lambda', learning_rate)

    def run(self, n_neurons):
        self.update_learning_rate(0.0)

        raw_latency = self.sim()
        self.time += self.h_time

        out_latency = convert_latency_pool([raw_latency], n_neurons, self.pool_size)
        y_pred = predict_from_latency_pool([out_latency[0]])

        return int(y_pred[0])

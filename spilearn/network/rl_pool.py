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

    def update_learning_rate_separate_out(self, learning_rate, action):
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

    def connect_layers_inh(self):
        for i in range(0, self.n_layer_out, self.pool_size):
            neurons = self.layer_out[
                i : i + self.pool_size
            ]
            self.interconnect_layer(
                nest.NodeCollection(neurons),
                self.model['syn_dict_inh'],
                self.model.get('conn_dict_inh', {'rule': 'all_to_all'}),
            )
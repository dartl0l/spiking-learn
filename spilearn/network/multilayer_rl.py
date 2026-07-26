# coding: utf-8

import nest
import numpy as np

from .rl_pool import LiteRlPoolNetwork
from ..utils import (
    convert_latency_pool,
    predict_from_latency_pool,
)


class MultilayerRLNetwork(LiteRlPoolNetwork):
    def __init__(
        self,
        settings,
        model,
        pool_size,
        learning_rate,
        learning_rate_hid,
        n_layer_hid,
        goodness_threshold_hid = None,
        **kwargs
    ):
        self.n_layer_hid = n_layer_hid
        self.layer_hid = None
        self.spike_detector_hidden = None
        self.voltmeter_hidden = None
        self.use_reciprocal = kwargs.get('use_reciprocal', False)
        self.w_target_hid = kwargs.get('w_target_hid') or 1
        self.learning_rate_hid = learning_rate_hid
        self.goodness_threshold_hid = goodness_threshold_hid or 0.9
        self.goodness_tau_hid = kwargs.get('goodness_tau_hid')
        self.goodness_mean_decay_hid = kwargs.get('goodness_mean_decay_hid', 0.995)
        self.last_hidden_latency = None
        self.last_hidden_goodness = False
        self.last_hidden_energy = 0.0
        self.last_hidden_threshold = goodness_threshold_hid
        self.goodness_running_mean = None
        self.hidden_energy_history = []

        super().__init__(
            settings=settings,
            model=model,
            pool_size=pool_size,
            learning_rate=learning_rate,
            n_layer_hid=n_layer_hid,
            **kwargs
        )
        self.w_targets = [self.w_target_hid, self.w_target]
        self.synapse_models = [
            self.model['syn_dict_exc_hid']['synapse_model'],
            self.model['syn_dict_exc']['synapse_model'],
        ]

    def create_layers(self):
        self.layer_out = nest.Create(self.model['neuron_out_model'], self.n_layer_out)
        self.layer_hid = nest.Create(self.model['neuron_hid_model'], self.n_layer_hid)
        self.input_layer = nest.Create('parrot_neuron', self.n_input)
        self.layers = [self.input_layer, self.layer_hid, self.layer_out]

    def create_devices(self):
        super().create_devices()
        self.spike_detector_hidden = nest.Create('spike_recorder')
        if self.need_devices:
            self.voltmeter_hidden = nest.Create('multimeter', 1, {'record_from': ['V_m']})

    def connect_layers(self):
        nest.Connect(
            self.input_layer,
            self.layer_hid,
            self.model.get('conn_dict_exc_hid', {'rule': 'all_to_all'}),
            syn_spec=self.model['syn_dict_exc_hid'],
        )
        nest.Connect(
            self.layer_hid,
            self.layer_out,
            self.model.get('conn_dict_exc', {'rule': 'all_to_all'}),
            syn_spec=self.model['syn_dict_exc'],
        )
        if self.use_reciprocal:
            nest.Connect(
                self.layer_out,
                self.layer_hid,
                self.model.get('conn_dict_rec', {'rule': 'all_to_all'}),
                syn_spec=self.model['syn_dict_rec'],
            )

    def connect_layers_static(self):
        nest.Connect(
            self.input_layer,
            self.layer_hid,
            self.model.get('conn_dict_exc_hid', {'rule': 'all_to_all'}),
            syn_spec='static_synapse',
        )
        nest.Connect(
            self.layer_hid,
            self.layer_out,
            self.model.get('conn_dict_exc', {'rule': 'all_to_all'}),
            syn_spec='static_synapse',
        )

    def connect_layers_inh(self):
        self.interconnect_layer(
            self.layer_hid,
            self.model['syn_dict_inh_hid'],
            self.model.get('conn_dict_inh_hid', {'rule': 'all_to_all'}),
        )
        super().connect_layers_inh()

    def connect_devices(self):
        super().connect_devices()
        nest.Connect(
            self.layer_hid,
            self.spike_detector_hidden,
            conn_spec={'rule': 'all_to_all'},
        )
        if self.voltmeter_hidden:
            nest.Connect(self.voltmeter_hidden, self.layer_hid)

    def get_devices(self):
        devices = super().get_devices()
        if self.voltmeter_hidden:
            devices['multimeter_hidden'] = nest.GetStatus(
                self.voltmeter_hidden, keys='events'
            )[0]
        if self.spike_detector_hidden:
            devices['spike_detector_hidden'] = nest.GetStatus(
                self.spike_detector_hidden, keys='events'
            )[0]
        return devices

    def set_neuron_status(self, override_threshold=False):
        super().set_neuron_status(override_threshold)
        nest.SetStatus(self.layer_hid, self.model['neuron_hid'])

    def update_learning_rate(self, learning_rate):
        self.update_learning_rate_hid(learning_rate)
        connection = nest.GetConnections(self.layer_hid, target=self.layer_out)
        nest.SetStatus(connection, 'lambda', learning_rate)

    def update_learning_rate_hid(self, learning_rate):
        connection = nest.GetConnections(self.input_layer, target=self.layer_hid)
        nest.SetStatus(connection, 'lambda', learning_rate)

    def update_learning_rate_separate_out(self, learning_rate, action):
        neurons = self.layer_out[
            action * self.pool_size : action * self.pool_size + self.pool_size
        ]
        connection = nest.GetConnections(self.layer_hid, target=neurons)
        nest.SetStatus(connection, 'lambda', learning_rate)

    def get_hidden_energy(self, raw_hidden_latency):
        if not raw_hidden_latency or 'spikes' not in raw_hidden_latency:
            return 0.0

        spikes = raw_hidden_latency['spikes']
        finite = np.isfinite(spikes)

        if not finite.any():
            return 0.0

        tau = self.goodness_tau_hid or (self.h_time / 3)
        return float(np.mean(np.exp(-spikes[finite] / tau)))

    def get_hidden_goodness(self, raw_hidden_latency):
        energy = self.get_hidden_energy(raw_hidden_latency)
        if self.goodness_running_mean is None:
            self.goodness_running_mean = energy
        else:
            self.goodness_running_mean = (
                self.goodness_mean_decay_hid * self.goodness_running_mean
                + (1 - self.goodness_mean_decay_hid) * energy
            )

        threshold = self.goodness_threshold_hid * self.goodness_running_mean
        self.last_hidden_threshold = threshold
        return energy > threshold

    def run(self, n_neurons):
        self.update_learning_rate(0.0)

        raw_latency = self.sim()
        self.last_hidden_latency = self.get_raw_latency(self.spike_detector_hidden)
        self.last_hidden_energy = self.get_hidden_energy(self.last_hidden_latency)
        self.last_hidden_goodness = self.get_hidden_goodness(self.last_hidden_latency)
        self.hidden_energy_history.append((self.last_hidden_energy, self.goodness_running_mean))
        self.time += self.h_time

        out_latency = convert_latency_pool([raw_latency], n_neurons, self.pool_size)
        y_pred = predict_from_latency_pool([out_latency[0]])
        return int(y_pred[0])

    def predict_from_latency_pool(self, latency):
        return predict_from_latency_pool(latency)

    def learn(self, action, correct_class):
        hidden_learning_rate = (
            self.learning_rate_hid
            if self.last_hidden_goodness
            else -0.1 * self.learning_rate_hid
        )

        self.update_learning_rate(0)
        self.update_learning_rate_hid(hidden_learning_rate)
        self.update_learning_rate_separate_out(self.learning_rate, action)
        self.update_learning_rate_separate_out(self.learning_rate_scaled, correct_class)

        nest.Run(self.h_time)
        self.time += self.h_time

# coding: utf-8

import nest

from .epoch import EpochNetwork


class TwoLayerNetwork(EpochNetwork):
    def __init__(self, settings, model, **kwargs):
        super().__init__(settings, model, **kwargs)
        self.n_layer_hid = kwargs.get(
            'n_layer_hid', settings['topology']['n_layer_hid']
        )
        self.use_reciprocal = kwargs.get(
            'use_reciprocal', False
        )
        self.w_target_hid = kwargs.get('w_target_hid') or 1
        self.w_targets = [self.w_target_hid, self.w_target]
        self.synapse_models = [
            self.model['syn_dict_exc_hid']['synapse_model'],
            self.model['syn_dict_exc']['synapse_model'],
        ]

    def create_layers(self):
        self.layer_out = nest.Create(self.model['neuron_out'], self.n_layer_out)
        self.layer_hid = nest.Create(self.model['neuron_hid'], self.n_layer_hid)
        self.input_layer = nest.Create('parrot_neuron', self.n_input)
        self.layers = [self.input_layer, self.layer_hid, self.layer_out]

    def create_devices(self):
        super().create_devices()
        self.spike_detector_hidden = nest.Create('spike_recorder')

        self.voltmeter_hidden = nest.Create(
            'multimeter', 1, {'withgid': True, 'withtime': True}
        )

    def connect_layers(self):
        nest.Connect(
            self.input_layer,
            self.layer_hid,
            'all_to_all',
            syn_spec=self.model['syn_dict_exc_hid'],
        )
        nest.Connect(
            self.layer_hid,
            self.layer_out,
            'all_to_all',
            syn_spec=self.model['syn_dict_exc'],
        )
        if self.use_reciprocal:
            nest.Connect(
                self.layer_out,
                self.layer_hid,
                'all_to_all',
                syn_spec=self.model['syn_dict_rec'],
            )

    def connect_layers_static(self):
        nest.Connect(
            self.input_layer, self.layer_hid, 'all_to_all', syn_spec='static_synapse'
        )
        nest.Connect(
            self.layer_hid, self.layer_out, 'all_to_all', syn_spec='static_synapse'
        )

    def connect_layers_inh(self):
        super().connect_layers_inh()
        self.interconnect_layer(
            self.layer_hid,
            self.model['syn_dict_inh_hid'],
            self.model.get('conn_dict_inh_hid', {'rule': 'all_to_all'}),
        )

    def connect_devices(self):
        super().connect_devices()
        nest.Connect(self.layer_hid, self.spike_detector_hidden, 'all_to_all')
        nest.Connect(self.voltmeter_hidden, self.layer_hid)

    def get_devices(self):
        devices = {
            'multimeter': nest.GetStatus(self.multimeter, keys='events')[0],
            'multimeter_hidden': nest.GetStatus(self.voltmeter_hidden, keys='events')[
                0
            ],
            'spike_detector_out': nest.GetStatus(
                self.spike_detector_out, keys='events'
            )[0],
            'spike_detector_input': nest.GetStatus(
                self.spike_detector_input, keys='events'
            )[0],
            'spike_detector_hidden': nest.GetStatus(
                self.spike_detector_hidden, keys='events'
            )[0],
        }
        return devices

    def set_neuron_status(self, override_threshold):
        super().set_neuron_status(override_threshold)
        nest.SetStatus(self.layer_hid, self.model['neuron_hid'])

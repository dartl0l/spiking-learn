# coding: utf-8

import nest

from .convolution import ConvolutionNetwork
from .rl import LiteRlNetwork


class ConvolutionRlNetwork(LiteRlNetwork, ConvolutionNetwork):
    def __init__(
        self,
        settings,
        model,
        n_layer_hid,
        kernel_size,
        stride,
        learning_rate,
        **kwargs
    ):
        self.n_layer_hid = n_layer_hid
        self.layer_hid = None
        self.spike_detector_hid = None
        self.multimeter_hid = None
        self.w_target_hid = kwargs.get('w_target_hid') or 1

        super().__init__(
            settings=settings,
            model=model,
            kernel_size=kernel_size,
            stride=stride,
            learning_rate=learning_rate,
            **kwargs
        )
        self.w_targets = [self.w_target_hid, self.w_target]
        self.synapse_models = [
            self.model['syn_dict_exc_hid']['synapse_model'],
            self.model['syn_dict_exc']['synapse_model']
        ]

    def create_layers(self):
        self.layer_out = nest.Create(self.model['neuron_out_model'], self.n_layer_out)
        self.layer_hid = nest.Create(self.model['neuron_hid_model'], self.n_layer_hid)
        self.input_layer = nest.Create('parrot_neuron', self.n_input)
        self.layers = [self.input_layer, self.layer_hid, self.layer_out]

    def create_devices(self):
        super().create_devices()
        if self.need_devices:
            self.spike_detector_hid = nest.Create('spike_recorder')
            self.multimeter_hid = nest.Create('multimeter', 1, {'record_from': ['V_m']})

    def connect_devices(self):
        super().connect_devices()
        if self.spike_detector_hid:
            nest.Connect(
                self.layer_hid,
                self.spike_detector_hid,
                conn_spec={'rule': 'all_to_all'},
            )
        if self.multimeter_hid:
            nest.Connect(self.multimeter_hid, self.layer_hid)

    def connect_layers(self):
        self.connect_exc(self.model['syn_dict_exc_hid'])
        nest.Connect(
            self.layer_hid,
            self.layer_out,
            conn_spec=self.model.get('conn_dict_exc', {'rule': 'all_to_all'}),
            syn_spec=self.model['syn_dict_exc'],
        )

    def connect_layers_static(self):
        self.connect_exc('static_synapse')
        nest.Connect(
            self.layer_hid,
            self.layer_out,
            conn_spec=self.model.get('conn_dict_exc', {'rule': 'all_to_all'}),
            syn_spec='static_synapse',
        )

    def connect_layers_inh(self):
        self.connect_inh(
            self.model['syn_dict_inh_hid'],
            self.model.get('conn_dict_inh_hid', {'rule': 'all_to_all'})
        )
        self.interconnect_layer(
            self.layer_out,
            self.model['syn_dict_inh'],
            self.model.get('conn_dict_inh', {'rule': 'all_to_all'}),
        )

    def get_neuron_number(self):
        return self.n_layer_hid

    def get_layer_conv(self):
        return self.layer_hid

    def get_previous_layer(self):
        return self.layer_hid

    def get_devices(self):
        devices = {
            'spike_detector_out': nest.GetStatus(
                self.spike_detector_out, keys='events'
            )[0]
        }
        if self.multimeter:
            devices['multimeter'] = nest.GetStatus(self.multimeter, keys='events')[0]
        if self.multimeter_hid:
            devices['multimeter_hidden'] = nest.GetStatus(self.multimeter_hid, keys='events')[0]
        if self.spike_detector_input:
            devices['spike_detector_input'] = nest.GetStatus(
                self.spike_detector_input, keys='events'
            )[0]
        if self.spike_detector_hid:
            devices['spike_detector_hidden'] = nest.GetStatus(
                self.spike_detector_hid, keys='events'
            )[0]
        return devices

    def set_neuron_status(self, override_threshold=False):
        super().set_neuron_status(override_threshold)
        nest.SetStatus(self.layer_hid, self.model['neuron_hid'])

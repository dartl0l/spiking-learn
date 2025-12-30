# coding: utf-8

import copy
import os
from typing import Optional

import nest
import numpy as np
from ..noise import NoiseGenerator
from ..spike_generator import TemporalSpikeGenerator
from ..teacher import Teacher

nest.set_verbosity('M_QUIET')

if 'NEST_MODULES' not in os.environ:
    nest.Install('stdptanhmodule')


class Network:
    """base class for different network types"""

    def __init__(
        self,
        settings,
        model,
        n_layer_out: Optional[int] = None,
        n_input: Optional[int] = None,
        epochs: Optional[int] = None,
        spike_generator: Optional[TemporalSpikeGenerator] = None,
        teacher: Optional[Teacher] = None,
        noise: Optional[NoiseGenerator] = None,
        h_time: Optional[float] = None,
        start_delta: Optional[float] = None,
        h: Optional[float] = None,
        need_devices: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        self.model = copy.deepcopy(model)
        self.teacher = teacher
        self.noise = noise
        self.need_devices = need_devices
        self.verbose = verbose

        network_config = settings['network']
        topology_config = settings['topology']
        learning_config = settings['learning']

        self.h = h or network_config.get('h', 0.01)
        self.h_time = h_time or network_config.get('h_time', 50)
        self.start_delta = start_delta or network_config.get('start_delta', 50)

        self.synapse_models = [self.model['syn_dict_exc']['synapse_model']]

        self.n_procs = kwargs.get('num_procs', network_config.get('num_procs', 1))
        self.n_threads = kwargs.get('num_threads', network_config.get('num_threads', 1))

        self.test_with_inhibition = kwargs.get(
            'test_with_inhibition', network_config.get('test_with_inhibition', False)
        )

        self.use_inhibition = kwargs.get(
            'use_inhibition', topology_config.get('use_inhibition', True)
        )
        self.n_layer_out = n_layer_out or topology_config.get('n_layer_out', 2)
        self.n_input = n_input or kwargs.get(
            'n_input', topology_config.get('n_input', 2)
        )

        self.epochs = epochs or learning_config.get('epochs', 1)
        self.learning_threshold = kwargs.get(
            'threshold', learning_config.get('threshold', None)
        )
        self.high_threshold_teacher = kwargs.get(
            'high_threshold_teacher',
            learning_config.get('high_threshold_teacher', False),
        )

        self.spike_generator = spike_generator or TemporalSpikeGenerator(
            self.n_input, self.epochs, self.h_time
        )

        if self.verbose:
            print(self.__dict__)

        self.tile_train = True
        self.data_len = None
        self.input_generators = None
        self.spike_detector_out = None
        self.spike_detector_input = None
        self.multimeter = None

    def _create_parameters(self, parameters):
        for parameter in parameters:
            if not isinstance(parameters[parameter], str):
                for param in parameters[parameter]:
                    if isinstance(parameters[parameter][param], dict):
                        if (
                            'type' in parameters[parameter][param]
                            and 'specs' in parameters[parameter][param]
                        ):
                            parameters[parameter][param] = nest.CreateParameter(
                                parameters[parameter][param]['type'],
                                parameters[parameter][param]['specs'],
                            )

    def reset_spike_detectors(self):
        if self.spike_detector_input:
            nest.SetStatus(self.spike_detector_input, {'n_events': 0})
        nest.SetStatus(self.spike_detector_out, {'n_events': 0})

    def reset_voltmeter(self):
        nest.SetStatus(self.multimeter, {'n_events': 0})

    def interconnect_layer(self, layer, syn_dict, conn_spec):
        for neuron_1 in layer:
            for neuron_2 in layer:
                if neuron_1 != neuron_2:
                    nest.Connect(
                        neuron_1, neuron_2, syn_spec=syn_dict, conn_spec=conn_spec
                    )

    def get_devices(self):
        devices = {
            'spike_detector_out': nest.GetStatus(
                self.spike_detector_out, keys='events'
            )[0]
        }
        if self.multimeter:
            devices['multimeter'] = nest.GetStatus(self.multimeter, keys='events')[0]
        if self.spike_detector_input:
            devices['spike_detector_input'] = nest.GetStatus(
                self.spike_detector_input, keys='events'
            )[0]
        return devices

    def get_spikes_of_pattern(self, spike_recorder, estimated_time, example_class):
        spikes = nest.GetStatus(spike_recorder, keys='events')[0]['times']
        senders = nest.GetStatus(spike_recorder, keys='events')[0]['senders']
        mask = spikes > estimated_time
        spikes = spikes[mask]
        senders = senders[mask]
        tmp_dict = {
            'latency': spikes - estimated_time,
            'senders': senders,
            'class': example_class,
        }
        return tmp_dict

    def save_weights(self, layers, synapse_models):
        weights = [None] * len(layers[1:])

        for i, layer in enumerate(layers[1:]):
            synapse_model = synapse_models[i]
            previous_layer = layers[i]
            weights[i] = {}
            for neuron_id in layer.tolist():
                weights[i][neuron_id] = {}
                for input_id in previous_layer.tolist():
                    conn = nest.GetConnections(
                        nest.NodeCollection([input_id]),
                        nest.NodeCollection([neuron_id]),
                        synapse_model=synapse_model,
                    )
                    weight_one = nest.GetStatus(conn, 'weight')
                    if len(weight_one) != 0:
                        weights[i][neuron_id][input_id] = weight_one[0]
                    else:
                        weights[i][neuron_id][input_id] = None
        return weights

    def init_network(self):
        np.random.seed()
        rng = np.random.randint(1, pow(2, 32) - 1)
        num_v_procs = self.n_threads * self.n_procs

        nest.ResetKernel()
        nest.SetKernelStatus(
            {
                'local_num_threads': self.n_threads,
                'total_num_virtual_procs': num_v_procs,
                'resolution': self.h,
            }
        )

        nest.rng_seed = rng
        self._create_parameters(self.model)

    def create_layers(self):
        self.layer_out = nest.Create(self.model['neuron_out_model'], self.n_layer_out)
        self.input_layer = nest.Create('parrot_neuron', self.n_input)
        self.layers = [self.input_layer, self.layer_out]

    def create_devices(self):
        self.spike_detector_out = nest.Create('spike_recorder')

        if self.need_devices:
            self.spike_detector_input = nest.Create('spike_recorder')

            self.multimeter = nest.Create('multimeter', 1, {'record_from': ['V_m']})

    def connect_devices(self):
        nest.Connect(
            self.layer_out, self.spike_detector_out, conn_spec={'rule': 'all_to_all'}
        )

        if self.spike_detector_input:
            nest.Connect(
                self.input_layer,
                self.spike_detector_input,
                conn_spec={'rule': 'all_to_all'},
            )
        if self.multimeter:
            nest.Connect(self.multimeter, self.layer_out)

    def connect_layers(self):
        nest.Connect(
            self.input_layer,
            self.layer_out,
            conn_spec=self.model.get('conn_dict_exc', {'rule': 'all_to_all'}),
            syn_spec=self.model['syn_dict_exc'],
        )

    def connect_layers_static(self):
        nest.Connect(
            self.input_layer,
            self.layer_out,
            conn_spec=self.model.get('conn_dict_exc', {'rule': 'all_to_all'}),
            syn_spec='static_synapse',
        )

    def connect_layers_inh(self):
        self.interconnect_layer(
            self.layer_out,
            self.model['syn_dict_inh'],
            self.model.get('conn_dict_inh', {'rule': 'all_to_all'}),
        )

    def set_neuron_status(self, override_threshold=False):
        nest.SetStatus(self.layer_out, self.model['neuron_out'])

        if override_threshold and self.learning_threshold:
            nest.SetStatus(self.layer_out, {'V_th': self.learning_threshold})

    def set_neurons_status(self, neuron_id, model):
        nest.SetStatus(self.layer_out[neuron_id], model)

    def set_weights(self, weights):
        for layer_weights in weights:
            for neuron_id in layer_weights:
                for input_id in layer_weights[neuron_id]:
                    connection = nest.GetConnections(
                        nest.NodeCollection([input_id]),
                        target=nest.NodeCollection([neuron_id]),
                    )
                    neuron_weights = layer_weights[neuron_id][input_id]
                    if neuron_weights is not None:
                        nest.SetStatus(connection, 'weight', neuron_weights)

    def simulate(self, full_time, spike_dict, teacher_dicts=None, epochs=1):
        if self.spike_generator:
            self.spike_generator.set_input_spikes(spike_dict=spike_dict)

        if self.teacher:
            self.teacher.set_teachers_input(teacher_dicts)

        nest.Simulate(full_time)

    def train(self, x, y=None):
        self.init_network()
        self.create_layers()
        self.create_devices()

        self.connect_devices()
        self.connect_layers()

        if self.use_inhibition:
            self.connect_layers_inh()

        self.set_neuron_status(self.high_threshold_teacher)

        self.data_len = len(x)

        spike_dict = None
        full_time = None
        if self.spike_generator:
            self.spike_generator.create_devices()
            self.spike_generator.connect_devices(self.input_layer)

            spike_dict, full_time = self.spike_generator.create_spike_dict(
                dataset=x, tile=self.tile_train, delta=self.start_delta
            )

        teacher_dicts = None
        if self.teacher:
            self.teacher.create_teacher_layer()
            self.teacher.connect_teacher(self.layer_out)
            teacher_dicts = self.teacher.create_teacher(input_spikes=x, classes=y)

        if self.noise:
            self.noise.create()
            self.noise.connect(self.input_layer)
            self.noise.set(x, tile=True)

        self.simulate(full_time, spike_dict, teacher_dicts, self.epochs)

        weights = self.save_weights(self.layers, self.synapse_models)
        output = {
            'spikes': nest.GetStatus(self.spike_detector_out, keys='events')[0][
                'times'
            ].tolist(),
            'senders': nest.GetStatus(self.spike_detector_out, keys='events')[0][
                'senders'
            ].tolist(),
        }
        devices = self.get_devices() if self.need_devices else None
        return weights, output, devices

    def test(self, x, weights):
        self.init_network()
        self.create_layers()
        self.create_devices()

        self.connect_devices()
        self.connect_layers_static()
        if self.use_inhibition and self.test_with_inhibition:
            self.connect_layers_inh()

        self.set_neuron_status()
        self.set_weights(weights)

        self.data_len = len(x)
        if self.spike_generator:
            self.spike_generator.create_devices()
            self.spike_generator.connect_devices(self.input_layer)

            spike_dict, full_time = self.spike_generator.create_spike_dict(
                dataset=x, tile=False, delta=self.start_delta
            )

        if self.noise and self.noise.test_with_noise:
            self.noise.create()
            self.noise.connect(self.input_layer)
            self.noise.set(x)

        self.simulate(full_time, spike_dict)

        output = {
            'spikes': nest.GetStatus(self.spike_detector_out, keys='events')[0][
                'times'
            ].tolist(),
            'senders': nest.GetStatus(self.spike_detector_out, keys='events')[0][
                'senders'
            ].tolist(),
        }

        devices = self.get_devices() if self.need_devices else None
        return output, devices

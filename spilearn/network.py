# coding: utf-8

import nest
import math
import copy
import numpy as np

from tqdm import trange

nest.set_verbosity('M_QUIET')

class Network:
    """base class for different network types"""
    def __init__(self, settings, model, teacher=None, **kwargs):
        self.model = copy.deepcopy(model)
        self.teacher = teacher
        self.synapse_models = [self.model['syn_dict_exc']['synapse_model']]

        self.h = kwargs.get('h', settings['network']['h'])
        self.n_procs = kwargs.get('num_procs', settings['network']['num_procs'])
        self.n_threads = kwargs.get('num_threads', settings['network']['num_threads'])
        
        self.h_time = kwargs.get('h_time', settings['network']['h_time'])
        self.start_delta = kwargs.get('start_delta', settings['network']['start_delta'])

        self.noise_freq = kwargs.get('noise_freq', settings['network']['noise_freq'])
        self.test_with_noise = kwargs.get('test_with_noise', settings['network']['test_with_noise'])
        self.noise_after_pattern = kwargs.get('noise_after_pattern', settings['network']['noise_after_pattern'])
        self.test_with_inhibition = kwargs.get('test_with_inhibition', settings['network']['test_with_inhibition'])

        self.use_inhibition = kwargs.get('use_inhibition', settings['topology']['use_inhibition'])
        self.n_layer_out = kwargs.get('n_layer_out', settings['topology']['n_layer_out'])
        self.n_input = kwargs.get('n_input', settings['topology']['n_input'])

        self.epochs = kwargs.get('epochs', settings['learning']['epochs'])
        self.learning_threshold = kwargs.get('threshold', settings['learning']['threshold'])
        self.high_threshold_teacher = kwargs.get('high_threshold_teacher',  settings['learning']['high_threshold_teacher'])

    def _create_parameters(self, parameters):
        for parameter in parameters:
            if not isinstance(parameters[parameter], str):
                for param in parameters[parameter]:
                    if isinstance(parameters[parameter][param], dict):
                        if 'type' in parameters[parameter][param] and 'specs' in parameters[parameter][param]:
                            parameters[parameter][param] = nest.CreateParameter(
                                parameters[parameter][param]['type'],
                                parameters[parameter][param]['specs'],
                            )

    def reset_spike_detectors(self):
        nest.SetStatus(self.spike_detector_input, {'n_events': 0})
        nest.SetStatus(self.spike_detector_out, {'n_events': 0})

    def reset_voltmeter(self):
        nest.SetStatus(self.multimeter, {'n_events': 0})
    
    def reset_teachers(self):
        self.teacher_layer.set({
                    'amplitude_times': [],
                    'amplitude_values': []
                })

    def disconnect_layers(self):
        layers_conn = nest.GetConnections(
            self.input_layer,
            self.layer_out, 
            self.model['syn_dict_exc']['synapse_model'])
        if layers_conn:
            nest.Disconnect(
                self.input_layer,
                self.layer_out, 'all_to_all',
                syn_spec=self.model['syn_dict_exc'])

    def disconnect_layers_static(self):
        layers_static_conn = nest.GetConnections(
            self.input_layer,
            self.layer_out, 
            'static_synapse')
        if layers_static_conn:
            nest.Disconnect(
                self.input_layer,
                self.layer_out, 'all_to_all',
                syn_spec='static_synapse')

    def set_input_spikes(self, spike_dict, spike_generators):
        assert len(spike_dict) == len(spike_generators)
        nest.SetStatus(spike_generators, spike_dict)

    def set_teachers_input(self, teacher_dicts):
        self.teacher_layer.set(list(teacher_dicts.values()))

    def set_poisson_noise(self, noise_dict, spike_generators):
        nest.SetStatus(spike_generators, noise_dict)

    def create_spike_dict(self, dataset, train, delta=0):
        delta = self.start_delta
        epochs = self.epochs if train else 1
        pattern_start_shape = (len(dataset[0]), 1)
        spikes = np.tile(dataset, (epochs, 1, 1))
        full_length = epochs * len(dataset)
        full_time = full_length * self.h_time + delta

        assert len(spikes) == full_length
        spike_times = self.create_spike_times(
            spikes, delta, full_time, pattern_start_shape
        )

        spike_dict = [None] * len(dataset[0])
        for input_neuron in range(len(dataset[0])):
            tmp_spikes = spike_times[:, input_neuron].reshape(len(spikes))
            spike_dict[input_neuron] = {
                'spike_times': tmp_spikes[np.isfinite(tmp_spikes)]
            }
        return spike_dict, full_time, spikes

    def create_spike_times(self, current_spikes, start, end, 
                           pattern_start_shape):
        times = np.arange(start, end, self.h_time)
        pattern_start_times = np.expand_dims(
            np.tile(times, pattern_start_shape).T, axis=2)
        assert len(current_spikes) == len(pattern_start_times)
        spike_times = np.add(current_spikes, pattern_start_times)
        return spike_times

    def create_poisson_noise(self, spikes_list):  # Network
        dt = 1.0 / 1000.0
        h = self.h
        h_time = self.h_time
        epochs = self.epochs
        start = self.start_delta
        noise_frequency = self.noise_freq

        num_inputs = len(spikes_list[0])

        num_total_patterns = epochs * len(spikes_list)
        end = num_total_patterns * h_time + start

        start_pattern_times = np.linspace(start, end, num_total_patterns,
                                          endpoint=False)
        end_pattern_times = start_pattern_times + h_time

        start_noise_times = np.tile(
            np.amax(spikes_list, axis=1), 
            epochs
        ).reshape(start_pattern_times.shape) + start_pattern_times

        num_per_pattern = int((end_pattern_times[0] - start_noise_times[0]) / h)
        times = np.linspace(
            start_noise_times,
            end_pattern_times,
            num_per_pattern,
            endpoint=False, axis=1).ravel()

        # np.random.random_sample is slow
        noise_dict = [None] * num_inputs
        for input_neuron in range(num_inputs):
            random_distribution_mask = \
                np.random.random_sample(len(times)) < noise_frequency * dt
            noise_dict[input_neuron] = {
                'spike_times': times[random_distribution_mask]
            }
        return noise_dict

    def interconnect_layer(self, layer, syn_dict):
        for neuron_1 in layer:
            for neuron_2 in layer:
                if neuron_1 != neuron_2:
                    nest.Connect(neuron_1, neuron_2, syn_spec=syn_dict)

    def get_devices(self):
        devices = {
                    'multimeter': nest.GetStatus(self.multimeter,
                                                keys="events")[0],
                    'spike_detector_out': nest.GetStatus(
                        self.spike_detector_out, keys="events")[0],
                    'spike_detector_input': nest.GetStatus(
                        self.spike_detector_input, keys="events")[0],
                   }
        return devices

    def get_spikes_of_pattern(self, spike_recorder, estimated_time, 
                              example_class):
        spikes = nest.GetStatus(spike_recorder, keys="events")[0]['times']
        senders = nest.GetStatus(spike_recorder, keys="events")[0]['senders']
        mask = spikes > estimated_time
        spikes = spikes[mask]
        senders = senders[mask]
        tmp_dict = {
                    'latency': spikes - estimated_time,
                    'senders': senders,
                    'class': example_class
                    }
        return tmp_dict

    def save_weights(self, layers):
        synapse_models = self.synapse_models

        weights = [None] * len(self.layers[1:])

        for i, layer in enumerate(self.layers[1:]):
            synapse_model = synapse_models[i]
            previous_layer = self.layers[i]
            weights[i] = {}
            for neuron_id in layer.tolist():
                tmp_weight = []
                for input_id in previous_layer.tolist():
                    conn = nest.GetConnections(
                        nest.NodeCollection([input_id]), 
                        nest.NodeCollection([neuron_id]), 
                        synapse_model=synapse_model
                    )
                    weight_one = nest.GetStatus(conn, 'weight')
                    if len(weight_one) != 0:
                        tmp_weight.append(weight_one[0])
                if len(tmp_weight) != 0:
                    weights[i][neuron_id] = tmp_weight
        return weights

    def init_network(self):
        np.random.seed()
        rng = np.random.randint(1, pow(2, 32) - 1)
        num_v_procs = self.n_threads * self.n_procs

        nest.ResetKernel()
        nest.SetKernelStatus({
             'local_num_threads': self.n_threads,
             'total_num_virtual_procs': num_v_procs,
             'resolution': self.h
        })

        nest.rng_seed = rng
        self._create_parameters(self.model)

    def create_layers(self):
        self.layer_out = nest.Create(
            self.model['neuron_out_model'],
            self.n_layer_out
        )
        self.input_layer = nest.Create(
            'parrot_neuron',
            self.n_input
        )
        self.layers = [self.input_layer, self.layer_out]

    def create_devices(self):
        self.teacher_layer = nest.Create(
            'step_current_generator', 
            self.n_layer_out)

        self.input_generators = nest.Create(
            'spike_generator', 
            self.n_input)

        self.interpattern_noise_generator = nest.Create(
            'spike_generator', 
            self.n_input)

        self.poisson_layer = nest.Create(
            'poisson_generator', 
            self.n_input)

        self.spike_detector_out = nest.Create('spike_recorder')
        self.spike_detector_input = nest.Create('spike_recorder')

        self.multimeter = nest.Create(
            'multimeter', 1,
            {'record_from': ['V_m']}
        )

    def connect_devices(self):
        nest.Connect(self.input_generators, 
                     self.input_layer, 'one_to_one', 
                     syn_spec='static_synapse')
        nest.Connect(self.poisson_layer,
                     self.input_layer, 'one_to_one',
                     syn_spec='static_synapse')

        nest.Connect(self.layer_out,
                     self.spike_detector_out,
                     'all_to_all')
        nest.Connect(self.input_layer,
                     self.spike_detector_input,
                     'all_to_all')
        nest.Connect(self.multimeter,
                     self.layer_out)

    def connect_teacher(self):
        nest.Connect(self.teacher_layer,
                     self.layer_out, 
                     'one_to_one',
                     syn_spec='static_synapse')

    def connect_layers(self):
        nest.Connect(self.input_layer,
                     self.layer_out, 
                     conn_spec={'rule': 'all_to_all'},
                     syn_spec=self.model['syn_dict_exc'])

    def connect_layers_static(self):
        nest.Connect(self.input_layer,
                     self.layer_out, 
                     'all_to_all',
                     syn_spec='static_synapse')

    def connect_layers_inh(self):
        self.interconnect_layer(self.layer_out,
                                self.model['syn_dict_inh'])

    def set_neuron_status(self):
        nest.SetStatus(self.layer_out,
                       self.model['neuron_out'])

        if self.high_threshold_teacher:
            nest.SetStatus(
                self.layer_out,
                {'V_th': self.learning_threshold})

    def set_noise(self):
        nest.SetStatus(
            self.poisson_layer,
            {
             'rate': self.noise_freq,
             'origin': 0.0
            }
        )

    def set_weights(self, weights):
        for layer_weights in weights:
            for neuron_id in layer_weights:
                connection = nest.GetConnections(
                    self.input_layer, target=nest.NodeCollection([neuron_id]))
                neuron_weights = layer_weights[neuron_id]
                nest.SetStatus(connection, 'weight', neuron_weights)

    def simulate(self, full_time, spike_dict, teacher_dicts=None):
        self.set_input_spikes(
            spike_dict=spike_dict,
            spike_generators=self.input_generators)

        if self.teacher:
            self.set_teachers_input(
                teacher_dicts)

        nest.Simulate(full_time)

    def train(self, x, y=None):
        self.init_network()
        self.create_layers()
        self.create_devices()

        self.connect_devices()
        self.connect_teacher()
        self.connect_layers()
        if self.use_inhibition:
            self.connect_layers_inh()

        self.set_neuron_status()

        spike_dict, full_time, input_spikes = self.create_spike_dict(
            dataset=x, train=True,
            delta=self.start_delta)

        teacher_dicts = self.teacher.create_teacher(
            input_spikes=x,
            classes=y,
            teachers=self.teacher_layer) if self.teacher else None

        if self.noise_after_pattern:
            noise_dict = self.create_poisson_noise(input_spikes)
            self.set_poisson_noise(
                noise_dict,
                self.interpattern_noise_generator)
        else:
            self.set_noise()

        self.simulate(full_time, spike_dict, teacher_dicts)

        weights = self.save_weights(self.layers)
        output = {
                  'spikes': nest.GetStatus(
                                self.spike_detector_out,
                                keys="events")[0]['times'].tolist(),
                  'senders': nest.GetStatus(
                                self.spike_detector_out,
                                keys="events")[0]['senders'].tolist()
                 }
        devices = self.get_devices()
        return weights, output, devices

    def test(self, x, weights):
        self.init_network()
        self.create_layers()
        self.create_devices()

        self.connect_devices()
        self.connect_layers_static()
        if self.use_inhibition \
                and self.test_with_inhibition:
            self.connect_layers_inh()

        self.set_neuron_status()
        if self.test_with_noise:
            self.set_noise()
        self.set_weights(weights)

        spike_dict, full_time, input_spikes = self.create_spike_dict(
            dataset=x, train=False,
            delta=self.start_delta)
        self.set_input_spikes(
            spike_dict=spike_dict,
            spike_generators=self.input_generators)
        
        nest.Simulate(full_time)

        output = {
                  'spikes': nest.GetStatus(
                                self.spike_detector_out,
                                keys="events")[0]['times'].tolist(),
                  'senders': nest.GetStatus(
                                self.spike_detector_out,
                                keys="events")[0]['senders'].tolist()
                 }

        devices = self.get_devices()
        return output, devices


class EpochNetwork(Network):
    def __init__(self, settings, model, teacher=None, progress=True, **kwargs):
        super().__init__(settings, model, teacher, **kwargs)
        self.progress = progress
        self.normalize_weights = kwargs.get('normalize_weights', False)

    def normalize(self, w_target=1):
        weights_all = self.save_weights(self.layers)
        for weights_of_split in weights_all:
            for layer_weights in weights_of_split:
                for neuron in layer_weights:
                    w = np.array(layer_weights[neuron])
                    w_norm = w / sum(abs(w))
                    w = w_target * w_norm
                    layer_weights[neuron] = w
        self.set_weights(weights_all)

    def create_spike_dict(self, dataset, delta=0.0):
        pattern_start_shape = (len(dataset[0]), 1)
        full_time = len(dataset) * self.h_time + delta

        spike_times = self.create_spike_times(
            dataset, delta, full_time, pattern_start_shape
        )

        spike_dict = [None] * len(dataset[0])
        for input_neuron in range(len(dataset[0])):
            tmp_spikes = spike_times[:, input_neuron]
            spike_dict[input_neuron] = {
                'spike_times': tmp_spikes[np.isfinite(tmp_spikes)]
            }
        return spike_dict, full_time

    def create_spikes(self, x, y):
        spike_dict, full_time = self.create_spike_dict(
            dataset=x,
            delta=self.start_delta)

        teacher_dicts = self.teacher.create_teacher(
            input_spikes=x,
            classes=y,
            teachers=self.teacher_layer) if self.teacher else None

        return full_time, spike_dict, teacher_dicts

    def simulate(self, full_time, spike_dict, teacher_dicts=None):
        t = trange(self.epochs) if self.progress \
            else range(self.epochs)
        for epoch in t:
            self.set_input_spikes(
                spike_dict=spike_dict,
                spike_generators=self.input_generators)
            if self.teacher:
                self.set_teachers_input(
                    teacher_dicts)
            if self.normalize_weights:
                nest.Simulate(self.start_delta)
                for _ in spike_dict:
                    nest.Simulate(self.h_time)
                    self.normalize()
            else:
                nest.Simulate(full_time)
            
            for spikes in spike_dict:
                spikes['spike_times'] += full_time
            if self.teacher:
                for teacher in teacher_dicts:
                    teacher_dicts[teacher]['amplitude_times'] += full_time
    
    def train(self, x, y=None):
        self.init_network()
        self.create_layers()
        self.create_devices()

        self.connect_devices()
        self.connect_teacher()
        self.connect_layers()
        if self.use_inhibition:
            self.connect_layers_inh()

        self.set_neuron_status()

        full_time, spike_dict, teacher_dicts = self.create_spikes(x, y)

        if self.noise_after_pattern:
            noise_dict = self.create_poisson_noise(spike_dict)
            self.set_poisson_noise(
                noise_dict,
                self.interpattern_noise_generator)
        else:
            self.set_noise()

        self.simulate(full_time, spike_dict, teacher_dicts)

        weights = self.save_weights(self.layers)
        output = {
                  'spikes': nest.GetStatus(
                                self.spike_detector_out,
                                keys="events")[0]['times'].tolist(),
                  'senders': nest.GetStatus(
                                self.spike_detector_out,
                                keys="events")[0]['senders'].tolist()
                 }
        devices = self.get_devices()
        return weights, output, devices

    def test(self, x, weights):
        self.init_network()
        self.create_layers()
        self.create_devices()

        self.connect_devices()
        self.connect_layers_static()
        if self.use_inhibition \
                and self.test_with_inhibition:
            self.connect_layers_inh()
        self.set_neuron_status()
        if self.test_with_noise:
            self.set_noise()
        self.set_weights(weights)

        spike_dict, full_time = self.create_spike_dict(
            dataset=x,
            delta=self.start_delta)
        self.set_input_spikes(
            spike_dict=spike_dict,
            spike_generators=self.input_generators)

        nest.Simulate(full_time)

        output = {
          'spikes': nest.GetStatus(
                        self.spike_detector_out,
                        keys="events")[0]['times'].tolist(),
          'senders': nest.GetStatus(
                         self.spike_detector_out,
                         keys="events")[0]['senders'].tolist()
         }

        devices = self.get_devices()
        return output, devices


class NotSoFastEpochNetwork(EpochNetwork):
    def __init__(self, settings, model, teacher=None, progress=True, **kwargs):
        super().__init__(settings, model, teacher, progress, **kwargs)
        self.init_network()
        self.create_layers()
        self.create_devices()
        
        self.connect_devices()
        self.time_elapsed = 0

    def get_devices(self):
        multimeter = nest.GetStatus(
            self.multimeter, keys="events")[0]
        multimeter['times'] -= self.time_elapsed
        
        spike_detector_out = nest.GetStatus(
            self.spike_detector_out, keys="events")[0]
        spike_detector_out['times'] -= self.time_elapsed
        
        spike_detector_input = nest.GetStatus(
            self.spike_detector_input, keys="events")[0]
        spike_detector_input['times'] -= self.time_elapsed
        
        devices = {
                    'multimeter': multimeter,
                    'spike_detector_out': spike_detector_out,
                    'spike_detector_input': spike_detector_input
                   }
        return devices

    def train(self, x, y):
        self.reset_spike_detectors()
        self.reset_voltmeter()
        self.reset_teachers()

        self.disconnect_layers()
        self.disconnect_layers_static()

        self.connect_teacher()
        self.connect_layers()
        
        if self.use_inhibition:
            self.connect_layers_inh()

        self.set_neuron_status()
        if not self.noise_after_pattern:
            self.set_noise()

        spike_dict, full_time = self.create_spike_dict(
            dataset=x,
            delta=self.start_delta)

        if self.teacher:
            teacher_dicts = self.teacher.create_teacher(
                input_spikes=x,
                classes=y,
                teachers=self.teacher_layer)
        if self.high_threshold_teacher:
            nest.SetStatus(
                self.layer_out, 
                {'V_th': self.learning_threshold})
        if self.noise_after_pattern:
            noise_dict = self.create_poisson_noise(x)
            self.set_poisson_noise(
                noise_dict,
                self.interpattern_noise_generator)

        for spikes in spike_dict:
            spikes['spike_times'] += self.time_elapsed
        if self.teacher:
            for teacher in teacher_dicts:
                teacher_dicts[teacher]['amplitude_times'] += self.time_elapsed

        t = trange(self.epochs) if self.progress \
            else range(self.epochs)
        time = 0
        for epoch in t:
            self.set_input_spikes(
                spike_dict=spike_dict,
                spike_generators=self.input_generators)
            if self.teacher:
                self.set_teachers_input(
                    teacher_dicts)
            nest.Simulate(full_time)
            time += full_time
            
            for spikes in spike_dict:
                spikes['spike_times'] += full_time
            if self.teacher:
                for teacher in teacher_dicts:
                    teacher_dicts[teacher]['amplitude_times'] += full_time

        weights = self.save_weights(self.layers)
        spikes = nest.GetStatus(
            self.spike_detector_out,
            keys="events")[0]['times']
        senders = nest.GetStatus(
             self.spike_detector_out,
             keys="events")[0]['senders']

        output = {
                  'spikes': spikes,
                  'senders': senders
                 }
        devices = self.get_devices()
        self.time_elapsed += time
        
        return weights, output, devices

    def test(self, x, weights):
        self.reset_spike_detectors()
        self.reset_voltmeter()
        self.reset_teachers()

        self.disconnect_layers()
        self.disconnect_layers_static()
        
        self.connect_layers_static()
        self.layers_static_connected = True
        if self.use_inhibition \
                and self.test_with_inhibition:
            self.connect_layers_inh()

        self.set_neuron_status()
        if self.test_with_noise:
            self.set_noise()
        self.set_weights(weights)

        spike_dict, full_time = self.create_spike_dict(
            dataset=x,
            delta=self.start_delta)
        
        for spikes in spike_dict:
            spikes['spike_times'] += self.time_elapsed
        
        self.set_input_spikes(
            spike_dict=spike_dict,
            spike_generators=self.input_generators)

        nest.Simulate(full_time)
       
        spikes = nest.GetStatus(
            self.spike_detector_out,
            keys="events")[0]['times'] - self.time_elapsed

        senders = nest.GetStatus(
             self.spike_detector_out,
             keys="events")[0]['senders']

        output = {
                  'spikes': spikes,
                  'senders': senders
                 }

        devices = self.get_devices()
        self.time_elapsed += full_time
        return output, devices


class ConvolutionNetwork(EpochNetwork):
    def __init__(self, settings, model, teacher=None, **kwargs):
        super().__init__(settings, model, teacher, **kwargs)
        self.kernel_size = kwargs.get('kernel_size', settings['topology']['convolution']['kernel_size'])
        self.stride = kwargs.get('stride', settings['topology']['convolution']['stride'])
        self.image_dimension = int(math.sqrt(self.n_input))
        self.n_combinations = (self.image_dimension - (self.kernel_size - self.stride)) ** 2
        self.n_combination_neurons = self.n_layer_out // self.n_combinations

        self.two_dimensional_image_indices = np.arange(
            self.n_input
        ).reshape(self.image_dimension, self.image_dimension)

    def connect_layers(self):
        current_combination = 0
        for image_row in range(0, self.image_dimension - self.kernel_size, self.stride):
            for image_column in range(0, self.image_dimension - self.kernel_size, self.stride):
                input_indexes = np.concatenate(self.two_dimensional_image_indices[
                    image_row: image_row + self.kernel_size,
                    image_column: image_column + self.kernel_size])
                input_indexes = np.array(self.input_layer)[input_indexes]
                output_indexes = np.array(self.layer_out[
                    current_combination : current_combination + self.n_combination_neurons])
                nest.Connect(input_indexes, output_indexes, 'all_to_all',
                             syn_spec=self.model['syn_dict_exc'])
                current_combination += self.n_combination_neurons

    def connect_layers_static(self):
        current_combination = 0
        for image_row in range(0, self.image_dimension - self.kernel_size, self.stride):
            for image_column in range(0, self.image_dimension - self.kernel_size, self.stride):
                input_indexes = np.concatenate(self.two_dimensional_image_indices[
                    image_row: image_row + self.kernel_size,
                    image_column: image_column + self.kernel_size])
                input_indexes = np.array(self.input_layer)[input_indexes]
                output_indexes = np.array(self.layer_out[
                    current_combination: current_combination + self.n_combination_neurons])
                nest.Connect(input_indexes, output_indexes, 'all_to_all',
                             syn_spec='static_synapse')
                current_combination += self.n_combination_neurons

    def connect_layers_inh(self):
        current_combination = 0
        for image_row in range(0, self.image_dimension - self.kernel_size, self.stride):
            for image_column in range(0, self.image_dimension - self.kernel_size, self.stride):
                input_indexes = np.concatenate(
                    self.two_dimensional_image_indices[
                        image_row : image_row + self.kernel_size,
                        image_column : image_column + self.kernel_size]
                )
                input_indexes = np.array(self.input_layer)[input_indexes]

                output_indexes = np.array(
                    self.layer_out[
                        current_combination:
                        current_combination + self.n_combination_neurons]
                )

                self.interconnect_layer(output_indexes,
                                        self.model['syn_dict_inh'])
                current_combination += self.n_combination_neurons


class TwoLayerNetwork(Network):
    def __init__(self, settings, model, teacher=None, **kwargs):
        super().__init__(settings, model, teacher, **kwargs)
        self.n_layer_hid = kwargs.get('n_layer_hid', settings['topology']['n_layer_hid'])
        self.use_reciprocal = kwargs.get('use_reciprocal', settings['topology']['use_reciprocal'])
        self.synapse_models = [self.model['syn_dict_exc_hid']['synapse_model'],
                               self.model['syn_dict_exc']['synapse_model']]

    def create_layers(self):
        self.layer_out = nest.Create(self.model['neuron_out_model'], 
                                     self.n_layer_out)
        self.layer_hid = nest.Create(self.model['neuron_hid_model'], 
                                     self.n_layer_hid)
        self.input_layer = nest.Create('parrot_neuron', 
                                       self.n_input)
        self.layers = [self.input_layer, self.layer_hid, self.layer_out]

    def create_devices(self):
        super().create_devices()
        self.spike_detector_hidden = nest.Create('spike_recorder')

        self.voltmeter_hidden = nest.Create(
            'multimeter', 1,
            {
             'withgid': True,
             'withtime': True
            }
        )

    def connect_layers(self):
        nest.Connect(self.input_layer,
                     self.layer_hid, 'all_to_all',
                     syn_spec=self.model['syn_dict_exc_hid'])
        nest.Connect(self.layer_hid,
                     self.layer_out, 'all_to_all',
                     syn_spec=self.model['syn_dict_exc'])
        if self.use_reciprocal:
            nest.Connect(self.layer_out,
                         self.layer_hid, 'all_to_all',
                         syn_spec=self.model['syn_dict_rec'])

    def connect_layers_static(self):
        nest.Connect(self.input_layer,
                     self.layer_hid, 'all_to_all', 
                     syn_spec='static_synapse')
        nest.Connect(self.layer_hid,
                     self.layer_out, 'all_to_all',
                     syn_spec='static_synapse')

    def connect_layers_inh(self):
        super().connect_layers_inh()
        self.interconnect_layer(self.layer_hid, 
                                self.model['syn_dict_inh_hid'])

    def connect_devices(self):
        super().connect_devices()
        nest.Connect(self.layer_hid,
                     self.spike_detector_hidden, 'all_to_all')
        nest.Connect(self.voltmeter_hidden,
                     self.layer_hid)

    def get_devices(self):
        devices = {
                    'multimeter': nest.GetStatus(self.multimeter,
                                                keys="events")[0],
                    'multimeter_hidden': nest.GetStatus(self.voltmeter_hidden,
                                                       keys="events")[0],
                    'spike_detector_out': nest.GetStatus(self.spike_detector_out,
                                                         keys="events")[0],
                    'spike_detector_input': nest.GetStatus(self.spike_detector_input,
                                                           keys="events")[0],
                    'spike_detector_hidden': nest.GetStatus(self.spike_detector_hidden,
                                                            keys="events")[0],
                   }
        return devices

    def set_neuron_status(self):
        super().set_neuron_status()
        nest.SetStatus(self.layer_hid,
                       self.model['neuron_hid'])

    def set_weights(self, weights):
        for neuron_id in weights['layer_0']:
            connection = nest.GetConnections(
                self.input_layer, target=[neuron_id])
            nest.SetStatus(connection, 'weight', 
                           weights['layer_0'][neuron_id])

        for neuron_id in weights['layer_1']:
            connection = nest.GetConnections(
                self.layer_hid, target=[neuron_id])
            nest.SetStatus(connection, 'weight',
                           weights['layer_1'][neuron_id])


class FrequencyNetwork(Network):
    """base class for different network types"""
    def __init__(self, settings, model, teacher=None, **kwargs):
        super().__init__(settings, model, teacher, **kwargs)
        self.pattern_length = kwargs.get('pattern_length', settings['data']['pattern_length'])
        self.synapse_models = [self.model['syn_dict_exc']['synapse_model']]

    def create_spike_dict(self, dataset, train, threads=48, delta=0.0):
        print("prepare spikes freq")
        spikes = []
        d_time = delta
        epochs = self.epochs if train else 1
        
        spike_dict = [None] * len(dataset[0])
        for input_neuron in range(len(dataset[0])):
            spike_dict[input_neuron] = {'spike_times': []}
            
        # TODO
        # calc spike times one time and concatenate
        for _ in range(epochs):
            for example in dataset:
                tmp_spikes = []
                for input_neuron in example:
                    spike_dict[input_neuron]['spike_times'] \
                        += map(lambda x: x + d_time, example[input_neuron])
                    tmp_spikes.append(example[input_neuron])
                spikes.append(tmp_spikes)
                d_time += self.h_time
        return spike_dict, d_time, spikes

    def create_poisson_noise(self, spikes_list):  # Network
        dt = 1.0 / 1000.0
        h = self.h
        h_time = self.h_time
        epochs = self.epochs
        start = self.start_delta
        pattern_length = self.pattern_length
        noise_frequency = self.noise_freq

        num_inputs = len(spikes_list[0])

        num_total_patterns = epochs * len(spikes_list)
        end = num_total_patterns * h_time + start

        start_pattern_times = np.linspace(start, end, num_total_patterns,
                                          endpoint=False)
        end_pattern_times = start_pattern_times + h_time
        
        start_noise_times = start_pattern_times + pattern_length

        num_per_pattern = int((end_pattern_times[0] - start_noise_times[0]) / h)
        times = np.linspace(
            start_noise_times,
            end_pattern_times,
            num_per_pattern,
            endpoint=False, axis=1).ravel()

        # np.random.random_sample is slow
        noise_dict = [None] * num_inputs
        for input_neuron in range(num_inputs):
            random_distribution_mask = \
                np.random.random_sample(len(times)) < noise_frequency * dt
            noise_dict[input_neuron] = {
                'spike_times': times[random_distribution_mask]
            }
        return noise_dict

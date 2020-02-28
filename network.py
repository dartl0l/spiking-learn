# coding: utf-8

import time
import nest
import numpy as np


class Network(object):
    """base class for different network types"""
    def __init__(self, settings):
        # super(Network, self).__init__()
        self.settings = settings
        self.synapse_models = [settings['model']['syn_dict_stdp']['model']]
      
    def set_input_spikes(self, spike_dict, spike_generators):
        # start_time = time.time()
        # for input_neuron, generator in zip(spike_dict, spike_generators):
        #     nest.SetStatus([generator], [spike_dict[input_neuron]])
        nest.SetStatus(spike_generators, spike_dict)
        # end_time = time.time()
        # print('set_input_spikes: ', end_time - start_time)

    def set_teachers_input(self, teacher_dicts):
        for teacher in teacher_dicts:
            nest.SetStatus([teacher], teacher_dicts[teacher])

    def set_poisson_noise(self, noise_dict, spike_generators):
        for input_neuron, generator in zip(noise_dict, spike_generators):
            nest.SetStatus([generator], [noise_dict[input_neuron]])

    def create_spike_dict(self, dataset, train):
        settings = self.settings

        epochs = settings['learning']['epochs'] if train else 1
        start = settings['network']['start_delta']

        spikes = np.tile(dataset, (epochs, 1, 1))

        full_time = epochs * len(dataset) * settings['network']['h_time'] + start
        times = np.arange(start, full_time, settings['network']['h_time'])
        pattern_start_times = np.expand_dims(np.tile(times, (len(dataset[0]), 1)).T, axis=2)

        assert len(spikes) == len(pattern_start_times)
        spike_times = np.add(spikes, pattern_start_times)

        # spike_dict = {}
        spike_dict = [None] * len(dataset[0])
        # print(spike_dict)
        # assert len(spike_dict) == len(dataset[0])
        for input_neuron in range(len(dataset[0])):
            spike_dict[input_neuron] = {'spike_times': spike_times[:, input_neuron].reshape(len(spikes))}

        return spike_dict, full_time, spikes

    def create_teacher_dict(self, input_spikes, classes, teachers):  # Network
        settings = self.settings

        h = settings['network']['h']
        h_time = settings['network']['h_time']
        epochs = settings['learning']['epochs']
        d_time = settings['network']['start_delta']
        single_neuron = settings['topology']['n_layer_out'] == 1
        reinforce_time = settings['learning']['reinforce_time']
        reinforce_delta = settings['learning']['reinforce_delta']
        teacher_amplitude = settings['learning']['teacher_amplitude']

        teacher_dicts = {}
        for teacher in teachers:
            teacher_dicts[teacher] = {
                                      'amplitude_times': [],
                                      'amplitude_values': []
                                     }
        # TODO
        # calc amplitude times one time and concatenate
        for _ in range(epochs):
            for spikes, cl in zip(input_spikes, classes):
                current_teacher_id = teachers[0] if single_neuron else teachers[cl]
                current_teacher = teacher_dicts[current_teacher_id]
                tmp_spikes = spikes[:]
                for i, spike in enumerate(tmp_spikes):
                    if len(spike) == 0:
                        tmp_spikes[i] = [np.nan]
                minimum = np.nanmin(tmp_spikes)

                start_of_stimulation = d_time \
                                     + minimum \
                                     + reinforce_delta
                end_of_stimulation = start_of_stimulation \
                                   + reinforce_time \
                                   + h 
                current_teacher['amplitude_times'].append(start_of_stimulation)
                current_teacher['amplitude_times'].append(end_of_stimulation)

                current_teacher['amplitude_values'].append(teacher_amplitude)
                current_teacher['amplitude_values'].append(0.0)
                d_time += h_time
        return teacher_dicts

    def create_poisson_noise(self, spikes_list):  # Network
        ''' 
            TODO make faster
        '''

        # def get_poisson_train(time, firing_rate, h):
        #     np.random.seed()
        #     dt = 1.0 / 1000.0
        #     times = np.arange(0, time, h)
        #     mask = np.random.random(int(time / h)) < firing_rate * dt
        #     spike_times = times[mask]
        #     return spike_times
        settings = self.settings

        h = settings['network']['h']
        h_time = settings['network']['h_time']
        epochs = settings['learning']['epochs']
        d_time = settings['network']['start_delta']
        noise_frequency = settings['network']['noise_freq']

        dt = 1.0 / 1000.0

        # noise = []
        noise_dict = {}
        for input_neuron in range(len(spikes_list[0])):
            noise_dict[input_neuron] = {'spike_times': [],
                                        'spike_weights': []}

        for _ in range(epochs):
            for spikes in spikes_list:
                start_of_noise = np.max(spikes) + d_time
                end_of_noise = h_time + d_time
                times = np.arange(start_of_noise, end_of_noise, h)
                noise_len = abs(start_of_noise - end_of_noise)
                poisson_spike_bins = int(noise_len / h)
                random_distribution = np.random.random((len(spikes), poisson_spike_bins))
                masks = random_distribution < noise_frequency * dt

                for input_neuron, mask in enumerate(masks):
                    noise_dict[input_neuron]['spike_times'] += times[mask].tolist()
                    noise_dict[input_neuron]['spike_weights'] += np.ones_like(times[mask]).tolist()
                d_time += h_time
        return noise_dict

    def interconnect_layer(self, layer, syn_dict):
        for neuron_1 in layer:
            for neuron_2 in layer:
                if neuron_1 != neuron_2:
                    nest.Connect([neuron_1], [neuron_2], syn_spec=syn_dict)
    
    def get_devices(self):
        devices = {
                    'voltmeter': self.voltmeter,
                    'spike_detector_out': self.spike_detector_out,
                    'spike_detector_input': self.spike_detector_input,
                    'spike_detector_hidden': self.spike_detector_hidden,
                   }
        return devices

    def get_spikes_of_pattern(spike_detector, estimated_time, example_class):
        spikes = nest.GetStatus(spike_detector, keys="events")[0]['times']
        senders = nest.GetStatus(spike_detector, keys="events")[0]['senders']
        mask = spikes > estimated_time
        spikes = spikes[mask]
        senders = senders[mask]
        tmp_dict = {
                    'latency': spikes - estimated_time,
                    'senders': senders,
                    'class': example_class
                    }
        return tmp_dict

    def save_weigths(self, layers):
        settings = self.settings
        synapse_models = self.synapse_models

        weights = {}

        for i, layer in enumerate(layers[1:]):
            synapse_model = synapse_models[i]
            previous_layer = layers[i]
            layer_name = 'layer_' + str(i)
            weights[layer_name] = {}
            for neuron_id in layer:
                tmp_weight = []
                for input_id in previous_layer:
                    conn = nest.GetConnections(
                        [input_id], [neuron_id], 
                        synapse_model=synapse_model
                    )
                    weight_one = nest.GetStatus(conn, 'weight')
                    if len(weight_one) != 0:
                        tmp_weight.append(weight_one[0])
                if len(tmp_weight) != 0:
                    weights[layer_name][neuron_id] = tmp_weight
        return weights

    def init_network(self):
        np.random.seed()
        # rank = nest.Rank()
        rng = np.random.randint(500)
        num_v_procs = self.settings['network']['num_threads'] \
                    * self.settings['network']['num_procs']

        nest.ResetKernel()
        nest.SetKernelStatus({
             'local_num_threads': self.settings['network']['num_threads'],
             'total_num_virtual_procs': num_v_procs,
             'resolution': self.settings['network']['h'],
             'rng_seeds': range(rng, rng + num_v_procs)
        })

    def create_layers(self):
        self.layer_out = nest.Create(self.settings['model']['neuron_out_model'], 
                                     self.settings['topology']['n_layer_out'])
        self.input_layer = nest.Create('parrot_neuron', 
                                        self.settings['topology']['n_input'])
        self.layers = [self.input_layer, self.layer_out]

    def create_devices(self):
        self.teacher_layer = nest.Create(
            'step_current_generator', 
            self.settings['topology']['n_layer_out'])

        self.input_generators = nest.Create(
            'spike_generator', 
            self.settings['topology']['n_input'])

        self.interpattern_noise_generator = nest.Create(
            'spike_generator', 
            self.settings['topology']['n_input'])

        self.poisson_layer = nest.Create(
            'poisson_generator', 
            self.settings['topology']['n_input'])

        self.spike_detector_out = nest.Create('spike_detector')
        self.spike_detector_input = nest.Create('spike_detector')
        self.spike_detector_hidden = nest.Create('spike_detector')

        self.voltmeter = nest.Create(
            'voltmeter', 1,
            {
             'withgid': True,
             'withtime': True
            }
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
        nest.Connect(self.voltmeter,
                     self.layer_out)

    def connect_teacher(self):
        nest.Connect(self.teacher_layer,
                     self.layer_out, 'one_to_one',
                     syn_spec='static_synapse')

    def connect_layers(self):
        nest.Connect(self.input_layer,
                     self.layer_out, 'all_to_all',
                     syn_spec=self.settings['model']['syn_dict_stdp'])

    def connect_layers_static(self):
        nest.Connect(self.input_layer,
                     self.layer_out, 'all_to_all',
                     syn_spec='static_synapse')

    def connect_layers_inh(self):
        self.interconnect_layer(self.layer_out,
                                self.settings['model']['syn_dict_inh'])

    def set_neuron_status(self):
        nest.SetStatus(self.layer_out,
                       self.settings['model']['neuron_out'])
        
    def set_noise(self):
        nest.SetStatus(
            self.poisson_layer,
            {
             'rate': self.settings['network']['noise_freq'],
             'origin': 0.0
            }
        )

    def set_weigths(self, weights):
        for neuron_id in weights['layer_0']:
            connection = nest.GetConnections(
                self.input_layer, target=[neuron_id])
            nest.SetStatus(connection, 'weight', 
                           weights['layer_0'][neuron_id])

    def train(self, data):
        print("start train")

        print("create network")
        self.init_network()
        print("create layers")
        self.create_layers()
        print("create devices")
        self.create_devices()

        print("connect")
        print("connect devices")
        self.connect_devices()
        print("connect teachers")
        self.connect_teacher()
        print("connect layers")
        self.connect_layers()
        if self.settings['topology']['use_inhibition']:
            self.connect_layers_inh()

        print("set status")
        self.set_neuron_status()
        if not self.settings['network']['noise_after_pattern']:
            self.set_noise()

        print("prepare spikes")
        spike_dict, full_time, input_spikes = self.create_spike_dict(
            dataset=data['input'], 
            train=True)
        self.set_input_spikes(
            spike_dict=spike_dict,
            spike_generators=self.input_generators)
        
        print("prepare teacher")
        teacher_dicts = self.create_teacher_dict(
            input_spikes=input_spikes,
            classes=data['class'], 
            teachers=self.teacher_layer)
        self.set_teachers_input(
            teacher_dicts)

        if self.settings['network']['noise_after_pattern']:
            print("prepare noise")
            noise_dict = self.create_poisson_noise(input_spikes)
            self.set_poisson_noise(
                noise_dict,
                self.interpattern_noise_generator)
        print("start simulation")
        nest.Simulate(full_time)

        weights = self.save_weigths(self.layers)
        output = {
                  'spikes': nest.GetStatus(self.spike_detector_out,
                                           keys="events")[0]['times'].tolist(),
                  'senders': nest.GetStatus(self.spike_detector_out,
                                            keys="events")[0]['senders'].tolist()
                 }
        devices = self.get_devices()
        return weights, output, devices

    def test(self, data, weights):
        print("start test")

        print("create network")
        self.init_network()
        print("create layers")
        self.create_layers()
        print("create devices")
        self.create_devices()

        print("connect")
        print("connect devices")
        self.connect_devices()
        print("connect layers")
        self.connect_layers_static()
        if self.settings['topology']['use_inhibition'] \
                and self.settings['network']['test_with_inhibition']:
            self.connect_layers_inh()
        print("set status")
        self.set_neuron_status()
        if self.settings['network']['test_with_noise']:
            self.set_noise()
        self.set_weigths(weights)

        print("prepare spikes")
        spike_dict, full_time, input_spikes = self.create_spike_dict(
            dataset=data['input'], 
            train=False)
        self.set_input_spikes(
            spike_dict=spike_dict,
            spike_generators=self.input_generators)
        
        print("start test simulation")
        nest.Simulate(full_time)

        spikes = nest.GetStatus(self.spike_detector_out,
                                keys="events")[0]['times'].tolist()
        senders = nest.GetStatus(self.spike_detector_out,
                                 keys="events")[0]['senders'].tolist()

        output = {
                  'spikes': spikes,
                  'senders': senders
                 }

        devices = {
                   'voltmeter': self.voltmeter,
                   'spike_detector_out': self.spike_detector_out,
                   'spike_detector_input': self.spike_detector_input,
                   'spike_detector_hidden': self.spike_detector_hidden,
                  }
        return output, devices


class TwoLayerNetwork(Network):
    def __init__(self, settings):
        # super(Network, self).__init__()
        self.settings = settings
        self.synapse_models = [settings['model']['syn_dict_stdp_hid']['model'], 
                               settings['model']['syn_dict_stdp']['model']]

    def create_layers(self):
        self.layer_out = nest.Create(self.settings['model']['neuron_out_model'], 
                                     self.settings['topology']['n_layer_out'])
        self.layer_hid = nest.Create(self.settings['model']['neuron_hid_model'], 
                                     self.settings['topology']['n_layer_hid'])
        self.input_layer = nest.Create('parrot_neuron', 
                                        self.settings['topology']['n_input'])
        self.layers = [self.input_layer, self.layer_hid, self.layer_out]

    def connect_layers(self):
        nest.Connect(self.input_layer,
                     self.layer_hid, 'all_to_all',
                     syn_spec=self.settings['model']['syn_dict_stdp_hid'])
        nest.Connect(self.layer_hid,
                     self.layer_out, 'all_to_all',
                     syn_spec=self.settings['model']['syn_dict_stdp'])
        if self.settings['topology']['use_reciprocal']:
            nest.Connect(self.layer_out,
                         self.layer_hid, 'all_to_all',
                         syn_spec=settings['model']['syn_dict_rec'])

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
                                self.settings['model']['syn_dict_inh'])

    def connect_devices(self):
        super().connect_devices()
        nest.Connect(self.layer_hid,
                     self.spike_detector_hidden, 'all_to_all')

    def set_neuron_status(self):
        super().set_neuron_status()
        nest.SetStatus(self.layer_hid,
                       self.settings['model']['neuron_hid'])

    def set_weigths(self, weights):
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
    def __init__(self, settings):
        # super(Network, self).__init__()
        self.settings = settings
        self.synapse_models = [settings['model']['syn_dict_stdp']['model']]

    def create_spike_dict(self, dataset, train):
        settings = self.settings

        spike_dict = {}
        spikes = []
        d_time = settings['network']['start_delta']
        epochs = settings['learning']['epochs'] if train else 1
        for input_neuron in dataset[0]:
            spike_dict[input_neuron] = {'spike_times': [],
                                        'spike_weights': []}
        # TODO
        # calc spike times one time and concatenate
        for _ in range(epochs):
            for example in dataset:
                tmp_spikes = []
                for input_neuron in example:
                    spike_dict[input_neuron]['spike_times'] \
                        += map(lambda x: x + d_time, example[input_neuron])
                    spike_dict[input_neuron]['spike_weights'] \
                        += np.ones_like(example[input_neuron]).tolist()
                    tmp_spikes.append(example[input_neuron])
                spikes.append(tmp_spikes)
                d_time += settings['network']['h_time']
        return spike_dict, d_time, spikes

    def create_teacher_dict(self, input_spikes, classes, teachers):  # Network
        settings = self.settings

        h = settings['network']['h']
        h_time = settings['network']['h_time']
        epochs = settings['learning']['epochs']
        d_time = settings['network']['start_delta']
        single_neuron = settings['topology']['n_layer_out'] == 1
        reinforce_time = settings['learning']['reinforce_time']
        reinforce_delta = settings['learning']['reinforce_delta']
        teacher_amplitude = settings['learning']['teacher_amplitude']

        teacher_dicts = {}
        for teacher in teachers:
            teacher_dicts[teacher] = {
                                      'amplitude_times': [],
                                      'amplitude_values': []
                                     }
        # TODO
        # calc amplitude times one time and concatenate
        for _ in range(epochs):
            for spikes, cl in zip(input_spikes, classes):
                current_teacher_id = teachers[0] if single_neuron else teachers[cl]
                current_teacher = teacher_dicts[current_teacher_id]
                start_of_stimulation = d_time \
                                     + reinforce_delta
                end_of_stimulation = start_of_stimulation \
                                   + reinforce_time \
                                   + h 
                current_teacher['amplitude_times'].append(start_of_stimulation)
                current_teacher['amplitude_times'].append(end_of_stimulation)

                current_teacher['amplitude_values'].append(teacher_amplitude)
                current_teacher['amplitude_values'].append(0.0)
                d_time += h_time
        return teacher_dicts
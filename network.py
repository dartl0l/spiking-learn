# coding: utf-8

import nest
import numpy as np


class Network(object):
    """base class for different network types"""
    def __init__(self, settings):
        # super(Network, self).__init__()
        self.settings = settings
      
    def set_input_spikes(self, dataset, spike_generators, train):
        spike_dict, d_time, spikes = self.create_spike_dict(dataset, train)
        for input_neuron in spike_dict:
            nest.SetStatus([spike_generators[input_neuron - 1]],
                           [spike_dict[input_neuron]])
        return d_time, spikes

    def set_teachers_input(self, input_spikes, classes, teachers):
        teacher_dicts = self.create_teacher_dict(input_spikes, classes,
                                                 teachers)
        for teacher in teacher_dicts:
            nest.SetStatus([teacher], teacher_dicts[teacher])

    def set_poisson_noise(self, spikes_list, spike_generators):
        noise_dict = self.create_poisson_noise(spikes_list)
        for input_neuron, generator in zip(noise_dict, spike_generators):
            nest.SetStatus([generator], [noise_dict[input_neuron]])

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
                tmp_spikes = spikes[:]
                for i, spike in enumerate(tmp_spikes):
                    if len(spike) == 0:
                        tmp_spikes[i] = [np.nan]
                minimum = np.nanmin(tmp_spikes)
#                 print(minimum)
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
        # def get_poisson_train(time, firing_rate, h):
        #     np.random.seed()
        #     dt = 1.0 / 1000.0 * h
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

        dt = 1.0 / 1000.0 * h

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

        weights = {}
        if settings['topology']['two_layers']:
            synapse_models = [settings['model']['syn_dict_stdp_hid']['model'], 
                              settings['model']['syn_dict_stdp']['model']]
        else:
            synapse_models = [settings['model']['syn_dict_stdp']['model']]

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

    def train(self, data):
        settings = self.settings

        np.random.seed()
        # rank = nest.Rank()
        rng = np.random.randint(500)
        num_v_procs = settings['network']['num_threads'] \
                    * settings['network']['num_procs']

        nest.ResetKernel()
        nest.SetKernelStatus({
             'local_num_threads': settings['network']['num_threads'],
             'total_num_virtual_procs': num_v_procs,
             'resolution': settings['network']['h'],
             'rng_seeds': range(rng, rng + num_v_procs)
        })

        layer_out = nest.Create('iaf_psc_exp', 
                                settings['topology']['n_layer_out'])
        if settings['topology']['two_layers']:
            layer_hid = nest.Create('iaf_psc_exp', 
                                    settings['topology']['n_layer_hid'])

        teacher_1 = nest.Create('step_current_generator',
                                settings['topology']['n_layer_out'])

        # teacher_2 = nest.Create('ac_generator',
        #                         settings['topology']['n_layer_out'])

        spike_generators_1 = nest.Create('spike_generator', 
                                         settings['topology']['n_input'])
        spike_generators_2 = nest.Create('spike_generator', 
                                         settings['topology']['n_input'])
        poisson_layer = nest.Create('poisson_generator', 
                                    settings['topology']['n_input'])
        parrot_layer = nest.Create('parrot_neuron', 
                                   settings['topology']['n_input'])

        spike_detector_1 = nest.Create('spike_detector')
        spike_detector_2 = nest.Create('spike_detector')
        spike_detector_3 = nest.Create('spike_detector')

        voltmeter = nest.Create(
            'voltmeter', 1,
            {
             'withgid': True,
             'withtime': True
            }
        )

        if not settings['network']['noise_after_pattern']:
            nest.SetStatus(
                poisson_layer,
                {
                 'rate': settings['network']['noise_freq'],
                 'origin': 0.0
                }
            )

        nest.Connect(spike_generators_1, parrot_layer, 'one_to_one', 
                     syn_spec='static_synapse')
        nest.Connect(poisson_layer, parrot_layer, 'one_to_one',
                     syn_spec='static_synapse')

        if settings['learning']['use_teacher']:
            nest.Connect(teacher_1, layer_out, 'one_to_one',
                         syn_spec='static_synapse')

        nest.Connect(layer_out, spike_detector_1, 'all_to_all')
        nest.Connect(parrot_layer, spike_detector_2, 'all_to_all')
        nest.Connect(voltmeter, layer_out)

        nest.SetStatus(layer_out, settings['model']['neuron_out'])

        if settings['topology']['two_layers']:
            if settings['topology']['use_inhibition']:
                self.interconnect_layer(layer_hid, settings['model']['syn_dict_inh'])

            nest.Connect(parrot_layer, layer_hid, 'all_to_all', 
                         syn_spec=settings['model']['syn_dict_stdp_hid'])
            nest.Connect(layer_hid, layer_out, 'all_to_all',
                         syn_spec=settings['model']['syn_dict_stdp'])
            if settings['topology']['use_reciprocal']:
                nest.Connect(layer_out, layer_hid, 'all_to_all',
                             syn_spec=settings['model']['syn_dict_rec'])
            nest.Connect(layer_hid, spike_detector_3, 'all_to_all')
            nest.SetStatus(layer_hid, settings['model']['neuron_hid'])
        else:
            nest.Connect(parrot_layer, layer_out, 'all_to_all',
                         syn_spec=settings['model']['syn_dict_stdp'])

        if settings['topology']['use_inhibition']:
            self.interconnect_layer(layer_out, settings['model']['syn_dict_inh'])

        np.random.seed(500)

        is_train = True
        full_time, \
        input_spikes = self.set_input_spikes(data['input'], 
                                             spike_generators_1,
                                             is_train)

        self.set_teachers_input(input_spikes, 
                                data['class'],
                                teacher_1)

        if settings['network']['noise_after_pattern']:
            nest.Connect(spike_generators_2, parrot_layer, 'one_to_one', 
                         syn_spec='static_synapse')
            self.set_poisson_noise(input_spikes, spike_generators_2)

        nest.Simulate(full_time)

        layers = [parrot_layer, layer_hid, layer_out] \
            if settings['topology']['two_layers'] \
            else [parrot_layer, layer_out]

        spikes = nest.GetStatus(spike_detector_1,
                                keys="events")[0]['times'].tolist()
        senders = nest.GetStatus(spike_detector_1,
                                 keys="events")[0]['senders'].tolist()

        weights = self.save_weigths(layers)
        output = {
                  'spikes': spikes,
                  'senders': senders
                 }
        devices = {
                   'voltmeter': voltmeter,
                   'spike_detector_1': spike_detector_1,
                   'spike_detector_2': spike_detector_2,
                   'spike_detector_3': spike_detector_3,
                  }
        return weights, output, devices

    def test(self, data, weights):
        settings = self.settings

        np.random.seed()
        # rank = nest.Rank()
        rng = np.random.randint(500)
        num_v_procs = settings['network']['num_threads'] \
                    * settings['network']['num_procs']

        nest.ResetKernel()
        nest.SetKernelStatus({
             'local_num_threads': settings['network']['num_threads'],
             'total_num_virtual_procs': num_v_procs,
             'resolution': settings['network']['h'],
             'rng_seeds': range(rng, rng + num_v_procs)
        })

        layer_out = nest.Create('iaf_psc_exp',
                                settings['topology']['n_layer_out'])
        if settings['topology']['two_layers']:
            layer_hid = nest.Create('iaf_psc_exp', 
                                    settings['topology']['n_layer_hid'])

        spike_generators_1 = nest.Create('spike_generator', 
                                         settings['topology']['n_input'])
        poisson_layer = nest.Create('poisson_generator', 
                                    settings['topology']['n_input'])
        parrot_layer = nest.Create('parrot_neuron', 
                                   settings['topology']['n_input'])

        spike_detector_1 = nest.Create('spike_detector')
        spike_detector_2 = nest.Create('spike_detector')
        spike_detector_3 = nest.Create('spike_detector')

        voltmeter = nest.Create(
            'voltmeter', 1,
            {
             'withgid': True,
             'withtime': True
            }
        )
        nest.Connect(spike_generators_1, parrot_layer,
                     'one_to_one', syn_spec='static_synapse')

        if settings['network']['test_with_noise']:
            nest.SetStatus(poisson_layer, 
                           {'rate': settings['network']['noise_freq']})
            nest.Connect(poisson_layer, parrot_layer,
                         'one_to_one', syn_spec='static_synapse')

        nest.Connect(layer_out, spike_detector_1, 'all_to_all')
        nest.Connect(parrot_layer, spike_detector_2, 'all_to_all')
        nest.Connect(voltmeter, layer_out)

        nest.SetStatus(layer_out, settings['model']['neuron_out'])

        if settings['topology']['two_layers']:
            if settings['topology']['use_inhibition'] and settings['network']['test_with_inhibition']:
                self.interconnect_layer(layer_hid,
                                        settings['model']['syn_dict_inh'])

            nest.Connect(parrot_layer, layer_hid,
                         'all_to_all', syn_spec='static_synapse')
            nest.Connect(layer_hid, layer_out,
                         'all_to_all', syn_spec='static_synapse')
            nest.Connect(layer_hid, spike_detector_3, 'all_to_all')
            nest.SetStatus(layer_hid, settings['model']['neuron_hid'])
        else:
            if settings['topology']['use_inhibition'] and settings['network']['test_with_inhibition']:
                self.interconnect_layer(layer_out, 
                                        settings['model']['syn_dict_inh'])
            nest.Connect(parrot_layer, layer_out,
                         'all_to_all', syn_spec='static_synapse')

        for neuron_id in weights['layer_0']:
            connection = nest.GetConnections(parrot_layer,
                                             target=[neuron_id])
            nest.SetStatus(connection, 'weight', 
                           weights['layer_0'][neuron_id])

        if settings['topology']['two_layers']:
            for neuron_id in weights['layer_1']:
                connection = nest.GetConnections(layer_hid,
                                                 target=[neuron_id])
                nest.SetStatus(connection, 'weight',
                               weights['layer_1'][neuron_id])

        np.random.seed(500)
        is_train = False
        full_time, input_spikes = self.set_input_spikes(data['input'],
                                                        spike_generators_1,
                                                        is_train)
        nest.Simulate(full_time)

        spikes = nest.GetStatus(spike_detector_1,
                                keys="events")[0]['times'].tolist()
        senders = nest.GetStatus(spike_detector_1,
                                 keys="events")[0]['senders'].tolist()

        output = {
                  'spikes': spikes,
                  'senders': senders
                 }

        devices = {
                   'voltmeter': voltmeter,
                   'spike_detector_1': spike_detector_1,
                   'spike_detector_2': spike_detector_2,
                   'spike_detector_3': spike_detector_3,
                  }
        return output, devices



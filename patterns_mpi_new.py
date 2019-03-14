import nest

import numpy as np
import operator
import json
import sys

from math import exp
from mpi4py import MPI
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold


class Settings(object):
    """docstring for Settings"""
    def __init__(self, settings_from_file):
        super(Settings, self).__init__()
        self.arg = settings_from_file


class Topology(object):
    """docstring for Toplogy"""
    def __init__(self, settings):
        super(Toplogy, self).__init__()
        self.settings = settings


class Network(object):
    """docstring for Network"""
    def __init__(self, settings_from_file):
        super(Network, self).__init__()
        self.settings = Settings(settings_from_file)

        topology = Topology(self.settings)
        
    def train(data):
        
        return latency

    def test(data):

        return


def set_spike_in_generators(data, spike_generators, start_time, end_time, h_time, start_h):
    spikes = []
    spike_times = []

    for generator_num in data:
        d_time = start_time
        spike_dict = {'spike_times': [],
                      'spike_weights': []}

        if data[generator_num]:
            spike_times.append(data[generator_num][0])

        while d_time < end_time:
            spike_dict['spike_times'] += map(lambda x: x + d_time + start_h, data[generator_num])
            spike_dict['spike_weights'] += np.ones_like(data[generator_num]).tolist()
            d_time += h_time
        # print gen_num, sp_tmp
        spikes.append(spike_dict)     

    for gen_num, spp in zip(data, spikes):
        # print gen_num, spike_generators_1[gen_num - 1]
        nest.SetStatus([spike_generators[gen_num - 1]], [spp])

    return spike_times, spikes


def prepare_spike_generator(data, start_time, end_time):
    spikes = []
    spike_times = []

    for generator_num in data:
        d_time = start_time
        spike_dict = {'spike_times': [],
                      'spike_weights': []}

        if data[generator_num]:
            spike_times.append(data[generator_num][0])

        while d_time < end_time:
            spike_dict['spike_times'] += map(lambda x: x + d_time + start_h, data[generator_num])
            spike_dict['spike_weights'] += np.ones_like(data[generator_num]).tolist()
            d_time += h_time
        # print gen_num, sp_tmp
        spikes.append(spike_dict)     

    # for gen_num, spp in zip(data, spikes):
    #     # print gen_num, spike_generators_1[gen_num - 1]
    #     nest.SetStatus([spike_generators[gen_num - 1]], [spp])

    return spikes


def create_spike_dict(dataset, epochs):
    epochs = settings['learning']['epochs'] if train else 1
    spike_dict = {}
    for input_neuron in dataset[0]:
        spike_dict[input_neuron] = {'spike_times': [],
                                    'spike_weights': []}
    print(spike_dict)

    d_time = settings['network']['start_delta']
    for _ in range(epochs):
        for example in dataset:
            for input_neuron in example:
                spike_dict[input_neuron]['spike_times'] 
                    += map(lambda x: x + d_time, example[input_neuron])
                spike_dict[input_neuron]['spike_weights']
                    += np.ones_like(example[input_neuron]).tolist()
            d_time += settings['network']['h_time']
    print(spike_dict)
    return spike_dict, d_time


def set_spike_input(dataset, spike_generators, settings, train):
    spike_dict, d_time = create_spike_dict(dataset, train, settings)
    for input_neuron in spike_dict:
        # print gen_num, spike_generators_1[gen_num - 1]
        nest.SetStatus([spike_generators[input_neuron - 1]],
                       [spike_dict[input_neuron]])
    return d_time


def set_teacher_input(start_of_stimulation, teacher, settings):  # Network
    ampl_times = []
    ampl_values = []

    h = settings['network']['h']
    
    end_of_stimulation = start_of_stimulation \
                       + 2 * h \
                       + settings['learning']['reinforce_time']  # x + 0.2
    ampl_times.append(start_of_stimulation - h)  # x - 1.1
    ampl_times.append(end_of_stimulation - h)  # x - 1.1
    ampl_values.append(settings['learning']['teacher_amplitude'])  # 1 mA 1000000.0
    ampl_values.append(0.0)  # 0 pA
    
    nest.SetStatus(teacher, {'amplitude_times': ampl_times,
                             'amplitude_values': ampl_values})


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


def save_latency_to_file(raw_latency, filename):
    rank = nest.Rank()
    with open((filename + str(rank) + '.json'), 'w') as latency_file:
        json.dump(raw_latency, latency_file, indent=4)


def merge_spikes_and_senders(raw_latency_list):
    # raw_latency = json.load(open('latency_0.json', 'r'))
    raw_latency = {
                   'spikes': [],
                   'senders': []
                  }
    for tmp_latency in raw_latency_list:
        raw_latency['spikes'].extend(tmp_latency['spikes'])
        raw_latency['senders'].extend(tmp_latency['senders'])
    raw_latency['spikes'] = np.array(raw_latency['spikes'])
    raw_latency['senders'] = np.array(raw_latency['senders'])
    return raw_latency


def split_spikes_and_senders(input_latency, n_splits, settings):
    output_latency = []
    d_time = settings['network']['start_delta']
    for _ in range(n_splits):
        mask = (input_latency['spikes'] > d_time) & \
               (input_latency['spikes'] < d_time + settings['network']['h_time'])
        spikes_tmp = input_latency['spikes'][mask]
        senders_tmp = input_latency['senders'][mask]
        tmp_dict = {
                    'spikes': spikes_tmp - d_time,
                    'senders': senders_tmp
                    }

        d_time += settings['network']['h_time']
        output_latency.append(tmp_dict)    
    return output_latency


def convert_latency(latency_list, settings):
    output_list = []
    base_latency_dict = {}

    n_neurons = settings['topology']['n_layer_out']
    neuron_out_ids = [i + 1 for i in range(n_neurons)]
    neuron_names = ['neuron_' + str(i) for i in range(n_neurons)]
    for neuron_name in neuron_names:
        base_latency_dict[neuron_name] = [float('Inf')]

    for latencies in latency_list:
        tmp_latency_dict = base_latency_dict.copy()
        for lat, sender in zip(latencies['spikes'], latencies['senders']):
            for neuron_name, neuron_id in zip(neuron_names, neuron_out_ids):
                if sender == [neuron_id] and [lat] < tmp_latency_dict[neuron_name]:
                    tmp_latency_dict[neuron_name] = [lat] 
        output_list.append(tmp_latency_dict)
    return output_list


def predict_from_latency(latency_list):
    output_list = []
    for latency in latency_list:
        tmp_list = [latency[neuron][:1] for neuron in sorted(latency.keys())]
        min_index, min_value = min(enumerate(tmp_list),
                                   key=operator.itemgetter(1))
        output_list.append(min_index)
    return output_list


def fitness_func_time(latency_list, data):
    fit_list = []

    for latency, y in zip(latency_list, data['class']):
        tmp_list = [latency[neuron_number][:1][0] for neuron_number in sorted(latency.keys())]
        latency_of_desired_neuron = tmp_list.pop(y)

        fit = -1 * latency_of_desired_neuron
        fit_list.append(fit)

    fitness = np.mean(fit_list)
    if np.isnan(fitness):
        fitness = 0
    return fitness


def fitness_func_sigma(latency_list, data):
    def sigmoid(x, alpha):
        return 1 / (1 + np.exp(-2 * alpha * x))

    fit_list = []
    for latency, y in zip(latency_list, data['class']):
        tmp_list = [latency[neuron_number][:1][0] for neuron_number in sorted(latency.keys())]
        latency_of_desired_neuron = tmp_list.pop(y)
        fit = 1
        for lat in tmp_list:
            fit *= sigmoid(lat - latency_of_desired_neuron, 0.1)
        fit_list.append(fit)
    fitness = np.mean(fit_list)
    if np.isnan(fitness):
        fitness = 0
    return fitness, fit_list


def fitness_func_exp(latency_list, data):
    fit_list = []
    for latency, y in zip(latency_list, data['class']):
        tmp_list = [latency[neuron_number][:1][0] for neuron_number in sorted(latency.keys())]
        latency_of_desired_neuron = tmp_list.pop(y)
        fit = 1
        for lat in tmp_list:
            fit -= exp(latency_of_desired_neuron - lat)
        fit_list.append(fit)
    fitness = np.mean(fit_list)
    # print(fit_list)
    # print(fitness)
    if np.isnan(fitness):
        fitness = 0
    return fitness, fit_list


def save_weigths(layers, settings):
    weights = {}
    for i, layer in enumerate(layers[1:]):
        previous_layer = layers[i]
        layer_name = 'layer_' + str(i)
        weights[layer_name] = {}
        for neuron_id in layer:
            tmp_weight = []
            for input_id in previous_layer:
                conn = nest.GetConnections([input_id], [neuron_id], 
                                           synapse_model=settings['model']['syn_dict_stdp_hid']['model'])
                weight_one = nest.GetStatus(conn, 'weight')
                if len(weight_one) != 0:
                    tmp_weight.append(weight_one[0])
            if len(tmp_weight) != 0:
                weights[layer_name][neuron_id] = tmp_weight
    return weights


def weight_norm(weights):
    norms = []
    for weight in weights:
        norm = []
        for neuron in weight:
            norm.append(np.linalg.norm(weights[neuron]))
        norms.append(np.linalg.norm(norm))
    return np.linalg.norm(norms)


def interconnect_layer(layer, syn_dict):
    for neuron_1 in layer:
        for neuron_2 in layer:
            if neuron_1 != neuron_2:
                nest.Connect([neuron_1], [neuron_2], syn_spec=syn_dict)


def prepare_data(data, train_index, test_index, settings):
    data_train = {}
    data_test = {}
    data_out = {}

    if settings['data']['use_valid']:
        data_valid = {}
        
        data_out = {'train': {},
                    'test': {},
                    'valid': {}}
        input_train, input_valid, y_train, y_valid = train_test_split(data['input'][train_index],
                                                                      data['class'][train_index],
                                                                      test_size=settings['data']['valid_size'],
                                                                      random_state=42)
        data_train['input'] = input_train
        data_train['class'] = y_train

        data_valid['input'] = input_valid
        data_valid['class'] = y_valid

        data_test['input'] = data['input'][test_index]
        data_test['class'] = data['class'][test_index]

        data_out['test']['full'] = data_test
        data_out['train']['full'] = data_train
        data_out['valid']['full'] = data_valid
    else:
        data_out = {'train': {},
                    'test': {}}
        data_train['input'] = data['input'][train_index]
        data_train['class'] = data['class'][train_index]

        data_test['input'] = data['input'][test_index]
        data_test['class'] = data['class'][test_index]

        data_out['test']['full'] = data_test
        data_out['train']['full'] = data_train

    return data_out


def train(settings, data):
    np.random.seed()
    rank = nest.Rank()
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
        'voltmeter', 
        1,
        {
         'withgid': True,
         'withtime': True
        }
    )

    if not settings['network']['noise_after_pattern']:
        nest.SetStatus(poisson_layer,
                       {'rate': settings['network']['noise_freq'],
                        'origin': 0.0})

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
        if settings['learning']['use_inhibition']:
            interconnect_layer(layer_hid, settings['model']['syn_dict_inh'])
            # nest.Connect(layer_out, layer_hid,
            #              'all_to_all', syn_spec=settings['syn_dict_inh'])
        
        nest.Connect(parrot_layer, spike_detector_3, 'all_to_all')
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
        interconnect_layer(layer_out, settings['model']['syn_dict_inh'])

    np.random.seed(500)

    i = 0
    hi = 1
    # last_norms = []
    norm_history = []
    output_latency = []
    weights_history = []
    
    early_stop = False
    d_time = settings['network']['start_delta']
    full_time = settings['learning']['epochs'] \
              * len(data['input']) \
              * settings['network']['h_time'] \
              + settings['network']['start_delta']
    
    # if settings['two_layers']:
    #     initial_weights = save_weigths_two_layers(parrot_layer, layer_hid, layer_out, settings)
    # else:
    #     initial_weights = save_weights_one_layer(parrot_layer, layer_out, settings)

    single_neuron = settings['topology']['n_layer_out'] == 1

    nest.Simulate(settings['network']['start_delta'])

    while not early_stop:
        spike_times, spikes = set_spike_in_generators(
            data['input'][i],
            spike_generators_1,
            d_time,
            d_time + settings['network']['h_time'],
            settings['network']['h_time'],
            settings['network']['h']
        )
        # spike_times_1 = []
        # for neuron_number in data['input'][i]:
        #     if data['input'][i][neuron_number]:
        #         spike_times_1.append(data['input'][i][neuron_number][0])
        
        # print(spike_times)
        # print(spike_times_1)

        if settings['learning']['use_teacher']:
            current_teacher = teacher_1 if single_neuron else [teacher_1[data['class'][i]]]
            teacher_impulse_time = np.min(spike_times) \
                                 + d_time \
                                 + settings['network']['h'] \
                                 + settings['learning']['reinforce_delta']
            set_teacher_input(teacher_impulse_time, current_teacher, settings)

        if settings['network']['noise_after_pattern']:
            nest.SetStatus(
                poisson_layer,
                {
                 'start': d_time + np.max(spike_times),
                 'stop': d_time + settings['network']['h_time'],
                 'rate': settings['network']['noise_freq']
                })

        nest.Simulate(settings['network']['h_time'])
        
        tmp_dict = get_spikes_of_pattern(spike_detector_1, d_time, data['class'][i])
        output_latency.append(tmp_dict)

        d_time += settings['network']['h_time']
        if i + hi + 1 > len(data['input']):
            i = 0
        else:
            i += hi
        
        if settings['network']['save_history']:

            layers = [parrot_layer, layer_hid, layer_out] if settings['topology']['two_layers'] \
                else [parrot_layer, layer_out]

            tmp_weights = save_weigths(layers, settings)
            tmp_norm = weight_norm(tmp_weights)

            norm_history.append(tmp_norm)
            weights_history.append(tmp_weights)
        early_stop = d_time > full_time

    layers = [parrot_layer, layer_hid, layer_out] if settings['topology']['two_layers'] \
        else [parrot_layer, layer_out]

    weights = save_weigths(layers, settings)

    # print(weights['layer_out'].keys())
    # with open('weights' + str(weights['layer_out'].keys()) + '.json', 'w') as outfile:
    #     json.dump(weights, outfile, indent=4)
    devices = {
               'voltmeter': voltmeter,
               'spike_detector_1': spike_detector_1,
               'spike_detector_2': spike_detector_2,
               'spike_detector_3': spike_detector_3,
              }
    return weights, output_latency, devices, weights_history, norm_history


def test(settings, data, weights):
    np.random.seed()
    rank = nest.Rank()
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
        'voltmeter', 
        1,
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
        if settings['topology']['use_inhibition']:
            interconnect_layer(layer_hid, settings['model']['syn_dict_inh'])
            # nest.Connect(layer_out, layer_hid,
            #              'all_to_all', syn_spec=settings['syn_dict_inh'])
        
        nest.Connect(parrot_layer, layer_hid,
                     'all_to_all', syn_spec='static_synapse')
        nest.Connect(layer_hid, layer_out,
                     'all_to_all', syn_spec='static_synapse')
        nest.Connect(layer_hid, spike_detector_3, 'all_to_all')
        nest.SetStatus(layer_hid, settings['model']['neuron_hid'])
    else:
        if settings['topology']['use_inhibition']:
            interconnect_layer(layer_out, settings['model']['syn_dict_inh'])
        nest.Connect(parrot_layer, layer_out,
                     'all_to_all', syn_spec='static_synapse')

    for neuron_id in weights['layer_0']:
        connection = nest.GetConnections(parrot_layer, target=[neuron_id])
        nest.SetStatus(connection, 'weight', weights['layer_0'][neuron_id])
    if settings['topology']['two_layers']:
        for neuron_id in weights['layer_1']:
            connection = nest.GetConnections(layer_hid, target=[neuron_id])
            nest.SetStatus(connection, 'weight', weights['layer_1'][neuron_id])

    np.random.seed(500)
    output_latency = []
    # d_time = settings['network']['start_delta']
    # nest.Simulate(settings['network']['start_delta'])

    # for example, ex_class in zip(data['input'], data['class']):
    #     set_spike_in_generators(
    #         example,
    #         spike_generators_1,
    #         d_time,
    #         d_time + settings['network']['h_time'],
    #         settings['network']['h_time'],
    #         settings['network']['h']
    #     )
    #     # nest.SetStatus(poisson_layer, {'start': 30.})
    #     nest.Simulate(settings['network']['h_time'])
    #     d_time += settings['network']['h_time']
    d_time = set_spike_input(data['input'], 
                             spike_generators_1,
                             settings,
                             False)
    nest.Simulate(d_time)

    spikes = nest.GetStatus(spike_detector_1, keys="events")[0]['times'].tolist()
    senders = nest.GetStatus(spike_detector_1, keys="events")[0]['senders'].tolist()

    output_latency = {
                      'spikes': spikes,
                      'senders': senders
                     }

    devices = {
               'voltmeter': voltmeter,
               'spike_detector_1': spike_detector_1,
               'spike_detector_2': spike_detector_2,
               'spike_detector_3': spike_detector_3,
              }
    return output_latency, devices


def test_network_acc(data, settings):
    comm = MPI.COMM_WORLD

    data_train = data['train']['full']
    weights, \
    latency_train, devices_train, \
    weights_history, norm_history = train(settings, data_train)
    
    fitness = 0
    if settings['data']['use_valid']:
        data_valid = data['valid']['full']
        raw_latency_valid, devices_valid = test(settings, data_valid, weights)
        all_latency_valid = comm.allgather(raw_latency_valid)
        comm.Barrier()
        raw_latency_valid = merge_spikes_and_senders(all_latency_valid)
        latency_valid = split_spikes_and_senders(raw_latency_valid, len(data_valid['class']), settings)
        full_latency_valid = convert_latency(latency_valid, settings)
        if settings['learning']['use_fitness_func'] and settings['learning']['fitness_func'] == 'exp':
            fitness, fit_list = fitness_func_exp(full_latency_valid, data_valid)
        elif settings['learning']['use_fitness_func'] and settings['learning']['fitness_func'] == 'sigma':
            fitness, fit_list = fitness_func_sigma(full_latency_valid, data_valid)
        elif settings['learning']['use_fitness_func'] and settings['learning']['fitness_func'] == 'time':
            fitness = fitness_func_time(full_latency_valid, data_valid)
        elif settings['learning']['use_fitness_func'] and settings['learning']['fitness_func'] == 'acc':
            y_valid = predict_from_latency(full_latency_valid)
            fitness = accuracy_score(data_valid['class'], y_valid)
        elif settings['learning']['use_fitness_func'] and settings['learning']['fitness_func'] == 'f1':
            y_valid = predict_from_latency(full_latency_valid)
            fitness = f1_score(data_valid['class'], y_valid, average='micro')

    raw_latency_test_train, devices_test_train = test(settings, data_train, weights)
    all_latency_test_train = comm.allgather(raw_latency_test_train)
    comm.Barrier()
    raw_latency_test_train = merge_spikes_and_senders(all_latency_test_train)
    latency_test_train = split_spikes_and_senders(raw_latency_test_train, len(data_train['class']), settings)
    full_latency_test_train = convert_latency(latency_test_train, settings)
    y_train = predict_from_latency(full_latency_test_train)

    score_train = 0
    if settings['learning']['metrics'] == 'acc':
        score_train = accuracy_score(data_train['class'], y_train)
    elif settings['learning']['metrics'] == 'f1':
        score_train = f1_score(data_train['class'], y_train, average='micro')

    data_test = data['test']['full']
    raw_latency_test, devices_test = test(settings, data_test, weights)
    all_latency_test = comm.allgather(raw_latency_test)
    comm.Barrier()
    raw_latency_test = merge_spikes_and_senders(all_latency_test)
    latency_test = split_spikes_and_senders(raw_latency_test, len(data_test['class']), settings)

    full_latency_test = convert_latency(latency_test, settings)
    y_test = predict_from_latency(full_latency_test)

    score_test = 0
    if settings['learning']['metrics'] == 'acc':
        score_test = accuracy_score(data_test['class'], y_test)
    elif settings['learning']['metrics'] == 'f1':
        score_test = f1_score(data_test['class'], y_test, average='micro')

    comm.Barrier()

    weights_all = comm.allgather(weights)

    out_dict = {
                'fitness': fitness,
                'acc_test': score_test,
                'acc_train': score_train,
                'output_list_test': y_test,
                'output_list_train': y_train,
               }
    
    return out_dict, weights_all


def test_network_acc_cv(data, settings):
    def solve_fold(input_data):
        data_fold = prepare_data(input_data['data'],
                                 input_data['train_index'],
                                 input_data['test_index'],
                                 input_data['settings'])
        return test_network_acc(data_fold, input_data['settings'])

    fit = []
    weights = []
    acc_test = []
    acc_train = []
    data_list = []

    skf = StratifiedKFold(n_splits=settings['learning']['n_splits'])
    for train_index, test_index in skf.split(data['input'], data['class']):
        input_data = {
                      'data': data,
                      'settings': settings,
                      'test_index': test_index,
                      'train_index': train_index,
                     }
        data_list.append(input_data)

    for result, weight in map(solve_fold, data_list):
        acc_test.append(result['acc_test'])
        acc_train.append(result['acc_train'])
        fit.append(result['fitness'])
        weights.append(weight)

    # print(fit)
    out_dict = {
                'fitness': fit,
                'fitness_mean': np.mean(fit),
                'accs_test': acc_test,
                'accs_test_mean': np.mean(acc_test),
                'accs_test_std': np.std(acc_test),
                'accs_train': acc_train,
                'accs_train_mean': np.mean(acc_train),
                'accs_train_std': np.std(acc_train),
                'weights': weights,
               }
    return out_dict


def grid_search(data, parameters, settings):
    result = {
              'accuracy': [],
              'std': [],
              'fitness': [],
              'parameter': [],
              'parameter_name': [],
              }
    settings_copy = settings
    for key in parameters.keys():
        if isinstance(parameters[key], dict):
            for key_key in parameters[key].keys():
                result['parameter'].append(settings_copy[key][key_key])
                result['parameter_name'].append(key_key)
                acc, std = test_network_acc_cv(data, settings_copy)
                result['accuracy'].append(acc)
                result['std'].append(std)
                settings_copy[key][key_key] += parameters[key][key_key]
    else:
        result['parameter'].append(settings_copy[key])
        result['parameter_name'].append(key)
        acc, std = test_network_acc_cv(data, settings_copy)
        result['accuracy'].append(acc)
        result['std'].append(std)
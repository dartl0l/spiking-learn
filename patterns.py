# coding: utf-8

import nest

import numpy as np
import operator
import json
import sys

from math import exp, floor
from mpi4py import MPI
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold


class Settings(object):
    """docstring for Settings"""
    def __init__(self, settings_dict):
        super(Settings, self).__init__()
        self.arg = settings_dict


class Topology(object):
    """docstring for Topology"""
    def __init__(self, settings):
        super(Topology, self).__init__()
        self.settings = settings


class Network(object):
    """docstring for Network"""
    def __init__(self, settings_dict):
        super(Network, self).__init__()
        self.settings = Settings(settings_dict)

        topology = Topology(self.settings)
        
    def train(data):
        weights = {}

        output = {'weights': weights}
        return output

    def test(data):

        pass


def create_spike_dict(dataset, train, settings):
    spike_dict = {}
    spikes = []
    d_time = settings['network']['start_delta']
    epochs = settings['learning']['epochs'] if train else 1
    for input_neuron in dataset[0]:
        spike_dict[input_neuron] = {'spike_times': [],
                                    'spike_weights': []}
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


def set_input_spikes(dataset, spike_generators, settings, train):
    spike_dict, d_time, spikes = create_spike_dict(dataset, train, settings)
    for input_neuron in spike_dict:
        nest.SetStatus([spike_generators[input_neuron - 1]],
                       [spike_dict[input_neuron]])
    return d_time, spikes


def create_teacher_dict(spikes_list, classes, teachers, settings):  # Network
    teacher_dicts = {}
    h = settings['network']['h']
    h_time = settings['network']['h_time']
    epochs = settings['learning']['epochs']
    d_time = settings['network']['start_delta']
    single_neuron = settings['topology']['n_layer_out'] == 1
    reinforce_time = settings['learning']['reinforce_time']
    reinforce_delta = settings['learning']['reinforce_delta']
    teacher_amplitude = settings['learning']['teacher_amplitude']
    for teacher in teachers:
        teacher_dicts[teacher] = {
                                  'amplitude_times': [],
                                  'amplitude_values': []
                                 }
    for _ in range(epochs):
        for spikes, cl in zip(spikes_list, classes):
            current_teacher_id = teachers[0] if single_neuron else teachers[cl]
            current_teacher = teacher_dicts[current_teacher_id]
            start_of_stimulation = np.min(spikes) \
                                 + d_time \
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


def set_teachers_input(spikes_list, classes, teachers, settings):
    teacher_dicts = create_teacher_dict(spikes_list, classes,
                                        teachers, settings)
    for teacher in teacher_dicts:
        nest.SetStatus([teacher], teacher_dicts[teacher])


def create_poisson_noise(spikes_list, settings):  # Network
    # def get_poisson_train(time, firing_rate, h):
    #     np.random.seed()
    #     dt = 1.0 / 1000.0 * h
    #     times = np.arange(0, time, h)
    #     mask = np.random.random(int(time / h)) < firing_rate * dt
    #     spike_times = times[mask]
    #     return spike_times

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


def set_poisson_noise(spikes_list, spike_generators, settings):
    noise_dict = create_poisson_noise(spikes_list, settings)
    for input_neuron, generator in zip(noise_dict, spike_generators):
        nest.SetStatus([generator], [noise_dict[input_neuron]])


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
                if sender == [neuron_id] \
                        and [lat] < tmp_latency_dict[neuron_name]:
                    tmp_latency_dict[neuron_name] = [lat] 
        output_list.append(tmp_latency_dict)
    return output_list


def predict_from_latency(latency_list):
    output_list = []
    for latency in latency_list:
        tmp_list = [latency[neuron][:1] 
                    for neuron in sorted(latency.keys())]
        min_index, min_value = min(enumerate(tmp_list),
                                   key=operator.itemgetter(1))
        output_list.append(min_index)
    return output_list


def fitness_func_time(latency_list, data):
    fit_list = []

    for latency, y in zip(latency_list, data['class']):
        tmp_list = [latency[neuron_number][:1][0] 
                    for neuron_number in sorted(latency.keys())]
        latency_of_desired_neuron = tmp_list.pop(y)

        fit = -1 * latency_of_desired_neuron
        fit_list.append(fit)

    fitness_score = np.mean(fit_list)
    if np.isnan(fitness_score):
        fitness_score = 0
    return fitness_score


def fitness_func_sigma(latency_list, data):
    def sigmoid(x, alpha):
        return 1 / (1 + np.exp(-2 * alpha * x))

    fit_list = []
    for latency, y in zip(latency_list, data['class']):
        tmp_list = [latency[neuron_number][:1][0] 
                    for neuron_number in sorted(latency.keys())]
        latency_of_desired_neuron = tmp_list.pop(y)
        fit = 1
        for lat in tmp_list:
            fit *= sigmoid(lat - latency_of_desired_neuron, 0.1)
        fit_list.append(fit)
    fitness_score = np.mean(fit_list)
    if np.isnan(fitness_score):
        fitness_score = 0
    return fitness_score, fit_list


def fitness_func_exp(latency_list, data):
    fit_list = []
    for latency, y in zip(latency_list, data['class']):
        tmp_list = [latency[neuron_number][:1][0] 
                    for neuron_number in sorted(latency.keys())]
        latency_of_desired_neuron = tmp_list.pop(y)
        fit = 1
        for lat in tmp_list:
            fit -= exp(latency_of_desired_neuron - lat)
        fit_list.append(fit)
    fitness_score = np.mean(fit_list)
    if np.isnan(fitness_score):
        fitness_score = 0
    return fitness_score, fit_list


def save_weigths(layers, settings):
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
    data_test = {}
    data_train = {}
    data_out = {'train': {},
                'test': {}}
    if settings['data']['use_valid']:
        data_valid = {}
        data_out['valid'] = {}

        input_train, \
        input_valid, \
        y_train, \
        y_valid = train_test_split(data['input'][train_index],
                                   data['class'][train_index],
                                   test_size=settings['data']['valid_size'],
                                   random_state=42)
        data_train['input'] = input_train
        data_train['class'] = y_train

        data_valid['input'] = input_valid
        data_valid['class'] = y_valid

        data_out['valid']['full'] = data_valid
    else:
        data_train['input'] = data['input'][train_index]
        data_train['class'] = data['class'][train_index]

    # shuffle_train = False
    if not settings['network']['separate_networks'] \
          and not settings['data']['shuffle_train']:
        data_train['input'] = np.array([x for y, x in sorted(zip(data_train['class'], 
                                                                 data_train['input']), 
                                                             key=operator.itemgetter(0))])
        data_train['class'] = np.array(sorted(data_train['class']))

    data_test['input'] = data['input'][test_index]
    data_test['class'] = data['class'][test_index]

    data_out['train']['full'] = data_train
    data_out['test']['full'] = data_test
    return data_out


def train(settings, data):
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
            interconnect_layer(layer_hid, settings['model']['syn_dict_inh'])

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

    is_train = True
    d_time, \
    input_spikes = set_input_spikes(data['input'], 
                                    spike_generators_1,
                                    settings, is_train)

    set_teachers_input(input_spikes, data['class'],
                       teacher_1, settings)

    if settings['network']['noise_after_pattern']:
        nest.Connect(spike_generators_2, parrot_layer, 'one_to_one', 
                     syn_spec='static_synapse')
        set_poisson_noise(input_spikes, spike_generators_2, settings)

    nest.Simulate(d_time)

    layers = [parrot_layer, layer_hid, layer_out] \
        if settings['topology']['two_layers'] \
        else [parrot_layer, layer_out]

    spikes = nest.GetStatus(spike_detector_1,
                            keys="events")[0]['times'].tolist()
    senders = nest.GetStatus(spike_detector_1,
                             keys="events")[0]['senders'].tolist()

    weights = save_weigths(layers, settings)
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


def test(settings, data, weights):
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
        if settings['topology']['use_inhibition']:
            interconnect_layer(layer_hid,
                               settings['model']['syn_dict_inh'])

        nest.Connect(parrot_layer, layer_hid,
                     'all_to_all', syn_spec='static_synapse')
        nest.Connect(layer_hid, layer_out,
                     'all_to_all', syn_spec='static_synapse')
        nest.Connect(layer_hid, spike_detector_3, 'all_to_all')
        nest.SetStatus(layer_hid, settings['model']['neuron_hid'])
    else:
        if settings['topology']['use_inhibition']:
            interconnect_layer(layer_out, 
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
    full_time, input_spikes = set_input_spikes(data['input'],
                                               spike_generators_1,
                                               settings, is_train)
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


def test_data(data, weights, settings, comm):
    raw_latency, devices = test(settings, data, weights)
    comm.Barrier()
    all_latency = comm.allgather(raw_latency)
    raw_latency = merge_spikes_and_senders(all_latency)
    all_latency = split_spikes_and_senders(raw_latency,
                                           len(data['class']),
                                           settings)
    out_latency = convert_latency(all_latency, settings)
    return out_latency, devices


def prediction_score(y, predicton, settings):
    score = 0
    if settings['learning']['metrics'] == 'acc':
        score = accuracy_score(y, predicton)
    elif settings['learning']['metrics'] == 'f1':
        score = f1_score(y, predicton, average='micro')
    return score


def fitness(full_latency, data, settings):
    fitness_score = 0
    if settings['learning']['fitness_func'] == 'exp':
        fitness_score, fit_list = fitness_func_exp(full_latency,
                                                   data)
    elif settings['learning']['fitness_func'] == 'sigma':
        fitness_score, fit_list = fitness_func_sigma(full_latency,
                                                     data)
    elif settings['learning']['fitness_func'] == 'time':
        fitness_score = fitness_func_time(full_latency, data)
    elif settings['learning']['fitness_func'] == 'acc':
        y_valid = predict_from_latency(full_latency)
        fitness_score = accuracy_score(data['class'], y_valid)
    elif settings['learning']['fitness_func'] == 'f1':
        y_valid = predict_from_latency(full_latency)
        fitness_score = f1_score(data['class'], y_valid, average='micro')
    return fitness_score


def test_network_acc(data, settings):
    comm = MPI.COMM_WORLD

    data_train = data['train']['full']
    
    weights, \
    latency_train, \
    devices_train = train(settings, data_train)

    full_latency_test_train, \
    devices_test_train = test_data(data_train,
                                   weights,
                                   settings,
                                   comm)

    y_train = predict_from_latency(full_latency_test_train)
    score_train = prediction_score(data_train['class'], 
                                   y_train, settings)

    fitness_score = 0
    if settings['data']['use_valid']:
        data_valid = data['valid']['full']
        full_latency_valid, \
        devices_valid = test_data(data_valid,
                                  weights,
                                  settings,
                                  comm)
        if settings['learning']['use_fitness_func']:
            fitness_score = fitness(full_latency_valid, 
                                    data_valid, settings)
    else:
        fitness_score = score_train

    data_test = data['test']['full']
    full_latency_test, \
    devices_test = test_data(data_test,
                             weights,
                             settings,
                             comm)
    y_test = predict_from_latency(full_latency_test)
    score_test = prediction_score(data_test['class'], 
                                  y_test, settings)

    comm.Barrier()
    weights_all = comm.allgather(weights)

    out_dict = {
                'fitness_score': fitness_score,
                'acc_test': score_test,
                'acc_train': score_train,
                'output_list_test': y_test,
                'output_list_train': y_train,
                'latency_test': full_latency_test,
                'latency_train': full_latency_test_train,
                'train_classes': data_train['class'],
                'test_classes': data_test['class']
               }
    
    return out_dict, weights_all


def test_separate_neurons_acc(data, settings):
    def merge_spikes_from_separate_networks(separate_latency_list):
        out_latency = []
        num_neurons = len(separate_latency_list)
        data_len = len(separate_latency_list[0])
        for i in range(data_len):
            tmp_latency = {}
            for j in range(num_neurons):
                tmp_latency['neuron_' + str(j)] = separate_latency_list[j][i]['neuron_0']
            out_latency.append(tmp_latency)
        return out_latency

    comm = MPI.COMM_WORLD

    n_classes = len(set(data['train']['full']['class']))

    class_keys = []
    for i in range(n_classes):
        current_class = 'class_' + str(i)
        mask = data['train']['full']['class'] == i
        data['train'][current_class] = {}
        data['train'][current_class]['input'] = data['train']['full']['input'][mask]
        data['train'][current_class]['class'] = data['train']['full']['class'][mask]
        class_keys.append(current_class)
    
    latency_test_list = []
    latency_valid_list = []
    latency_test_train_list = []

    data_test = data['test']['full']
    for class_key in class_keys:
        data_train = data['train'][class_key]

        weights, latency_train, \
            devices_train = train(settings, data_train)

        full_latency_test_train, \
            devices_test_train = test_data(data['train']['full'], weights,
                                           settings, comm)
        latency_test_train_list.append(full_latency_test_train)

        full_latency_test, \
            devices_test = test_data(data_test, weights,
                                     settings, comm)
        latency_test_list.append(full_latency_test)

        if settings['data']['use_valid']:
            data_valid = data['valid']['full']
            full_latency_valid, \
                devices_valid = test_data(data_valid, weights,
                                          settings, comm)
            latency_valid_list.append(full_latency_valid)

    data_train = data['train']['full']
    merged_latency_test_train = merge_spikes_from_separate_networks(latency_test_train_list)
    y_train = predict_from_latency(merged_latency_test_train)
    score_train = prediction_score(data_train['class'], 
                                   y_train, settings)

    fitness_score = 0
    if settings['data']['use_valid']:
        data_valid = data['valid']['full']
        merged_latency_valid = merge_spikes_from_separate_networks(latency_valid_list)
        
        if settings['learning']['use_fitness_func']:
            fitness_score = fitness(merged_latency_valid, data_valid, settings)
    else:
        fitness_score = score_train

    merged_latency_test = merge_spikes_from_separate_networks(latency_test_list)
    y_test = predict_from_latency(merged_latency_test)
    score_test = prediction_score(data_test['class'], 
                                  y_test, settings)

    comm.Barrier()

    weights_all = comm.allgather(weights)

    out_dict = {
                'fitness_score': fitness_score,
                'acc_test': score_test,
                'acc_train': score_train,
                'output_list_test': y_test,
                'output_list_train': y_train,
                'latency_test': merged_latency_test,
                'latency_train': merged_latency_test_train,
                'train_classes': data_train['class'],
                'test_classes': data_test['class']
               }
    
    return out_dict, weights_all


def test_network_acc_cv(data, settings):
    def solve_fold(input_data):
        data_fold = prepare_data(input_data['data'],
                                 input_data['train_index'],
                                 input_data['test_index'],
                                 input_data['settings'])
        return test_network_acc(data_fold, input_data['settings'])

    def solve_fold_separate(input_data):
        data_fold = prepare_data(input_data['data'],
                                 input_data['train_index'],
                                 input_data['test_index'],
                                 input_data['settings'])
        return test_separate_neurons_acc(data_fold, input_data['settings'])

    fit = []
    weights = []
    acc_test = []
    acc_train = []
    data_list = []
    latency_test_list = []
    latency_train_list = []

    test_classes_list = []
    train_classes_list = []
    
    skf = StratifiedKFold(n_splits=settings['learning']['n_splits'])
    for train_index, test_index in skf.split(data['input'], data['class']):
        input_data = {
                      'data': data,
                      'settings': settings,
                      'test_index': test_index,
                      'train_index': train_index,
                     }
        data_list.append(input_data)

    fold_map = []
    if settings['network']['separate_networks']:
        fold_map = map(solve_fold_separate, data_list)
    else:
        fold_map = map(solve_fold, data_list)

    for result, weight in fold_map:
        acc_test.append(result['acc_test'])
        acc_train.append(result['acc_train'])
        fit.append(result['fitness_score'])
        latency_test_list.append(result['latency_test'])
        latency_train_list.append(result['latency_train'])
        test_classes_list.append(result['test_classes'])
        train_classes_list.append(result['train_classes'])
        weights.append(weight)

    # print(fit)
    out_dict = {
                'fitness_score': fit,
                'fitness_mean': np.mean(fit),
                'accs_test': acc_test,
                'accs_test_mean': np.mean(acc_test),
                'accs_test_std': np.std(acc_test),
                'accs_train': acc_train,
                'accs_train_mean': np.mean(acc_train),
                'accs_train_std': np.std(acc_train),
                'latency_test': latency_test_list,
                'latency_train': latency_train_list,
                'test_classes': test_classes_list,
                'train_classes': train_classes_list,
                'weights': weights,
               }
    return out_dict


def grid_search(data, parameters, settings):
    result = {
              'accuracy': [],
              'std': [],
              'fitness_score': [],
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

import nest

import numpy as np
import operator
import json
import sys

from math import exp
from mpi4py import MPI
from sklearn.model_selection import train_test_split, StratifiedKFold


def set_spike_in_generators(data, spike_generators, start_time, end_time, h_time, start_h):
    sp = []

    for gen_num in data:
        d_time = start_time
        sp_tmp = {'spike_times': [],
                  'spike_weights': []}

        while d_time < end_time:
            sp_tmp['spike_times'] += map(lambda x: x + d_time + start_h, data[gen_num])
            sp_tmp['spike_weights'] += np.ones_like(data[gen_num]).tolist()
            d_time += h_time
        # print gen_num, sp_tmp
        sp.append(sp_tmp)

    for gen_num, spp in zip(data, sp):
        # print gen_num, spike_generators_1[gen_num - 1]
        nest.SetStatus([spike_generators[gen_num - 1]], [spp])

        
def set_teacher_input(x, teacher, settings):  # Network
    ampl_times = []
    ampl_values = []

    h = settings['network']['h']
    
    y = x + 2 * h + settings['learning']['reinforce_time']  # x + 0.2
    ampl_times.append(x - h)  # x - 1.1
    ampl_times.append(y - h)  # x - 1.1
    ampl_values.append(settings['learning']['teacher_amplitude'])  # 1 mA 1000000.0
    ampl_values.append(0.0)  # 0 pA
    
    nest.SetStatus(teacher, {'amplitude_times': ampl_times,
                             'amplitude_values': ampl_values})

    
def count_acc(latency, data):
    acc = 0
    output_list = []

    for i in range(len(data['input'])):
        tmp_list = [latency[neuron_number][i]['latency'][:1] for neuron_number in latency]

        min_index, min_value = min(enumerate(tmp_list),
                                   key=operator.itemgetter(1))
        if min_index == data['class'][i]:
            acc += 1
        output_list.append([tmp_list, data['class'][i], min_index])

    acc = float(acc) / len(data['input'])
    # print acc
    return acc, output_list


def merge_raw_latency(raw_latency_list):
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


def save_latency_to_file(raw_latency, filename):
    rank = nest.Rank()
    with open((filename + str(rank) + '.json'), 'w') as latency_file:
        json.dump(raw_latency, latency_file, indent=4)


def create_latency(input_latency, data, settings):
    output_latency = []
    d_time = settings['network']['start_delta']
    for example, ex_class in zip(data['input'], data['class']):
        mask = (input_latency['spikes'] > d_time) & \
               (input_latency['spikes'] < d_time + settings['network']['h_time'])
        spikes_tmp = input_latency['spikes'][mask]
        senders_tmp = input_latency['senders'][mask]
        tmp_dict = {'latency': spikes_tmp - d_time,
                    'senders': senders_tmp,
                    'class': ex_class}

        d_time += settings['network']['h_time']
        output_latency.append(tmp_dict)
    return output_latency


def create_full_latency(latency, settings):
    full_latency = {}
    n_neurons = settings['topology']['n_layer_out']
    neuron_out_ids = [i + 1 for i in range(settings['topology']['n_layer_out'])]
    for i in range(n_neurons):
        tmp_str = 'neuron_' + str(i)
        full_latency[tmp_str] = []
    for latencies in latency:
        tmp_dicts = []
        for _ in range(n_neurons):
            tmp_dict = {'latency': [float('Inf')],
                        'class': latencies['class']}
            tmp_dicts.append(tmp_dict)
        for lat, sender in zip(latencies['latency'], latencies['senders']):
            for num, neuron_id in enumerate(neuron_out_ids):
                if sender == [neuron_id]: 
                    tmp_dicts[num]['latency'] = [lat]
        for latency_key, tmp_dict in zip(full_latency, tmp_dicts):
            full_latency[latency_key].append(tmp_dict)
    return full_latency


def fitness_func_time(latency, data):
    fit_list = []

    for i in range(len(data['input'])):
        tmp_list = [latency[neuron_number][i]['latency'][:1][0] for neuron_number in latency]
        latency_of_desired_neuron = tmp_list.pop(data['class'][i])

        fit = -1 * latency_of_desired_neuron
        fit_list.append(fit)

    fitness = np.mean(fit_list)
    if np.isnan(fitness):
        fitness = 0
    return fitness


def fitness_func_sigma(latency, data):
    def sigmoid(x, alpha):
        return 1 / (1 + np.exp(-2 * alpha * x))

    fit_list = []
    for i in range(len(data['input'])):
        tmp_list = [latency[neuron_number][i]['latency'][:1][0] for neuron_number in latency]
        latency_of_desired_neuron = tmp_list.pop(data['class'][i])
        fit = 1
        for lat in tmp_list:
            fit *= sigmoid(lat - latency_of_desired_neuron, 0.1)
        fit_list.append(fit)
    fitness = np.mean(fit_list)
    if np.isnan(fitness):
        fitness = 0
    return fitness


def fitness_func_exp(latency, data):
    fit_list = []
    for i in range(len(data['input'])):
        tmp_list = [latency[neuron_number][i]['latency'][:1][0] for neuron_number in latency]
        latency_of_desired_neuron = tmp_list.pop(data['class'][i])
        fit = 1
        for lat in tmp_list:
            fit -= exp(latency_of_desired_neuron - lat)
        fit_list.append(fit)
    fitness = np.mean(fit_list)
    if np.isnan(fitness):
        fitness = 0
    return fitness, fit_list


def save_weights_one_layer(input_layer, output_layer, settings):
    weights = {'layer_out': {}}
    for neuron_id in output_layer:
        tmp_weight = []
        for input_id in input_layer:
            conn = nest.GetConnections([input_id], [neuron_id], 
                                       synapse_model=settings['model']['syn_dict_stdp']['model'])
            weight_one = nest.GetStatus(conn, 'weight')
            if len(weight_one) != 0:
                tmp_weight.append(weight_one[0])
            # else:
            #     tmp_weight.append([])
        if len(tmp_weight) != 0:
            weights['layer_out'][neuron_id] = tmp_weight
    return weights


def save_weigths_two_layers(input_layer, hidden_layer, output_layer, settings):
    weights = {'layer_out': {},
               'layer_hid': {}}
    
    for neuron_id in hidden_layer:
        tmp_weight = []
        for input_id in input_layer:
            conn = nest.GetConnections([input_id], [neuron_id], 
                                       synapse_model=settings['model']['syn_dict_stdp_hid']['model'])
            weight_one = nest.GetStatus(conn, 'weight')
            if len(weight_one) != 0:
                tmp_weight.append(weight_one[0])
            # else:
            #     tmp_weight.append([])
        if len(tmp_weight) != 0:
            weights['layer_hid'][neuron_id] = tmp_weight
        
    for neuron_id in output_layer:
        tmp_weight = []
        for input_id in hidden_layer:
            conn = nest.GetConnections([input_id], [neuron_id], 
                                       synapse_model=settings['model']['syn_dict_stdp']['model'])
            weight_one = nest.GetStatus(conn, 'weight')
            if len(weight_one) != 0:
                tmp_weight.append(weight_one[0])
            # else:
            #     tmp_weight.append([])
        if len(tmp_weight) != 0:
            weights['layer_out'][neuron_id] = tmp_weight
    return weights


def weight_norm(weights):
    norm = []
    for neuron in weights:
        norm.append(np.linalg.norm(weights[neuron]))
    return np.linalg.norm(norm)


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
    last_norms = []
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

    nest.Simulate(settings['network']['start_delta'])
    while not early_stop:
        set_spike_in_generators(
            data['input'][i],
            spike_generators_1,
            d_time,
            d_time + settings['network']['h_time'],
            settings['network']['h_time'],
            settings['network']['h']
        )
        # if True:
        #     set_spike_in_generators(data['input'][i], spike_generators_1,
        #                             d_time + 1.5, d_time + settings['h_time'] + 1.5,
        #                             settings['h_time'], settings['h'])
        #     set_spike_in_generators(data['input'][i], spike_generators_1,
        #                             d_time + 3.0, d_time + settings['h_time'] + 3.0,
        #                             settings['h_time'], settings['h'])

        spike_times = []
        for neuron_number in data['input'][i]:
            if data['input'][i][neuron_number]:
                spike_times.append(data['input'][i][neuron_number][0])
        
        if settings['learning']['use_teacher']:
            if settings['topology']['n_layer_out'] == 1:
                set_teacher_input(
                    np.min(spike_times) \
                    + d_time \
                    + settings['network']['h'] \
                    + settings['learning']['reinforce_delta'],
                    teacher_1, 
                    settings
                )
            else:
                set_teacher_input(
                    np.min(spike_times) \
                    + d_time \
                    + settings['network']['h'] \
                    + settings['learning']['reinforce_delta'],
                    [teacher_1[data['class'][i]]],
                    settings
                )  

        if settings['network']['noise_after_pattern']:
            nest.SetStatus(
                poisson_layer,
                {'start': d_time + np.max(spike_times),
                 'stop': float(d_time + settings['network']['h_time']),
                 'rate': settings['network']['noise_freq']}
            )

        nest.Simulate(settings['network']['h_time'])
        
        ex_class = data['class'][i]
        spikes = nest.GetStatus(spike_detector_1, keys="events")[0]['times']
        senders = nest.GetStatus(spike_detector_1, keys="events")[0]['senders']
        mask = spikes > d_time
        spikes = spikes[mask]
        senders = senders[mask]
        tmp_dict = {
                    'latency': spikes - d_time,
                    'senders': senders,
                    'class': ex_class
                   }

        output_latency.append(tmp_dict)
        
        d_time += settings['network']['h_time']
        if i + hi + 1 > len(data['input']):
            i = 0
        else:
            i += hi
        
        if settings['network']['save_history']:
            if settings['topology']['two_layers']:
                tmp_weights = save_weigths_two_layers(parrot_layer, layer_hid,
                                                      layer_out, settings)
                tmp_norm_hid = weight_norm(tmp_weights['layer_hid'])
                tmp_norm_out = weight_norm(tmp_weights['layer_out'])
                tmp_norm = np.linalg.norm([tmp_norm_hid, tmp_norm_out])
                norm_history.append(tmp_norm)
            else:
                tmp_weights = save_weights_one_layer(parrot_layer, layer_out,
                                                     settings)
                tmp_norm_out = weight_norm(tmp_weights['layer_out'])
                norm_history.append(tmp_norm_out)
            weights_history.append(tmp_weights)

        #     if len(norm_history) > 5 * len(data['input']) and settings['early_stop']:
        #         early_stop = np.std(norm_history[-5 * len(data['input']):]) < 0.025
        # else:
        early_stop = d_time > full_time

    if settings['topology']['two_layers']:
        weights = save_weigths_two_layers(
            parrot_layer,
            layer_hid,
            layer_out,
            settings
        )
    else:
        weights = save_weights_one_layer(
            parrot_layer,
            layer_out,
            settings
        )
    
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
        'voltmeter', 1,
        {'withgid': True,
         'withtime': True}
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

    if settings['topology']['two_layers']:
        for neuron_id in weights['layer_hid']:
            connection = nest.GetConnections(parrot_layer, target=[neuron_id])
            nest.SetStatus(connection, 'weight', weights['layer_hid'][neuron_id])

        for neuron_id in weights['layer_out']:
            connection = nest.GetConnections(layer_hid, target=[neuron_id])
            nest.SetStatus(connection, 'weight', weights['layer_out'][neuron_id])
    else:
        for neuron_id in weights['layer_out']:
            connection = nest.GetConnections(parrot_layer, target=[neuron_id])
            nest.SetStatus(connection, 'weight', weights['layer_out'][neuron_id])

    np.random.seed(500)
    output_latency = []
    d_time = settings['network']['start_delta']
    nest.Simulate(settings['network']['start_delta'])

    for example, examples_class in zip(data['input'], data['class']):
        set_spike_in_generators(
            example,
            spike_generators_1,
            d_time,
            d_time + settings['network']['h_time'],
            settings['network']['h_time'],
            settings['network']['h'])
        # nest.SetStatus(poisson_layer, {'start': 30.})
        nest.Simulate(settings['network']['h_time'])
        d_time += settings['network']['h_time']

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
        raw_latency_valid = merge_raw_latency(all_latency_valid)
        latency_valid = create_latency(raw_latency_valid, data_valid, settings)
        full_latency_valid = create_full_latency(latency_valid, settings)

        if settings['learning']['use_fitness_func'] \
                and settings['learning']['fitness_func'] == 'exp':
            fitness, fit_list = fitness_func_exp(full_latency_valid, data_valid)
        elif settings['learning']['use_fitness_func'] \
                and settings['learning']['fitness_func'] == 'sigma':
            fitness = fitness_func_sigma(full_latency_valid, data_valid)
        elif settings['learning']['use_fitness_func'] \
                and settings['learning']['fitness_func'] == 'time':
            fitness = fitness_func_time(full_latency_valid, data_valid)
        elif settings['learning']['use_fitness_func'] \
                and settings['learning']['fitness_func'] == 'acc':
            fitness, out = count_acc(full_latency_valid, data_valid)
        # elif settings['use_fitness_func'] and settings['fitness_func'] == 'weights':
        #     final_weights = list(list(weights.values())[0].values())
        #     np.savetxt('final_weights.txt', final_weights)
        #     desired_weights = np.loadtxt('../desired_weights/final_weights.txt')
        #     fitness = -1 * np.linalg.norm(np.subtract(final_weights, desired_weights))

    raw_latency_test_train, devices_test_train = test(settings, data_train, weights)
    all_latency_test_train = comm.allgather(raw_latency_test_train)
    comm.Barrier()
    raw_latency_test_train = merge_raw_latency(all_latency_test_train)
    latency_test_train = create_latency(raw_latency_test_train, data_train, settings)
    full_latency_test_train = create_full_latency(latency_test_train, settings)
    acc_train, output_list_train = count_acc(full_latency_test_train, data_train)

    data_test = data['test']['full']
    raw_latency_test, devices_test = test(settings, data_test, weights)
    all_latency_test = comm.allgather(raw_latency_test)
    comm.Barrier()
    raw_latency_test = merge_raw_latency(all_latency_test)
    latency_test = create_latency(raw_latency_test, data_test, settings)
    full_latency_test = create_full_latency(latency_test, settings)
    acc_test, output_list_test = count_acc(full_latency_test, data_test)
    # print(output_list_test)
    comm.Barrier()

    weights_all = comm.allgather(weights)

    out_dict = {
                'fitness': fitness,
                'acc_test': acc_test,
                'acc_train': acc_train,
                'output_list_test': output_list_test,
                'output_list_train': output_list_train,
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
              'fitness'
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
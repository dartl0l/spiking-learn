# import csv
import nest

import numpy as np
import matplotlib.patches as mpatches
import operator

from sys import path
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold

# path.append("./")

#import andrews_curve
#import converter

def set_spike_in_generators(data, spike_generators, start_time, end_time, h_time, start_h):
    sp = []

    for gen_num in data:
        d_time = start_time
        sp_tmp = {'spike_times': [],
                  'spike_weights': []}

        while d_time < end_time:
            sp_tmp['spike_times'] += map(lambda x: x + d_time + start_h,
                                         data[gen_num])
            sp_tmp['spike_weights'] += np.ones_like(data[gen_num]).tolist()
            d_time += h_time
        # print gen_num, sp_tmp
        sp.append(sp_tmp)

    for gen_num, spp in zip(data, sp):
        # print gen_num, spike_generators_1[gen_num - 1]
        nest.SetStatus([spike_generators[gen_num - 1]], [spp])

        
def set_teacher_input(x, teach_input, settings):  # Network
    ampl_times = []
    ampl_values = []

    h = settings['h']
    
    y = x + 2 * h + settings['reinforce_time']  # x + 0.2
    ampl_times.append(x - h)  # x - 1.1
    ampl_times.append(y - h)  # x - 1.1
    ampl_values.append(settings['teacher_amplitude'])  # 1 mA 1000000.0
    ampl_values.append(0.0)  # 0 pA
    
    nest.SetStatus(teach_input, {'amplitude_times': ampl_times,
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


def create_full_latency(latency, neuron_out_ids):
    full_latency = {}
    n_neurons = len(neuron_out_ids)
    
    for i in range(n_neurons):
        tmp_str = 'neuron_' + str(i)
        full_latency[tmp_str] = []
    
#     print full_latency
    
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


def get_latency(t, h_time, h, max_time):
    return int(t / h) % int(h_time / h) * h / max_time


def interconnect_layer(layer, syn_dict):
    for neuron_1 in layer:
        for neuron_2 in layer:
            if neuron_1 != neuron_2:
                nest.Connect([neuron_1], [neuron_2], syn_spec=syn_dict)


def prepare_data(data, train_index, test_index, settings):
    data_train = {}
    data_test = {}
    data_out = {}

    if settings['use_valid']:
        data_valid = {}
        
        data_out = {'train': {},
                    'test': {},
                    'valid': {}}
        input_train, input_valid, y_train, y_valid = train_test_split(data['input'][train_index],
                                                                      data['class'][train_index],
                                                                      test_size=settings['valid_size'],
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


def prepare_data_genetic(data, train_index, test_index, settings):
    data_train = {}
    data_test = {}
    data_valid = {}
    
    data_out = {'train': {},
                'test': {},
                'valid': {}}

    input_train, input_valid, y_train, y_valid = train_test_split(data['input'][train_index],
                                                                  data['class'][train_index],
                                                                  test_size=settings['valid_size'],
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

    return data_out


def train(settings, data):
    nest.ResetKernel()
    np.random.seed()
    rng = np.random.randint(500)
    nest.SetKernelStatus({'local_num_threads': settings['num_threads'],
                          'resolution': settings['h'],
                          # 'total_num_virtual_procs': 1,
                          'rng_seeds': range(rng, rng + settings['num_threads'])})

    spike_generators_1 = nest.Create('spike_generator', settings['n_input'])

    poisson_layer = nest.Create('poisson_generator', settings['n_input'])

    spike_generators_2 = nest.Create('spike_generator', settings['n_input'])

    parrot_layer = nest.Create('parrot_neuron', settings['n_input'])

    layer_1 = nest.Create('iaf_psc_exp', settings['n_layer_1'])

    teacher_1 = nest.Create('step_current_generator', settings['n_layer_1'])
    # teacher_2 = nest.Create("step_current_generator")
    # teacher_3 = nest.Create("step_current_generator")

    spike_detector_1 = nest.Create('spike_detector')
    spike_detector_2 = nest.Create('spike_detector')

    voltmeter = nest.Create('voltmeter', 1,
                            {'withgid': True,
                             'withtime': True})
    
    if not settings['noise_after_pattern']:
	    nest.SetStatus(poisson_layer, {'rate': settings['noise_freq'],
	                                   'origin': 0.0})

    nest.Connect(spike_generators_1, parrot_layer,
                 'one_to_one', syn_spec='static_synapse')
    nest.Connect(poisson_layer, parrot_layer,
                 'one_to_one', syn_spec='static_synapse')
    
    if settings['use_teacher']:
        nest.Connect(teacher_1, layer_1,
                     'one_to_one', syn_spec='static_synapse')
    
    nest.Connect(layer_1, spike_detector_1, 'all_to_all')
    nest.Connect(parrot_layer, spike_detector_2, 'all_to_all')
    # nest.Connect(layer_hid, spike_detector_2, 'all_to_all')

    nest.Connect(voltmeter, layer_1)

    nest.SetStatus(layer_1, settings['neuron_out'])

    layer_hid = ()
    if settings['two_layers']:
        layer_hid = nest.Create('iaf_psc_exp', settings['n_layer_hid'])

        # if settings['use_inhibition']:
        #     interconnect_layer(layer_hid, settings['syn_dict_inh'])

        nest.Connect(parrot_layer, layer_hid,
                     'all_to_all', syn_spec=settings['syn_dict_stdp'])
        nest.Connect(layer_hid, layer_1,
                     'all_to_all', syn_spec=settings['syn_dict_stdp'])

        nest.SetStatus(layer_hid, settings['neuron_hid'])
    else:
        nest.Connect(parrot_layer, layer_1,
                     'all_to_all', syn_spec=settings['syn_dict_stdp'])
        if settings['use_inhibition']:
                interconnect_layer(layer_1, settings['syn_dict_inh'])

    np.random.seed(500)

    i = 0
    hi = 1
    
    last_norms = []
    norm_history = []
    output_latency = []
    weights_history = []
    
    nest.Simulate(settings['start_delta'])
    d_time = settings['start_delta']
    
    full_time = settings['epochs'] * len(data['input']) * settings['h_time'] + settings['start_delta']
    
    early_stop = False

    # while d_time < full_time:
    while not early_stop:
        set_spike_in_generators(data['input'][i], spike_generators_1,
                                d_time, d_time + settings['h_time'],
                                settings['h_time'], settings['h'])
        
        spike_times = []
        for neuron_number in data['input'][i]:
            if data['input'][i][neuron_number]:
                spike_times.append(data['input'][i][neuron_number][0])
        
        if settings['use_teacher']:
            if settings['n_layer_1'] == 1:
                set_teacher_input(np.min(spike_times) + d_time + settings['h'] + settings['reinforce_delta'],
                                  teacher_1, settings)
            else:
                # print np.min(spike_times) + d_time
                set_teacher_input(np.min(spike_times) + d_time + settings['h'] + settings['reinforce_delta'],
                                  [teacher_1[data['class'][i]]], settings)  

        if settings['noise_after_pattern']:
            nest.SetStatus(poisson_layer, {'start': d_time + np.max(spike_times),
                                           'stop': float(d_time + settings['h_time']),
                                           'rate': settings['noise_freq']})

        nest.Simulate(settings['h_time'])
        
        ex_class = data['class'][i]
        
        spikes = nest.GetStatus(spike_detector_1, keys="events")[0]['times']
        senders = nest.GetStatus(spike_detector_1, keys="events")[0]['senders']
        
        mask = spikes > d_time
        spikes = spikes[mask]
        senders = senders[mask]

        # print spikes

        tmp_dict = {'latency': spikes - d_time,
                    'senders': senders,
                    'class': ex_class}

        output_latency.append(tmp_dict)
        
        d_time += settings['h_time']
        if i + hi + 1 > len(data['input']):
            i = 0
        else:
            i += hi
        
        if settings['two_layers']:
            tmp_weights = {'layer_0': {},
                           'layer_hid': {}}

            norms = []
            for neuron_id in layer_hid:
                tmp_weight = []
                for input_id in parrot_layer:
                    conn = nest.GetConnections([input_id], [neuron_id], 
                                               synapse_model=settings['syn_dict_stdp']['model'])
                    weight_one = nest.GetStatus(conn, 'weight')
                    tmp_weight.append(weight_one[0])
                tmp_weights['layer_hid'][neuron_id] = tmp_weight
                norms.append(np.linalg.norm(tmp_weight))
            
            for neuron_id in layer_1:
                tmp_weight = []
                for input_id in layer_hid:
                    conn = nest.GetConnections([input_id], [neuron_id], 
                                               synapse_model=settings['syn_dict_stdp']['model'])
                    weight_one = nest.GetStatus(conn, 'weight')
                    tmp_weight.append(weight_one[0])
                tmp_weights['layer_0'][neuron_id] = tmp_weight
                norms.append(np.linalg.norm(tmp_weight))
            weights_history.append(tmp_weights)
            norm_history.append(np.linalg.norm(norms))
        else:
            tmp_weights = {'layer_0': {}}

            norms = []
            for neuron_id in layer_1:
                tmp_weight = []
                for input_id in parrot_layer:
                    conn = nest.GetConnections([input_id], [neuron_id], 
                                               synapse_model=settings['syn_dict_stdp']['model'])
                    weight_one = nest.GetStatus(conn, 'weight')
                    tmp_weight.append(weight_one[0])
                tmp_weights['layer_0'][neuron_id] = tmp_weight
                norms.append(np.linalg.norm(tmp_weight))
            weights_history.append(tmp_weights)
            norm_history.append(np.linalg.norm(norms))

        if len(norm_history) > 5 * len(data['input']):
            early_stop = np.std(norm_history[-5 * len(data['input']):]) < 0.025
        early_stop = d_time > full_time or early_stop

    weights = {}

    if settings['two_layers']:
        weights = {'layer_0': {},
                  'layer_hid': {}}
        
        for neuron_id in layer_hid:
            tmp_weight = []
            for input_id in parrot_layer:
                conn = nest.GetConnections([input_id], [neuron_id], 
                                           synapse_model=settings['syn_dict_stdp']['model'])
                weight_one = nest.GetStatus(conn, 'weight')
                tmp_weight.append(weight_one[0])
            weights['layer_hid'][neuron_id] = tmp_weight
            
        for neuron_id in layer_1:
            tmp_weight = []
            for input_id in layer_hid:
                conn = nest.GetConnections([input_id], [neuron_id], 
                                           synapse_model=settings['syn_dict_stdp']['model'])
                weight_one = nest.GetStatus(conn, 'weight')
                tmp_weight.append(weight_one[0])
            weights['layer_0'][neuron_id] = tmp_weight
    else:
        weights = {'layer_0': {}}
        for neuron_id in layer_1:
            tmp_weight = []
            for input_id in parrot_layer:
                conn = nest.GetConnections([input_id], [neuron_id], 
                                           synapse_model=settings['syn_dict_stdp']['model'])
                weight_one = nest.GetStatus(conn, 'weight')
                tmp_weight.append(weight_one[0])
            weights['layer_0'][neuron_id] = tmp_weight
    
    devices = {
               'voltmeter': voltmeter,
               'spike_detector_1': spike_detector_1,
               'spike_detector_2': spike_detector_2,
              }

    return weights, output_latency, devices, weights_history, norm_history


def test(settings, data, weights):
    nest.ResetKernel()
    np.random.seed()
    rng = np.random.randint(500)
    nest.SetKernelStatus({'local_num_threads': settings['num_threads'],
                          'resolution': settings['h'],
                          # 'total_num_virtual_procs': 1,
                          'rng_seeds': range(rng, rng + settings['num_threads'])})

    spike_generators_1 = nest.Create('spike_generator', settings['n_input'])

    poisson_layer = nest.Create('poisson_generator', settings['n_input'])

    spike_generators_2 = nest.Create('spike_generator', settings['n_input'])

    parrot_layer = nest.Create('parrot_neuron', settings['n_input'])

    layer_1 = nest.Create('iaf_psc_exp', settings['n_layer_1'])

    spike_detector_1 = nest.Create('spike_detector')
    spike_detector_2 = nest.Create('spike_detector')

    voltmeter = nest.Create('voltmeter', 1,
                            {'withgid': True,
                             'withtime': True})

    nest.Connect(spike_generators_1, parrot_layer,
                 'one_to_one', syn_spec='static_synapse')

    if settings['test_with_noise']:
        nest.SetStatus(poisson_layer, {'rate': settings['noise_freq']})
        nest.Connect(poisson_layer, parrot_layer,
                    'one_to_one', syn_spec='static_synapse')

    nest.Connect(layer_1, spike_detector_1, 'all_to_all')
    nest.Connect(parrot_layer, spike_detector_2, 'all_to_all')
    # nest.Connect(layer_hid, spike_detector_2, 'all_to_all')

    if settings['use_inhibition']:
        interconnect_layer(layer_1, settings['syn_dict_inh'])

    nest.Connect(voltmeter, layer_1)

    nest.SetStatus(layer_1, settings['neuron_out'])

    if settings['two_layers']:

        layer_hid = nest.Create('iaf_psc_exp', settings['n_layer_hid'])

#         interconnect_layer(layer_hid, settings['syn_dict_inh'])

        nest.Connect(parrot_layer, layer_hid,
                     'all_to_all', syn_spec='static_synapse')
        nest.Connect(layer_hid, layer_1,
                     'all_to_all', syn_spec='static_synapse')

        nest.SetStatus(layer_hid, settings['neuron_hid'])
    else:
        nest.Connect(parrot_layer, layer_1,
                     'all_to_all', syn_spec='static_synapse')

    for neuron_id in weights['layer_0']:
        nest.SetStatus(nest.GetConnections(parrot_layer,
                                           target=[neuron_id]),
                       'weight', weights['layer_0'][neuron_id])
    if settings['two_layers']:
        for neuron_id in weights['layer_hid']:
            nest.SetStatus(nest.GetConnections(layer_hid,
                                               target=[neuron_id]),
                           'weight', weights['layer_hid'][neuron_id])

    np.random.seed(500)

    # i = 0
    # hi = 1
    # d_time = 0

    output_latency = []

    nest.Simulate(settings['start_delta'])
    
    d_time = settings['start_delta']

    for example, ex_class in zip(data['input'], data['class']):
        set_spike_in_generators(example, spike_generators_1,
                                d_time, d_time + settings['h_time'],
                                settings['h_time'], settings['h'])
        #     nest.SetStatus(poisson_layer, {'start': 30.})
        nest.Simulate(settings['h_time'])

        # print nest.GetStatus(spike_detector_1, keys="events")
        
        spikes = nest.GetStatus(spike_detector_1, keys="events")[0]['times']
        senders = nest.GetStatus(spike_detector_1, keys="events")[0]['senders']

        mask = spikes > d_time
        spikes = spikes[mask]
        senders = senders[mask]

        # print spikes

        tmp_dict = {'latency': spikes - d_time,
                    'senders': senders,
                    'class': ex_class}

        output_latency.append(tmp_dict)

        d_time += settings['h_time']

    devices = {
               'voltmeter': voltmeter,
               'spike_detector_1': spike_detector_1,
               'spike_detector_2': spike_detector_2,
              }
    return output_latency, devices


def test_network_acc_for_genetic(data, settings):
    data_train = data['train']['full']
    data_test = data['test']['full']
    data_valid = data['valid']['full']

    weights, latency_train, devices_train, weights_history, norm_history = train(settings, data_train)

    latency_valid, devices_valid = test(settings, data_valid, weights)

    full_latency_valid = create_full_latency(latency_valid, settings['neuron_out_ids'])

    fitness = 0
    if settings['use_fitness_func'] and settings['fitness_func'] == 'sigma':
        fitness = fitness_func_sigma(full_latency_valid, data_valid)
    elif settings['use_fitness_func'] and settings['fitness_func'] == 'time':
        fitness = fitness_func_time(full_latency_valid, data_valid)
    elif settings['use_fitness_func'] and settings['fitness_func'] == 'acc':
        fitness, out = count_acc(full_latency_valid, data_valid)
    elif settings['use_fitness_func'] and settings['fitness_func'] == 'weights':
        # Get rid of these nasty nested dictionaries
        final_weights = list(list(weights.values())[0].values())
        np.savetxt('final_weights.txt', final_weights)
        desired_weights = np.loadtxt('../desired_weights/final_weights.txt')
        fitness = -1 * np.linalg.norm(np.subtract(final_weights, desired_weights))

    latency_test_train, devices_test_train = test(settings, data_train, weights)
    full_latency_test_train = create_full_latency(latency_test_train, settings['neuron_out_ids'])
    acc_train, output_list_train = count_acc(full_latency_test_train, data_train)

    latency_test, devices_test = test(settings, data_test, weights)
    full_latency_test = create_full_latency(latency_test, settings['neuron_out_ids'])
    acc_test, output_list_test = count_acc(full_latency_test, data_test)

    out_dict = {
                'fitness': fitness,
                'acc_test': acc_test,
                'acc_train': acc_train,
                'output_list_test': output_list_test,
                'output_list_train': output_list_train,
                }
    
    return out_dict


def test_network_acc_cv_for_genetic(data, settings):
    acc_test = []
    acc_train = []
    fit = []
    skf = StratifiedKFold(n_splits=settings['n_splits'])

    for train_index, test_index in skf.split(data['input'], data['class']):
        data_fold = prepare_data_genetic(data, train_index, test_index, settings)
        result_dict = test_network_acc_for_genetic(data_fold, settings)
        acc_test.append(result_dict['acc_test'])
        acc_train.append(result_dict['acc_train'])
        fit.append(result_dict['fitness'])

    print(fit)
    out_dict = {
                'fitness': fit,
                'fitness_mean': np.mean(fit),
                'accs_test': acc_test,
                'accs_test_mean': np.mean(acc_test),
                'accs_test_std': np.std(acc_test),
                'accs_train': acc_train,
                'accs_train_mean': np.mean(acc_train),
                'accs_train_std': np.std(acc_train),
                }
    return out_dict


def test_parameter(data, parameters, settings, n_times):
    result = {'accuracy': [],
              'std': [],
              'parameter': [],
              'parameter_name': [],
              }
    settings_copy = settings
    for _ in range(n_times):
        for key in parameters.keys():
            if isinstance(parameters[key], dict):
                for key_key in parameters[key].keys():
                    result['parameter'].append(settings_copy[key][key_key])
                    result['parameter_name'].append(key_key)
                    acc, std = test_3_neuron_acc_cv(data, settings_copy)
                    result['accuracy'].append(acc)
                    result['std'].append(std)
                    settings_copy[key][key_key] += parameters[key][key_key]
            else:
                result['parameter'].append(settings_copy[key])
                result['parameter_name'].append(key)
                acc, std = test_3_neuron_acc_cv(data, settings_copy)
                result['accuracy'].append(acc)
                result['std'].append(std)
                settings_copy[key] += parameters[key]
    return result


def test_parameter_network(data, parameters, settings, n_times):
    result = {'accuracy': [],
              'std': [],
              'parameter': [],
              'parameter_name': [],
              }
    settings_copy = settings
    for _ in range(n_times):
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
                settings_copy[key] += parameters[key]
    return result

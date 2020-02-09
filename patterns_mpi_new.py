# coding: utf-8

import sys




# class Settings(object):
#     """docstring for Settings"""
#     def __init__(self, settings_dict):
#         super(Settings, self).__init__()
#         self.arg = settings_dict


# class Topology(object):
#     """docstring for Topology"""
#     def __init__(self, settings):
#         super(Topology, self).__init__()
#         self.settings = settings


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


def weight_norm(weights):
    norms = []
    for weight in weights:
        norm = []
        for neuron in weight:
            norm.append(np.linalg.norm(weights[neuron]))
        norms.append(np.linalg.norm(norm))
    return np.linalg.norm(norms)


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

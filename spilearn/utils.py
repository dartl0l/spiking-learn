import json
import math
import pickle
import numpy as np

from functools import partial

from hyperopt import hp, fmin, tpe, space_eval, Trials


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def optimize(X, y, func, space, path, filename, new_trial=True, max_evals=100, h_evals=10, X_test=None, y_test=None):
    trials = Trials() if new_trial else pickle.load(open(path + "/" + filename, "rb"))

    n_evals = h_evals
    while n_evals <= max_evals:
        best = fmin(
            fn=partial(func, X=X, y=y, path=path, X_test=X_test, y_test=y_test), 
            space=space, algo=tpe.suggest, 
            trials=trials, max_evals=n_evals
        )
        pickle.dump(trials, open(path + "/" + filename, "wb"))
        best_space = space_eval(space, best)
        json.dump(best_space, open(path + '/best_space.json', 'w'), indent=4, cls=NpEncoder)
        n_evals += h_evals
    best_space = space_eval(space, best)
    json.dump(best_space, open(path + '/best_space.json', 'w'), indent=4, cls=NpEncoder)
    return best_space


def round_decimals(value):
    i = 0
    while value < 1:
        value *= 10
        i += 1
    return i


def print_settings(settings):
    for key in settings:
        if isinstance(settings[key], dict):
            print(key)
            for parameter in settings[key]:
                if isinstance(settings[key][parameter], dict):
                    print('\t' + parameter)
                    for parameter2 in settings[key][parameter]:
                        print('\t\t' + parameter2 + ' : ' + str(settings[key][parameter][parameter2]))
                else:
                    print('\t' + parameter + ' : ' + str(settings[key][parameter]))
        else:
            print('\t' + key + ' : ' + str(settings[key]))


def split_spikes_and_senders(input_latency, n_examples, start_delta, h_time):
    output_latency = []
    d_time = start_delta
    input_latency['spikes'] = np.array(input_latency['spikes'])
    input_latency['senders'] = np.array(input_latency['senders'])
    for _ in range(n_examples):
        mask = (input_latency['spikes'] > d_time) & \
                (input_latency['spikes'] < d_time + h_time)
        spikes_tmp = input_latency['spikes'][mask]
        senders_tmp = input_latency['senders'][mask]
        tmp_dict = {
                    'spikes': spikes_tmp - d_time,
                    'senders': senders_tmp
                    }

        d_time += h_time
        output_latency.append(tmp_dict)    
    return output_latency


def convert_latency(latency_list, n_neurons):
    output_array = []
    for latencies in latency_list:
        tmp_list = [np.nan] * n_neurons
        senders = set(latencies['senders'])
        for sender in senders:
            mask = latencies['senders'] == sender
            tmp_list[sender - 1] = latencies['spikes'][mask][0]
        output_array.append(tmp_list)
    return output_array


def convert_latency_pool(latency_list, n_neurons, pool_size):
    output_array = convert_latency(latency_list, n_neurons)
    output_array = np.array(output_array).reshape(
        len(output_array),
        int(n_neurons / pool_size),
        pool_size
    )
    output_array = np.mean(output_array, axis=2)
    return output_array


def predict_from_latency(latency_list, func=np.nanargmin):
    latency_list = np.array(latency_list)
    mask = np.logical_not(np.all(np.isnan(latency_list), axis=1))
    prediction = np.zeros(len(latency_list))
    prediction[mask] = func(latency_list[mask], axis=1)
    return prediction


def convert_layer_weights(weights):
    out_weights = {}
    for neuron_id in weights:
        out_weights[neuron_id] = np.zeros(len(weights[neuron_id]))
        for i, input_id in enumerate(weights[neuron_id]):
            out_weights[neuron_id][i] = weights[neuron_id][input_id]
    return out_weights


def convert_weights(weights):
    out_weights = []
    for layer_weights in weights:
        out_weights.append(convert_layer_weights(layer_weights))
    return out_weights


def fitness_func_time(latency_list, Y):
    fit_list = []

    for latency, y in zip(latency_list, Y):
        latency_of_desired_neuron = latency.pop(y)

        fit = -1 * latency_of_desired_neuron
        fit_list.append(fit)

    fitness_score = np.mean(fit_list)
    if np.isnan(fitness_score):
        fitness_score = 0
    return fitness_score


def fitness_func_sigma(latency_list, Y):
    def sigmoid(x, alpha):
        return 1 / (1 + np.exp(-2 * alpha * x))

    fit_list = []
    for latency, y in zip(latency_list, Y):
        latency_of_desired_neuron = latency.pop(y)
        fit = 1
        for lat in latency:
            fit *= sigmoid(lat - latency_of_desired_neuron, 0.1)
        fit_list.append(fit)
    fitness_score = np.mean(fit_list)
    if np.isnan(fitness_score):
        fitness_score = 0
    return fitness_score, fit_list


def fitness_func_exp(latency_list, Y):
    fit_list = []
    for latency, y in zip(latency_list, Y):
        latency_of_desired_neuron = latency.pop(y)
        fit = 1
        for lat in latency:
            fit -= math.exp(latency_of_desired_neuron - lat)
        fit_list.append(fit)
    fitness_score = np.mean(fit_list)
    if np.isnan(fitness_score):
        fitness_score = 0
    return fitness_score, fit_list

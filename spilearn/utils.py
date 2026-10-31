import os
import pickle

import numpy as np

from struct import unpack

from scipy.stats import mode

from .experiments._tuning import (
    optimize_ray,
    run_cv,
    run_experiment,
    run_train_test,
)


def get_labeled_data(picklename, MNIST_data_path, bTrain=True):
    """Read input-vector (image) and target class (label, 0-9) and return
    it as list of tuples.
    """
    picklename = MNIST_data_path + picklename
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename, 'rb'))
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(MNIST_data_path + 'train-images.idx3-ubyte', 'rb')
            labels = open(MNIST_data_path + 'train-labels.idx1-ubyte', 'rb')
        else:
            images = open(MNIST_data_path + 't10k-images.idx3-ubyte', 'rb')
            labels = open(MNIST_data_path + 't10k-labels.idx1-ubyte', 'rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]
        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        for i in range(N):
            if i % 1000 == 0:
                print('i: %i' % i)
            x[i] = [
                [unpack('>B', images.read(1))[0] for unused_col in range(cols)]
                for unused_row in range(rows)
            ]
            y[i] = unpack('>B', labels.read(1))[0]
        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open('%s.pickle' % picklename, 'wb'))
    return data


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
                        print(
                            '\t\t'
                            + parameter2
                            + ' : '
                            + str(settings[key][parameter][parameter2])
                        )
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
        mask = (input_latency['spikes'] > d_time) & (
            input_latency['spikes'] < d_time + h_time
        )
        spikes_tmp = input_latency['spikes'][mask]
        senders_tmp = input_latency['senders'][mask]
        tmp_dict = {'spikes': spikes_tmp - d_time, 'senders': senders_tmp}

        d_time += h_time
        output_latency.append(tmp_dict)
    return output_latency


def convert_latency(latency_list, n_neurons, offset=1):
    output_array = []
    for latencies in latency_list:
        tmp_list = [np.nan] * n_neurons
        senders_array = np.array(latencies['senders'])
        spikes_array = np.array(latencies['spikes'])
        senders = set(senders_array)
        for sender in senders:
            mask = senders_array == sender
            tmp_list[int(sender) - offset] = spikes_array[mask][0]
        output_array.append(tmp_list)
    return output_array


def convert_latency_pool_mean(latency_list, n_neurons, pool_size):
    output_array = convert_latency(latency_list, n_neurons)
    output_array = np.array(output_array).reshape(
        len(output_array), int(n_neurons / pool_size), pool_size
    )
    output_array = np.mean(output_array, axis=2)
    return output_array


def predict_from_latency(latency_list, func=np.nanargmin):
    latency_list = np.array(latency_list)
    mask = np.logical_not(np.all(np.isnan(latency_list), axis=1))
    prediction = np.zeros(len(latency_list))
    prediction[mask] = func(latency_list[mask], axis=1)
    return prediction


def convert_latency_pool(latency_list, n_neurons, pool_size):
    output_array = convert_latency(latency_list, n_neurons)
    output_array = np.array(output_array).reshape(
        len(output_array), int(n_neurons / pool_size), pool_size
    )
    return output_array


def predict_from_latency_pool(latency_list, func=np.nanargmin):
    latency_list = np.array(latency_list)
    mask = np.isnan(latency_list)
    mask = np.all(mask, axis=1)
    if np.any(mask):
        prediction = np.full(len(latency_list), -1, dtype=int)
        valid = np.logical_not(mask).all(axis=1)
        if np.any(valid):
            predictions = func(latency_list[valid], axis=1)
            prediction[valid] = mode(predictions, axis=1).mode.flatten()
        return prediction

    predictions = func(latency_list, axis=1)
    prediction = mode(predictions, axis=1).mode.flatten()
    return prediction


def predict_from_earliest_latency_pool(latency_list, func=np.argmin):
    latency_list = np.array(latency_list)
    class_latency = np.min(
        np.where(np.isnan(latency_list), np.inf, latency_list),
        axis=2,
    )
    valid = np.isfinite(class_latency).any(axis=1)
    prediction = np.full(len(latency_list), -1, dtype=int)
    prediction[valid] = func(class_latency[valid], axis=1)
    return prediction


CLASS_POOL_DECODERS = ('vote', 'mean_latency', 'min_latency')


def predict_from_class_pool(latency_list, decoder='min_latency'):
    """Decode class pools; silent outputs are NaN/inf and all-silent is -1."""
    if decoder not in CLASS_POOL_DECODERS:
        raise ValueError(f'Unknown class-pool decoder: {decoder!r}')
    if decoder == 'min_latency':
        return predict_from_earliest_latency_pool(latency_list)

    latency = np.asarray(latency_list, dtype=float)
    latency = np.where(np.isnan(latency), np.inf, latency)
    earliest = latency.min(axis=2)
    valid = np.isfinite(latency).any(axis=(1, 2))
    if decoder == 'vote':
        scores = np.isfinite(latency).sum(axis=2)
        tied = scores == scores.max(axis=1, keepdims=True)
    else:
        scores = latency.mean(axis=2)
        tied = scores == scores.min(axis=1, keepdims=True)
    # argmin chooses the lowest class ID if earliest latencies also tie.
    prediction = np.where(tied, earliest, np.inf).argmin(axis=1)
    prediction[~valid] = -1
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

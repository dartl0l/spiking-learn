import os
import copy
import json
import math
import pickle

import numpy as np

from struct import unpack

from scipy.stats import mode

from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

from ray import train, tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def get_labeled_data(picklename, MNIST_data_path, bTrain = True):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    picklename = MNIST_data_path + picklename
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename, 'rb'))
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(MNIST_data_path + 'train-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 'train-labels.idx1-ubyte','rb')
        else:
            images = open(MNIST_data_path + 't10k-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 't10k-labels.idx1-ubyte','rb')
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
                print("i: %i" % i)
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in range(cols)]  for unused_row in range(rows) ]
            y[i] = unpack('>B', labels.read(1))[0]
        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("%s.pickle" % picklename, 'wb'))
    return data


def run_train_test(params, X, y, model_in, create_pipe, X_test=None, y_test=None):
    model = copy.deepcopy(model_in)

    pipe = create_pipe(model, params)

    pipe.fit(X, y)
    y_test_pred = pipe.predict(X_test)
    train.report({"mean_accuracy": f1_score(y_test, y_test_pred, average='macro')})


def run_cv(params, X, y, model_in, create_pipe, X_test=None, y_test=None):
    model = copy.deepcopy(model_in)

    pipe = create_pipe(model, params)

    scores = cross_val_score(
        pipe, X, y, cv=5, scoring='f1_macro', n_jobs=5)
    train.report({"mean_accuracy": scores.mean()})


def optimize_ray(X, y, func, create_pipe, space, model, exp_name, new_trial=True, max_evals=100, 
                 X_test=None, y_test=None, max_concurrent=11):
    ray_path = os.path.expanduser("~/ray_results")
    exp_dir = os.path.join(ray_path, exp_name)

    if tune.Tuner.can_restore(exp_dir) and not new_trial:
        tuner = tune.Tuner.restore(
            exp_dir,
            trainable=tune.with_parameters(
                func, X=X, y=y, model_in=model,
                create_pipe=create_pipe, X_test=X_test, y_test=y_test
            ),
            resume_errored=True)
    else:
        search_alg = OptunaSearch(seed=42)

        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_concurrent)
        tuner = tune.Tuner(
            tune.with_parameters(
                func, X=X, y=y, model_in=model,
                create_pipe=create_pipe, X_test=X_test, y_test=y_test
            ),
            tune_config=tune.TuneConfig(
                num_samples=max_evals,
                search_alg=search_alg,
                scheduler=ASHAScheduler(),
                metric="mean_accuracy",
                mode="max",
            ),
            run_config=train.RunConfig(
                name=exp_name, 
            ),
            param_space=space
        )
    return tuner.fit()


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


def convert_latency_pool_mean(latency_list, n_neurons, pool_size):
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


def convert_latency_pool(latency_list, n_neurons, pool_size):
    output_array = convert_latency(latency_list, n_neurons)
    output_array = np.array(output_array).reshape(
        len(output_array),
        int(n_neurons / pool_size),
        pool_size
    )
    return output_array


def predict_from_latency_pool(latency_list, func=np.nanargmin):
    predictions = func(latency_list, axis=1)
    prediction = mode(predictions, axis=1).mode.flatten()
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

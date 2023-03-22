import json
import pickle
import numpy as np

from functools import partial

from hyperopt import hp, fmin, tpe, space_eval, Trials
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

from spilearn.estimators import *


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def run(params, X, y, path):
    settings = json.load(open(path + '/settings.json', 'r'))
    model = json.load(open(path + '/model.json', 'r'))

    settings['data']['scale'] = params['scale']
    model['neuron_out']['V_th'] = params['V_th']
    model['neuron_out']['tau_m'] = params['tau_m']

    model['syn_dict_inh']['weight'] = params['inh_weight']
    # settings['data']['n_coding_neurons'] = n_coding_neurons
    # settings['topology']['n_input'] = settings['data']['n_coding_neurons'] * len(X[0])
    settings['topology']['n_layer_out'] = params['n_layer_out']
    settings['network']['noise_freq'] = 0.

    round_to = 2
    
    trans = ReceptiveFieldsTransformer(
        settings['data']['n_coding_neurons'],
        settings['data']['coding_sigma'], 
        round_to,
        scale=settings['data']['scale'],
        reverse=False,
        no_last=False,
        reshape=True
    )
    net = UnsupervisedTemporalTransformer(settings, model)
    ev = FirstSpikeVotingClassifier(settings['network']['h_time'])
    pipe = make_pipeline(
        trans, 
        net, 
        ev
    )
    scores = cross_val_score(
        pipe, X, y, cv=5, scoring='f1_macro')

    return -1.0 * scores.mean()


def optimize(X, y, space, path, filename, new_trial=True, max_evals=100, h_evals=10):
    if new_trial:
        pickle.dump(Trials(), open(path + "/" + filename, "wb"))

    n_evals = h_evals
    while n_evals <= max_evals:
        trials = pickle.load(open(path + "/" + filename, "rb"))
        best = fmin(
            fn=partial(run, X=X, y=y, path=path), 
            space=space, algo=tpe.suggest, 
            trials=trials, max_evals=n_evals
        )
        pickle.dump(trials, open(path + "/" + filename, "wb"))
        best_space = space_eval(space, best)
        print(best_space)
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

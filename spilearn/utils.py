import numpy as np


def round_decimals(value):
    i = 0
    while value < 1:
        value *= 10
        i += 1
    return i


def print_settings(settings):
    for key in settings:
        print(key)
        for parameter in settings[key]:
            if isinstance(settings[key][parameter], dict):
                print('\t' + parameter)
                for parameter2 in settings[key][parameter]:
                    print('\t\t' + parameter2 + ' : ' + str(settings[key][parameter][parameter2]))
            else:
                print('\t' + parameter + ' : ' + str(settings[key][parameter]))


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

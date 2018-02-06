import csv
import nest
import nest.raster_plot
import nest.voltage_trace
import pylab as pl
import numpy as np
import matplotlib.patches as mpatches

import operator

from sys import path
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


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


def convert_data_to_patterns(x, y, pattern_time, h):
    input_for_nn = []
#     h = 0.1
#     time = 60
    rad = 2 * np.pi
    rad_to_deg = 57

    for xx, yy in zip(x, y):
        pattern = {}
        curve = AndrewsCurve(xx, rad, h)
#         pattern['times'] = np.round(curve.theta() * rad_to_deg, 1)
        pattern['times'] = np.round(curve.theta() * (pattern_time / rad), 1)

        pattern['neurons'] = map(int, curve.discrete_curve())
        pattern['class'] = int(yy)
        input_for_nn.append(pattern)

    min_neuron = 0

    for pattern in input_for_nn:
        if np.amin(pattern['neurons']) < min_neuron:
            min_neuron = np.amin(pattern['neurons'])

    for pattern in input_for_nn:
        pattern['neurons'] += np.abs(min_neuron)

    output = []
    for pattern in input_for_nn:
        tmp_dic = {}
        input_dic = dict.fromkeys(pattern['neurons'])

        for key in input_dic:
            input_dic[key] = []

        for neuron, time in zip(pattern['neurons'], pattern['times']):
            input_dic[neuron].append(time)

        tmp_dic['input'] = input_dic
        tmp_dic['class'] = pattern['class']

        output.append(tmp_dic)
    return output


def convert_data_to_patterns_spatio_temp(x, y, min_time, h):
    rad = 2 * np.pi
    rad_to_deg = 57

    output = []
    for xx, yy in zip(x, y):
        pattern = {}

        curve = AndrewsCurve(xx, rad, h).curve()

        tmp_dict = dict.fromkeys(np.arange(0, int(rad / h)))

        for i in xrange(int(rad / h)):
            tmp_dict[i] = np.round([(curve[i] + min_time) * 20], 1)

        pattern['input'] = tmp_dict
        pattern['class'] = yy

        output.append(pattern)
    return output


def convert_data_to_patterns_uniform(x, y, pattern_time, h, mult):
    output = []

    for xx, yy in zip(x, y):
        pattern = {}

        tmp_dict = dict.fromkeys(np.arange(0, len(xx) * mult))

        i = 0

        for x in xx:
            time = x * pattern_time

            for j in xrange(mult):
                tmp_dict[i] = np.sort(np.random.uniform(0, int(time), int(time / h)) + 0.1)
                # get_poisson_train(time, firing_rate, h)
                # tmp_dict[i] = get_poisson_train(time, firing_rate, h)
                i += 1

        pattern['input'] = tmp_dict
        pattern['class'] = yy

        output.append(pattern)
    return output


def get_gaussian(x, sigma2, mu):
    return (1 / np.sqrt(2 * sigma2 * np.pi)) * np.e ** (- (x - mu) ** 2 / (2 * sigma2))


def plot_field(sigma2, max_x, n_fields):
    h_mu = max_x / (n_fields - 1)
    mu = 0
    for i in xrange(n_fields):
        xx = np.arange(0, max_x, 0.01)
        yy = [get_gaussian(j, sigma2, mu) for j in xx]
        pl.plot(xx, yy)
        # left += h_mu
        # right += h_mu
#         print mu
        mu += h_mu
    pl.xlabel('Value $x$ of the input vector component')
    pl.ylabel('Spike times of pattern')
    pl.show()


def convert_data_to_patterns_gaussian_receptive_field(x, y, sigma2, max_x,
                                                      n_fields, k_time, k_round, reverse):
    output = {'input': [],
              'class': []}

    h_mu = max_x / (n_fields - 1)

    max_y = np.round(get_gaussian(h_mu, sigma2, h_mu), 0)

    for xx, yy in zip(x, y):
        tmp_dict = dict.fromkeys(np.arange(0, n_fields * len(xx)))

        tmp_fields = 0
        for x in xx:
            mu = 0
            for i in xrange(n_fields):
                time = np.round(get_gaussian(x, sigma2, mu), k_round)
                # time = get_gaussian(x, sigma2, mu)
                if time > 0.09:
                    if reverse:
                        tmp_dict[i + tmp_fields] = [k_time * (max_y - time)]
                    else:
                        tmp_dict[i + tmp_fields] = [k_time * time]
                else:
                    tmp_dict[i + tmp_fields] = []

                mu += h_mu
            tmp_fields += n_fields
        output['input'].append(tmp_dict)
        output['class'].append(yy)
    output = {'input': np.array(output['input']),
              'class': np.array(output['class'])}
    return output, max_y


def get_poisson_train(time, firing_rate, h):
    np.random.seed()
    dt = 1.0 / 1000.0 * h
    times = np.arange(0, time, h)
    mask = np.random.random(int(time / h)) < firing_rate * dt
    spike_times = times[mask]
    return spike_times


def convert_data_to_patterns_poisson(x, y, pattern_time, firing_rate, h, mult):
    output = []

    for xx, yy in zip(x, y):
        pattern = {}

        tmp_dict = dict.fromkeys(np.arange(0, len(xx) * mult))

        i = 0

        for x in xx:
            time = x * pattern_time

            for j in xrange(mult):
                tmp_dict[i] = get_poisson_train(time, firing_rate, h)
                i += 1

        pattern['input'] = tmp_dict
        pattern['class'] = yy

        output.append(pattern)
    return output


def plot_weights(weights, show=True):
    pl.title('Weight distribution')
    for neuron in weights:
        pl.plot(range(len(weights[neuron])),
                weights[neuron], '.', label=str(neuron))
    pl.legend()
    if show:
        pl.show()
    else:
        return pl.plot()


def plot_animated_weights(weights_history, save, show):
    all_plot = []
    fig = pl.figure()

    for weights in weights_history:
        all_plot.append(plot_weights(weights, False))

    weights_anim = animation.ArtistAnimation(fig, all_plot, interval=40,
                                             repeat_delay=len(weights_history),
                                             blit=True)
    if save is True:
        weights_anim.save(path['result'] + 'weights.mp4')
    if show is True:
        pl.show()

    
def plot_devices(devices):
    nest.voltage_trace.from_device(devices['voltmeter'])
    pl.show()

    nest.raster_plot.from_device(devices['spike_detector_2'], hist=False)
    pl.show()

    nest.raster_plot.from_device(devices['spike_detector_1'], hist=False)
    pl.show()

    
def plot_devices_start(devices, settings):
    nest.voltage_trace.from_device(devices['voltmeter'])
    pl.xlim(settings['start_delta'], settings['start_delta'] + settings['h_time'])
    pl.show()

    nest.raster_plot.from_device(devices['spike_detector_2'], hist=False)
    pl.xlim(settings['start_delta'], settings['start_delta'] + settings['h_time'])
    pl.show()

    nest.raster_plot.from_device(devices['spike_detector_1'], hist=False)
    pl.xlim(settings['start_delta'], settings['start_delta'] + settings['h_time'])
    pl.show()

    
def plot_devices_end(devices, settings):
    nest.voltage_trace.from_device(devices['voltmeter'])
    pl.xlim(settings['full_time'] - settings['h_time'], settings['full_time'])
    pl.show()

    nest.raster_plot.from_device(devices['spike_detector_2'], hist=False)
    pl.xlim(settings['full_time'] - settings['h_time'], settings['full_time'])
    pl.show()

    nest.raster_plot.from_device(devices['spike_detector_1'], hist=False)
    pl.xlim(settings['full_time'] - settings['h_time'], settings['full_time'])
    pl.show()
    

def plot_latencies(latencies):
    pl.title('Output latencies')
    for latency in latencies:
        if list(latency['latency']):
            if latency['class'] == 0:
                pl.plot(latency['latency'][:1], 1, 'rx')
            elif latency['class'] == 1:
                pl.plot(latency['latency'][:1], 2, 'gx')
            elif latency['class'] == 2:
                pl.plot(latency['latency'][:1], 3, 'bx')
    pl.show()


def plot_train_latency(latency_train):
    latency_paint = {'latency': [],
                     'epoch': []}

    epoch = 1

    for latency in latency_train:
        if latency['latency']:
            for lat in latency['latency']:
                latency_paint['latency'].append(lat)
                latency_paint['epoch'].append(epoch)
        else:
            latency_paint['latency'].append('nan')
            latency_paint['epoch'].append(epoch)
        epoch += 1

    pl.plot(latency_paint['epoch'], latency_paint['latency'], 'b.')
    pl.xlabel('Epochs')
    pl.ylabel('Latency')
    pl.show()


def plot_pattern(pattern):
    pl.ylim(0, 30)
    pl.xlim(0, len(pattern))
    pl.title('Temporal pattern')
    for neuron in pattern.keys():
        if pattern[neuron]:
            pl.plot(neuron, pattern[neuron], 'b.')
    pl.show()

    
def count_acc(full_latency, data_test):
    acc = 0
    output_list = []

    for i in xrange(len(data_test['input'])):
        tmp_list = [full_latency[tmp_latency][i]['latency'][:1] for tmp_latency in full_latency]

        min_index, min_value = min(enumerate(tmp_list),
                                   key=operator.itemgetter(1))
        if min_index == data_test['class'][i]:
            acc += 1
        output_list.append([tmp_list, data_test['class'][i], min_index])

    acc = float(acc) / len(data_test['input'])
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
        
        for i in range(n_neurons):
            tmp_dict = {'latency': ['inf'],
                        'class': latencies['class']}
            tmp_dicts.append(tmp_dict)
    
        for lat, sender in zip(latencies['latency'], latencies['senders']):
        
            for num, neuron_id in enumerate(neuron_out_ids):
                if sender == [neuron_id]: 
                    tmp_dicts[num]['latency'] = [lat]
        
        for latency_key, tmp_dict in zip(full_latency, tmp_dicts):
            full_latency[latency_key].append(tmp_dict)

    return full_latency


def get_latency(t, h_time, h, max_time):
    return int(t / h) % int(h_time / h) * h / max_time


def interconnect_layer(layer, syn_dict):
    for neuron_1 in layer:
        for neuron_2 in layer:
            if neuron_1 != neuron_2:
                nest.Connect([neuron_1], [neuron_2], syn_spec=syn_dict)


def prepare_data_iris(data, settings):
    data_train_0 = {}
    data_test_0 = {}

    data_train_1 = {}
    data_test_1 = {}

    data_train_2 = {}
    data_test_2 = {}

    data_train = {}
    data_test = {}
    
    data_out = {'train': {},
                'test': {}}

    mask_0 = data['class'] == 0
    mask_1 = data['class'] == 1
    mask_2 = data['class'] == 2

    # print data['class'][mask_0]
    # print data['class'][mask_1]
    # print data['class'][mask_2]

    input_train_0, input_test_0, y_train_0, y_test_0 = train_test_split(data['input'][mask_0],
                                                                        data['class'][mask_0],
                                                                        test_size=settings['test_size'],
                                                                        random_state=settings['random_state'])

    data_train_0['input'] = input_train_0
    data_train_0['class'] = y_train_0

    data_test_0['input'] = input_test_0
    data_test_0['class'] = y_test_0

    input_train_1, input_test_1, y_train_1, y_test_1 = train_test_split(data['input'][mask_1],
                                                                        data['class'][mask_1],
                                                                        test_size=settings['test_size'],
                                                                        random_state=settings['random_state'])

    data_train_1['input'] = input_train_1
    data_train_1['class'] = y_train_1

    data_test_1['input'] = input_test_1
    data_test_1['class'] = y_test_1

    input_train_2, input_test_2, y_train_2, y_test_2 = train_test_split(data['input'][mask_2],
                                                                        data['class'][mask_2],
                                                                        test_size=settings['test_size'],
                                                                        random_state=settings['random_state'])

    data_train_2['input'] = input_train_2
    data_train_2['class'] = y_train_2

    data_test_2['input'] = input_test_2
    data_test_2['class'] = y_test_2

    data_test['input'] = np.concatenate((input_test_0, input_test_1, input_test_2))
    data_test['class'] = np.concatenate((y_test_0, y_test_1, y_test_2))
    
    data_train['input'] = np.concatenate((input_train_0, input_train_1, input_train_2))
    data_train['class'] = np.concatenate((y_train_0, y_train_1, y_train_2))
    
    data_out['train']['class_0'] = data_train_0
    data_out['train']['class_1'] = data_train_1
    data_out['train']['class_2'] = data_train_2
    
    data_out['test']['class_0'] = data_test_0
    data_out['test']['class_1'] = data_test_1
    data_out['test']['class_2'] = data_test_2
    
    data_out['test']['full'] = data_test
    data_out['train']['full'] = data_train
    
    return data_out


def prepare_data_cancer(data, settings):
    data_train_0 = {}
    data_test_0 = {}

    data_train_1 = {}
    data_test_1 = {}

    data_train = {}
    data_test = {}
    
    data_out = {'train': {},
                'test': {}}

    input_train, input_test, y_train, y_test = train_test_split(data['input'], data['class'],
                                                                shuffle=False,
                                                                test_size=settings['test_size'],
                                                                random_state=settings['random_state'])
    data_train_0['input'] = input_train[y_train == 0]
    data_train_0['class'] = y_train[y_train == 0]

    data_test_0['input'] = input_test[y_test == 0]
    data_test_0['class'] = y_test[y_test == 0]

    data_train_1['input'] = input_train[y_train == 1]
    data_train_1['class'] = y_train[y_train == 1]

    data_test_1['input'] = input_test[y_test == 1]
    data_test_1['class'] = y_test[y_test == 1]

    data_test['input'] = input_test
    data_test['class'] = y_test
    
    data_train['input'] = input_train
    data_train['class'] = y_train
    
    data_out['train']['class_0'] = data_train_0
    data_out['train']['class_1'] = data_train_1
    
    data_out['test']['class_0'] = data_test_0
    data_out['test']['class_1'] = data_test_1
    
    data_out['test']['full'] = data_test
    data_out['train']['full'] = data_train
    
    return data_out


def train(settings, data):
    nest.ResetKernel()
    np.random.seed()
    rng = np.random.randint(500)
    nest.SetKernelStatus({'local_num_threads': settings['num_threads'],
                          'resolution': settings['h'],
                          'rng_seeds': range(rng, rng + settings['num_threads'])})

    spike_generators_1 = nest.Create('spike_generator', settings['n_input'])

    poisson_layer = nest.Create('poisson_generator', settings['n_input'])

    spike_generators_2 = nest.Create('spike_generator', settings['n_input'])

    parrot_layer = nest.Create('parrot_neuron', settings['n_input'])

    layer_1 = nest.Create('iaf_psc_exp', settings['n_layer_1'])

    teacher_1 = nest.Create("step_current_generator", settings['n_layer_1'])
    # teacher_2 = nest.Create("step_current_generator")
    # teacher_3 = nest.Create("step_current_generator")

    spike_detector_1 = nest.Create('spike_detector')
    spike_detector_2 = nest.Create('spike_detector')

    voltmeter = nest.Create('voltmeter', 1,
                            {'withgid': True,
                             'withtime': True})

    nest.SetStatus(poisson_layer, {'rate': settings['noise_freq']})

    nest.Connect(spike_generators_1, parrot_layer,
                 'one_to_one', syn_spec='static_synapse')
    nest.Connect(poisson_layer, parrot_layer,
                 'one_to_one', syn_spec='static_synapse')
    
    if settings['use_teacher']:
        # for teacher, neuron in zip(teacher_1, layer_1):
        #     print teacher, neuron
        #     nest.Connect([teacher], [neuron],
        #                  'one_to_one', syn_spec='static_synapse')
            
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

        if settings['use_inhibition']:
            interconnect_layer(layer_hid, settings['syn_dict_inh'])

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
    
    output_latency = []
    weights_history = []

    nest.Simulate(settings['start_delta'])
    d_time = settings['start_delta']
    
    while d_time < settings['full_time']:
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
            
        tmp_weights = {'layer_0': {}}
        
        for neuron_id in layer_1:
            tmp_weight = []
            for input_id in parrot_layer:
                conn = nest.GetConnections([input_id], [neuron_id], 
                                           synapse_model=settings['syn_dict_stdp']['model'])
                weight_one = nest.GetStatus(conn, 'weight')
                tmp_weight.append(weight_one[0])
            tmp_weights['layer_0'][neuron_id] = tmp_weight
        weights_history.append(tmp_weights)

    weights = {'layer_0': {}}

    if settings['two_layers']:
        weights['layer_0'] = nest.GetStatus(nest.GetConnections(layer_1,
                                            synapse_model=settings['syn_dict_stdp']['model']),
                                            'weight')
        weights['layer_1'] = nest.GetStatus(nest.GetConnections(layer_hid,
                                            synapse_model=settings['syn_dict_stdp']['model']),
                                            'weight')
    else:
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

    return weights, output_latency, devices, weights_history


def test(settings, data, weights):
    nest.ResetKernel()
    np.random.seed()
    rng = np.random.randint(500)
    nest.SetKernelStatus({'local_num_threads': settings['num_threads'],
                          'resolution': settings['h'],
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

    nest.SetStatus(poisson_layer, {'rate': settings['noise_freq']})

    nest.Connect(spike_generators_1, parrot_layer,
                 'one_to_one', syn_spec='static_synapse')
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

        interconnect_layer(layer_hid, settings['syn_dict_inh'])

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


def test_3_neuron_acc(data, settings):
    print settings['random_state']

    data_out = prepare_data(data, settings)

    data_train_0 = data_out['train']['class_0']
    data_train_1 = data_out['train']['class_1']
    data_train_2 = data_out['train']['class_2']

    # data_test_0 = data_out['test']['class_0']
    # data_test_1 = data_out['test']['class_1']
    # data_test_2 = data_out['test']['class_2']

    data_test = data_out['test']['full']

    print "Class 0"

    weights_0, latency_train_0, devices, weights_history = train(settings, data_train_0)
    # plot_weights(weights_0['layer_0'])
    latency_0, devices_test = test(settings, data_test, weights_0)
    # plot_latencies(latency_0)
    # window_size = 5 * 1 * len(data_test_2['input'])
    # plot_devices(devices_test)

    print "Class 1"

    weights_1, latency_train_1, devices, weights_history = train(settings, data_train_1)
    # plot_weights(weights_1['layer_0'])
    latency_1, devices_test = test(settings, data_test, weights_1)
    # plot_latencies(latency_1)
    # window_size = 5 * 1 * len(data_test['input'])
    # plot_devices(devices_test)

    print "Class 2"

    weights_2, latency_train_2, devices, weights_history = train(settings, data_train_2)
    # plot_weights(weights_2['layer_0'])
    latency_2, devices_test = test(settings, data_test, weights_2)
    # plot_latencies(latency_2)
    # print latency_2
    # window_size = 5 * 1 * len(data_test['input'])
    # plot_devices(devices_test)

    print "Test latencies"

    full_latency = {
                    'neuron_0': latency_0,
                    'neuron_1': latency_1,
                    'neuron_2': latency_2,
                   }

    acc, output_list = count_acc(full_latency, data_test)
    # for res in output_list:
    #     print res
    return acc, output_list


def test_3_neuron_acc_cv(data, settings):
    acc = []
    for rnd_state in settings['random_states']:
        settings['random_state'] = rnd_state
        accuracy, ouput_list = test_3_neuron_acc(data, settings)
        acc.append(accuracy)
    return np.mean(acc), np.std(acc)


def test_network_acc(data, settings):
    data_out = {}
    if settings['dataset'] == 'cancer':
        data_out = prepare_data_cancer(data, settings)
    elif settings['dataset'] == 'iris':
        data_out = prepare_data_iris(data, settings)

    data_train = data_out['train']['full']
    data_test = data_out['test']['full']

    weights, latency_train, devices_train, weights_history = train(settings, data_train)

    latency_test, devices_test = test(settings, data_test, weights)

    full_latency = create_full_latency(latency_test, settings['neuron_out_ids'])

    acc, output_list = count_acc(full_latency, data_test)
    return acc, output_list


def test_network_acc_cv(data, settings):
    acc = []
    for rnd_state in settings['random_states']:
        settings['random_state'] = rnd_state
        accuracy, ouput_list = test_network_acc(data, settings)
        acc.append(accuracy)
    return np.mean(acc), np.std(acc)


def test_parameter(data, parameters, settings, n_times):
    acc = []
    result = {'accuracy': [],
              'std': [],
              'parameter': [],
              'parameter_name': [],
              }
    settings_copy = settings
    for i in xrange(n_times):
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
    acc = []
    result = {'accuracy': [],
              'std': [],
              'parameter': [],
              'parameter_name': [],
              }
    settings_copy = settings
    for i in xrange(n_times):
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

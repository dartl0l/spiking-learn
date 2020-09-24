# coding: utf-8

import json
import time
import sys
import pickle

from math import exp
from mpi4py import MPI

# import numpy as np

from sklearn import preprocessing

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.decomposition import PCA

from .network import *
from .converter import *
from .plotter import *


class Solver(object):
    """base class for different Solvers"""
    def __init__(self, settings):
        # super(Solver, self).__init__()
        self.settings = settings

    def test_acc(self, data):
        pass

    def shuffle_data(self, x, y):
        assert len(x) == len(y)
        p = np.random.permutation(len(x))
        return x[p], y[p]

    def convert_latency(self, latency_list):
        output_array = []
        n_neurons = self.settings['topology']['n_layer_out']
        for latencies in latency_list:
            tmp_list = [np.nan] * n_neurons
            senders = set(latencies['senders'])
            for sender in senders:
                mask = latencies['senders'] == sender
                tmp_list[sender - 1] = latencies['spikes'][mask][0]
            output_array.append(tmp_list)
        return output_array

    def merge_spikes_and_senders(self, raw_latency_list):
        raw_latency = {
                       'spikes': [],
                       'senders': []
                      }
        for tmp_latency in raw_latency_list:
            raw_latency['spikes'].extend(tmp_latency['spikes'])
            raw_latency['senders'].extend(tmp_latency['senders'])
        return raw_latency

    def split_spikes_and_senders(self, input_latency, n_examples):
        output_latency = []
        d_time = self.settings['network']['start_delta']
        input_latency['spikes'] = np.array(input_latency['spikes'])
        input_latency['senders'] = np.array(input_latency['senders'])
        for _ in range(n_examples):
            mask = (input_latency['spikes'] > d_time) & \
                   (input_latency['spikes'] < d_time + self.settings['network']['h_time'])
            spikes_tmp = input_latency['spikes'][mask]
            senders_tmp = input_latency['senders'][mask]
            tmp_dict = {
                        'spikes': spikes_tmp - d_time,
                        'senders': senders_tmp
                        }

            d_time += self.settings['network']['h_time']
            output_latency.append(tmp_dict)    
        return output_latency

    def predict_from_latency(self, latency_list):
        latency_list = np.array(latency_list)
        mask = np.logical_not(np.all(np.isnan(latency_list), axis=1))
        prediction = np.zeros(len(latency_list))
        prediction[mask] = np.nanargmin(latency_list[mask], axis=1)
        return prediction

    def fitness_func_time(self, latency_list, data):
        fit_list = []

        for latency, y in zip(latency_list, data['class']):
            latency_of_desired_neuron = latency.pop(y)

            fit = -1 * latency_of_desired_neuron
            fit_list.append(fit)

        fitness_score = np.mean(fit_list)
        if np.isnan(fitness_score):
            fitness_score = 0
        return fitness_score

    def fitness_func_sigma(self, latency_list, data):
        def sigmoid(x, alpha):
            return 1 / (1 + np.exp(-2 * alpha * x))

        fit_list = []
        for latency, y in zip(latency_list, data['class']):
            latency_of_desired_neuron = latency.pop(y)
            fit = 1
            for lat in latency:
                fit *= sigmoid(lat - latency_of_desired_neuron, 0.1)
            fit_list.append(fit)
        fitness_score = np.mean(fit_list)
        if np.isnan(fitness_score):
            fitness_score = 0
        return fitness_score, fit_list

    def fitness_func_exp(self, latency_list, data):
        fit_list = []
        for latency, y in zip(latency_list, data['class']):
            latency_of_desired_neuron = latency.pop(y)
            fit = 1
            for lat in latency:
                fit -= exp(latency_of_desired_neuron - lat)
            fit_list.append(fit)
        fitness_score = np.mean(fit_list)
        if np.isnan(fitness_score):
            fitness_score = 0
        return fitness_score, fit_list

    def fitness(self, full_latency, data):
        settings = self.settings

        fitness_score = 0
        if settings['learning']['fitness_func'] == 'exp':
            fitness_score, fit_list = self.fitness_func_exp(full_latency,
                                                            data)
        elif settings['learning']['fitness_func'] == 'sigma':
            fitness_score, fit_list = self.fitness_func_sigma(full_latency,
                                                              data)
        elif settings['learning']['fitness_func'] == 'time':
            fitness_score = self.fitness_func_time(full_latency, data)
        elif settings['learning']['fitness_func'] == 'acc':
            y_valid = self.predict_from_latency(full_latency)
            fitness_score = accuracy_score(data['class'], y_valid)
        elif settings['learning']['fitness_func'] == 'f1':
            y_valid = self.predict_from_latency(full_latency)
            fitness_score = f1_score(data['class'], y_valid, average='micro')
        return fitness_score

    def prediction_score(self, y, prediction):
        settings = self.settings
        score = 0
        if settings['learning']['metrics'] == 'acc':
            score = accuracy_score(y, prediction)
        elif settings['learning']['metrics'] == 'f1':
            score = f1_score(y, prediction, average='micro')
        return score

    def split_data(self, data):
        data_list = []
        skf = StratifiedKFold(n_splits=self.settings['learning']['n_splits'])
        for train_index, test_index in skf.split(data['input'], data['class']):
            print("prepare data")
            data_fold = self.prepare_data(data,
                                          train_index,
                                          test_index)
            data_list.append(data_fold)
        return data_list

    def prepare_data(self, data, train_index, test_index):
        settings = self.settings

        data_test = {}
        data_train = {}
        data_out = {'train': {},
                    'test': {}}
        if settings['data']['use_valid']:
            data_valid = {}
            data_out['valid'] = {}

            input_train, input_valid, \
                y_train, y_valid = train_test_split(
                    data['input'][train_index],
                    data['class'][train_index],
                    test_size=settings['data']['valid_size'],
                    random_state=42
                )
            data_train['input'] = input_train
            data_train['class'] = y_train

            data_valid['input'] = input_valid
            data_valid['class'] = y_valid

            data_out['valid']['full'] = data_valid
        else:
            data_train['input'] = data['input'][train_index]
            data_train['class'] = data['class'][train_index]

        data_test['input'] = data['input'][test_index]
        data_test['class'] = data['class'][test_index]

        if settings['data']['shuffle_train']:
            data_train['input'], \
                data_train['class'] = self.shuffle_data(
                    data_train['input'],
                    data_train['class']
                )

        if settings['data']['shuffle_test']:
            data_test['input'], \
                data_test['class'] = self.shuffle_data(
                    data_test['input'],
                    data_test['class']
                )

        data_out['train']['full'] = data_train
        data_out['test']['full'] = data_test
        
        return data_out        

    def test_acc_cv(self, data):
        data_list = self.split_data(data)

        fold_map = map(self.test_acc, data_list)

        out_dict = self.create_result_dict(fold_map)
        return out_dict

    def create_result_dict(self, fold_map):
        fit = []
        y_test_list = []
        y_train_list = []
        weights = []
        acc_test = []
        acc_train = []
        latency_test_list = []
        latency_train_list = []
        devices_test_list = []
        devices_train_list = []
        test_classes_list = []
        train_classes_list = []

        for result, weight in fold_map:
            acc_test.append(result['acc_test'])
            acc_train.append(result['acc_train'])
            fit.append(result['fitness_score'])
            latency_test_list.append(result['latency_test'])
            latency_train_list.append(result['latency_train'])
            test_classes_list.append(result['test_classes'])
            train_classes_list.append(result['train_classes'])
            devices_test_list.append(result['devices_test'])
            devices_train_list.append(result['devices_train'])
            y_test_list.append(result['y_test'])
            y_train_list.append(result['y_train'])
            weights.append(weight)

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
            'y_test': y_test_list,
            'y_train': y_train_list,
            'test_classes': test_classes_list,
            'train_classes': train_classes_list,
            'devices_test': devices_test_list,
            'devices_train': devices_train_list,
            'weights': weights,
        }
        return out_dict


class NetworkSolver(Solver):
    """solver for network"""
    def __init__(self, settings, plot=False):
        super().__init__(settings)
        self.plot = plot

        if settings['topology']['two_layers']:
            if settings['topology']['use_convolution']:
                self.network = ConvolutionNetwork(settings)
            else:
                self.network = TwoLayerNetwork(settings)
        elif settings['data']['frequency_coding']:
            self.network = FrequencyNetwork(settings)
        else:
            self.network = Network(settings)

    def test_data(self, data, weights):
        raw_latency, devices = self.network.test(data['input'], weights)
        all_latency = self.split_spikes_and_senders(raw_latency, len(data['class']))
        out_latency = self.convert_latency(all_latency)
        return out_latency, devices

    def test_acc(self, data):
        plot = Plotter()

        data_train = data['train']['full']
        data_for_train = data['train']['full']
        if self.settings['learning']['reverse_learning']:
            data_for_train['input'] = np.nanmax(data_train['input']) - data_train['input']
            assert data_for_train['input'].shape == data_train['input'].shape

        weights, latency_train, \
            devices_train = self.network.train(data_for_train['input'], data_for_train['class'])

        if self.plot:
            plot.plot_devices(devices_train,
                              self.settings['topology']['two_layers'])

        full_latency_test_train, \
            devices_test_train = self.test_data(data_train,
                                                weights)

        if self.plot:
            plot.plot_devices(devices_test_train,
                              self.settings['topology']['two_layers'])

        y_train = self.predict_from_latency(full_latency_test_train)
        score_train = self.prediction_score(data_train['class'],
                                            y_train)

        fitness_score = 0
        if self.settings['data']['use_valid']:
            data_valid = data['valid']['full']
            full_latency_valid, \
                devices_valid = self.test_data(data_valid,
                                               weights)

            if self.settings['learning']['use_fitness_func']:
                fitness_score = self.fitness(full_latency_valid,
                                             data_valid)
        else:
            fitness_score = score_train

        data_test = data['test']['full']
        full_latency_test, \
            devices_test = self.test_data(data_test,
                                          weights)

        if self.plot:
            plot.plot_devices(devices_test,
                              self.settings['topology']['two_layers'])

        y_test = self.predict_from_latency(full_latency_test)
        score_test = self.prediction_score(data_test['class'],
                                           y_test)

        out_dict = {
                    'fitness_score': fitness_score,
                    'acc_test': score_test,
                    'acc_train': score_train,
                    'y_test': y_test,
                    'y_train': y_train,
                    'latency_test': full_latency_test,
                    'latency_train': full_latency_test_train,
                    'train_classes': data_train['class'],
                    'test_classes': data_test['class'],
                    'devices_test': devices_test,
                    'devices_train': devices_train,
                   }

        return out_dict, weights


class MPINetworkSolver(NetworkSolver):
    """solver for network"""
    def __init__(self, settings, plot=False):
        super().__init__(settings, plot)
        self.comm = MPI.COMM_WORLD

    def test_data(self, data, weights):
        raw_latency, devices = self.network.test(data['input'], weights)
        self.comm.Barrier()
        all_latency = self.comm.allgather(raw_latency)
        raw_latency = self.merge_spikes_and_senders(all_latency)
        all_latency = self.split_spikes_and_senders(raw_latency,
                                                    len(data['class']))
        out_latency = self.convert_latency(all_latency)
        return out_latency, devices

    def test_acc(self, data):
        plot = Plotter()

        data_train = data['train']['full']
        data_for_train = data_train
        if self.settings['learning']['reverse_learning']:
            data_for_train['input'] = np.amax(data_train['input']) - data_train['input']
            assert data_for_train['input'].shape == data_train['input'].shape

        weights, latency_train, \
            devices_train = self.network.train(data_for_train['input'], data_for_train['class'])

        if self.plot:
            plot.plot_devices(devices_train,
                              self.settings['topology']['two_layers'])

        full_latency_test_train, \
            devices_test_train = self.test_data(data_train,
                                                weights)

        if self.plot:
            plot.plot_devices(devices_test_train,
                              self.settings['topology']['two_layers'])

        y_train = self.predict_from_latency(full_latency_test_train)
        score_train = self.prediction_score(data_train['class'],
                                            y_train)

        fitness_score = 0
        if self.settings['data']['use_valid']:
            data_valid = data['valid']['full']
            full_latency_valid, \
                devices_valid = self.test_data(data_valid,
                                               weights)

            if self.settings['learning']['use_fitness_func']:
                fitness_score = self.fitness(full_latency_valid,
                                             data_valid)
        else:
            fitness_score = score_train

        data_test = data['test']['full']
        full_latency_test, \
            devices_test = self.test_data(data_test,
                                          weights)

        if self.plot:
            plot.plot_devices(devices_test,
                              self.settings['topology']['two_layers'])

        y_test = self.predict_from_latency(full_latency_test)
        score_test = self.prediction_score(data_test['class'],
                                           y_test)

        self.comm.Barrier()
        weights_all = self.comm.allgather(weights)

        out_dict = {
                    'fitness_score': fitness_score,
                    'acc_test': score_test,
                    'acc_train': score_train,
                    'y_test': y_test,
                    'y_train': y_train,
                    'latency_test': full_latency_test,
                    'latency_train': full_latency_test_train,
                    'train_classes': data_train['class'],
                    'test_classes': data_test['class'],
                    'devices_test': devices_test,
                    'devices_train': devices_train,
                   }

        return out_dict, weights_all


class FrequencyNetworkSolver(NetworkSolver):
    def __init__(self, settings, plot=False):
        self.plot = plot
        
        self.settings = settings
        self.network = FrequencyNetwork(settings)

    def convert_latency(self, latency_list):
        output_array = []
        n_neurons = self.settings['topology']['n_layer_out']
        for latencies in latency_list:
            tmp_list = [np.nan] * n_neurons
            senders = set(latencies['senders'])
            for sender in senders:
                mask = latencies['senders'] == sender
                tmp_list[sender - 1] = len(latencies['spikes'][mask]) \
                    / self.settings['data']['pattern_length']
            output_array.append(tmp_list)
        return output_array

    def predict_from_latency(self, latency_list):
        return np.nanargmax(latency_list, axis=1)


class SeparateNetworkSolver(NetworkSolver):
    """solver for separate network"""

    def __init__(self, settings, plot=False):
        super().__init__(settings, plot)

    def merge_spikes(self, separate_latency_list):
        out_latency = []
        num_neurons = len(separate_latency_list)
        data_len = len(separate_latency_list[0])
        for i in range(data_len):
            tmp_latency = {}
            for j in range(num_neurons):
                tmp_latency['neuron_' + str(j)] = separate_latency_list[j][i][0]
            out_latency.append(tmp_latency)
        return out_latency

    def test_acc(self, data):
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
                devices_train = self.network.train(data_train['input'], data_train['class'])

            full_latency_test_train, \
                devices_test_train = self.test_data(data['train']['full'],
                                                    weights)
            latency_test_train_list.append(full_latency_test_train)

            full_latency_test, \
                devices_test = self.test_data(data_test,
                                              weights)
            latency_test_list.append(full_latency_test)

            if self.settings['data']['use_valid']:
                data_valid = data['valid']['full']
                full_latency_valid, \
                    devices_valid = self.test_data(data_valid,
                                                   weights)
                latency_valid_list.append(full_latency_valid)

        data_train = data['train']['full']
        merged_latency_test_train = self.merge_spikes(latency_test_train_list)
        y_train = self.predict_from_latency(merged_latency_test_train)
        score_train = self.prediction_score(data_train['class'],
                                            y_train)

        fitness_score = 0
        if self.settings['data']['use_valid']:
            data_valid = data['valid']['full']
            merged_latency_valid = self.merge_spikes(latency_valid_list)

            if self.settings['learning']['use_fitness_func']:
                fitness_score = self.fitness(merged_latency_valid, data_valid)
        else:
            fitness_score = score_train

        merged_latency_test = self.merge_spikes(latency_test_list)
        y_test = self.predict_from_latency(merged_latency_test)
        score_test = self.prediction_score(data_test['class'],
                                           y_test)

        out_dict = {
            'fitness_score': fitness_score,
            'acc_test': score_test,
            'acc_train': score_train,
            'y_test': y_test,
            'y_train': y_train,
            'latency_test': merged_latency_test,
            'latency_train': merged_latency_test_train,
            'train_classes': data_train['class'],
            'test_classes': data_test['class'],
            'devices_test': devices_test,
            'devices_train': devices_train,
        }

        return out_dict, weights


class MPISeparateNetworkSolver(MPINetworkSolver):
    """solver for separate network"""
    def __init__(self, settings, plot=False):
        super().__init__(settings, plot)


    def merge_spikes(self, separate_latency_list):
        out_latency = []
        num_neurons = len(separate_latency_list)
        data_len = len(separate_latency_list[0])
        for i in range(data_len):
            tmp_latency = {}
            for j in range(num_neurons):
                tmp_latency['neuron_' + str(j)] = separate_latency_list[j][i][0]
            out_latency.append(tmp_latency)
        return out_latency

    def test_acc(self, data):
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
                devices_train = self.network.train(data_train['input'], data_train['class'])

            full_latency_test_train, \
                devices_test_train = self.test_data(data['train']['full'],
                                                    weights)
            latency_test_train_list.append(full_latency_test_train)

            full_latency_test, \
                devices_test = self.test_data(data_test,
                                              weights)
            latency_test_list.append(full_latency_test)

            if self.settings['data']['use_valid']:
                data_valid = data['valid']['full']
                full_latency_valid, \
                    devices_valid = self.test_data(data_valid,
                                                   weights)
                latency_valid_list.append(full_latency_valid)

        data_train = data['train']['full']
        merged_latency_test_train = self.merge_spikes(latency_test_train_list)
        y_train = self.predict_from_latency(merged_latency_test_train)
        score_train = self.prediction_score(data_train['class'], 
                                            y_train)

        fitness_score = 0
        if self.settings['data']['use_valid']:
            data_valid = data['valid']['full']
            merged_latency_valid = self.merge_spikes(latency_valid_list)
            
            if self.settings['learning']['use_fitness_func']:
                fitness_score = self.fitness(merged_latency_valid, data_valid)
        else:
            fitness_score = score_train

        merged_latency_test = self.merge_spikes(latency_test_list)
        y_test = self.predict_from_latency(merged_latency_test)
        score_test = self.prediction_score(data_test['class'], 
                                           y_test)

        self.comm.Barrier()

        weights_all = self.comm.allgather(weights)

        out_dict = {
                    'fitness_score': fitness_score,
                    'acc_test': score_test,
                    'acc_train': score_train,
                    'y_test': y_test,
                    'y_train': y_train,
                    'latency_test': merged_latency_test,
                    'latency_train': merged_latency_test_train,
                    'train_classes': data_train['class'],
                    'test_classes': data_test['class'],
                    'devices_test': devices_test,
                    'devices_train': devices_train,
                   }
        
        return out_dict, weights_all


class BaseLineSolver(Solver):
    """solver for separate network"""
    def __init__(self, settings):
        super().__init__(settings)
        from sklearn.ensemble import GradientBoostingClassifier

        self.settings = settings
        self.clf = GradientBoostingClassifier()

    def test_acc_cv(self, data):
        data_list = self.split_data(data)

        fold_map = map(self.test_acc, data_list)

        fit = []
        acc_test = []
        acc_train = []
        for result in fold_map:
            fit.append(result['fitness_score'])
            acc_test.append(result['acc_test'])
            acc_train.append(result['acc_train'])

        out_dict = {
                    'fitness_score': fit,
                    'fitness_mean': np.mean(fit),
                    'accs_test': acc_test,
                    'accs_test_mean': np.mean(acc_test),
                    'accs_test_std': np.std(acc_test),
                    'accs_train': acc_train,
                    'accs_train_mean': np.mean(acc_train),
                    'accs_train_std': np.std(acc_train),
                   }
        return out_dict

    def test_acc(self, data):
        data_train = data['train']['full']

        self.clf.fit(data_train['input'], data_train['class'])

        score_train = self.clf.score(data_train['input'], data_train['class'])

        fitness_score = 0
        if self.settings['data']['use_valid']:
            data_valid = data['valid']['full']

            if self.settings['learning']['use_fitness_func']:
                fitness_score = self.clf.score(data_valid['input'], data_valid['class'])
        else:
            fitness_score = score_train

        data_test = data['test']['full']
        score_test = self.clf.score(data_test['input'], data_test['class'])

        out_dict = {
            'fitness_score': fitness_score,
            'acc_test': score_test,
            'acc_train': score_train,
            'train_classes': data_train['class'],
            'test_classes': data_test['class']
        }

        return out_dict

    def create_result_dict(self, fold_map):
        fit = []
        acc_test = []
        acc_train = []
        for result in fold_map:
            fit.append(result['fitness_score'])
            acc_test.append(result['acc_test'])
            acc_train.append(result['acc_train'])

        out_dict = {
                    'fitness_score': fit,
                    'fitness_mean': np.mean(fit),
                    'accs_test': acc_test,
                    'accs_test_mean': np.mean(acc_test),
                    'accs_test_std': np.std(acc_test),
                    'accs_train': acc_train,
                    'accs_train_mean': np.mean(acc_train),
                    'accs_train_std': np.std(acc_train),
                   }
        return out_dict


def round_decimals(value):
    i = 0
    while value < 1:
        value *= 10
        i += 1
    return i


def create_folds_from_result_dict(result_dict, reshape=False):
    folds = []
    for fold_latency_train, y_train, fold_latency_test, y_test in zip(result_dict['latency_train'],
                                                                      result_dict['train_classes'],
                                                                      result_dict['latency_test'],
                                                                      result_dict['test_classes']):
        data_train = {}
        data_test = {}
        data_out = {'train': {},
                    'test': {}}
        x_train = np.array(fold_latency_train)
        x_test = np.array(fold_latency_test)

        if reshape:
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
        else:
            x_train = np.nan_to_num(x_train, nan=0)
            x_test = np.nan_to_num(x_test, nan=0)

        data_train['input'] = x_train
        data_train['class'] = y_train

        data_test['input'] = x_test
        data_test['class'] = y_test

        data_out['train']['full'] = data_train
        data_out['test']['full'] = data_test

        folds.append(data_out)
    return folds


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


def print_score(result_dict):
    print('Score train ' + "\t".join(map(lambda n: '%.2f' % n, result_dict['accs_train'])) + '\n')
    print('Score train mean: ' + str(result_dict['accs_train_mean']) + '\n'
          'Std train mean: ' + str(result_dict['accs_train_std']) + '\n')
    print('Score test ' + "\t".join(map(lambda n: '%.2f' % n, result_dict['accs_test'])) + '\n')
    print('Score test mean: ' + str(result_dict['accs_test_mean']) + '\n'
          'Std test mean: ' + str(result_dict['accs_test_std']) + '\n')


def solve_task(task_path='./', redirect_out=True, filename='settings.json', input_settings=None):
    if input_settings is None:
        settings = json.load(open(task_path + filename, 'r'))
    else:
        settings = input_settings

    if redirect_out:
        sys.stdout = open(task_path + 'out.txt', 'w')
        sys.stderr = open(task_path + 'err.txt', 'w')

    print("Start train and test")
    start = time.time()

    if settings['data']['dataset'] == 'iris':
        data = load_iris()
    elif settings['data']['dataset'] == 'cancer':
        data = load_breast_cancer()
    elif settings['data']['dataset'] == 'digits':
        data = {}
        digits = load_digits()
        data['data'] = digits.images.reshape((len(digits.images), -1))
        data['target'] = digits.target

    x = data['data']
    y = data['target']

    if 'pca' in settings['data']['preprocessing']:
        pca = PCA(n_components=4)
        x = pca.fit_transform(x)

    if 'normalize' in settings['data']['normalization']:
        x = preprocessing.normalize(x)
        print('normalize')
    if 'minmax' in settings['data']['normalization']:
        x = preprocessing.minmax_scale(x)
        print('minmax')


    # round_to = 2
    round_to = round_decimals(settings['network']['h'])

    print('convert')
    reverse = 'reverse' in settings['data']['conversion']
    no_last = 'no_last' in settings['data']['conversion']

    if 'receptive_fields' in settings['data']['conversion']:
        n_coding_neurons = settings['data']['n_coding_neurons']
        sigma = settings['data']['coding_sigma']
        settings['topology']['n_input'] = len(x[0]) * n_coding_neurons

        converter = ReceptiveFieldsConverter(sigma, 1.0, n_coding_neurons, round_to, reverse=reverse, no_last=no_last)
        data = converter.convert(x, y)
    elif 'temporal' in settings['data']['conversion']:
        converter = TemporalConverter(settings['data']['pattern_length'], round_to, reverse=reverse, no_last=no_last)
        data = converter.convert(x, y)
        settings['topology']['n_input'] = len(x[0])


    print('solve')
    if settings['network']['separate_networks']:
        if settings['network']['use_mpi']:
            solver = MPISeparateNetworkSolver(settings)
        else:
            solver = SeparateNetworkSolver(settings)
    else:
        if settings['network']['use_mpi']:
            solver = MPINetworkSolver(settings)
        else:
            solver = NetworkSolver(settings)

    result_dict = solver.test_acc_cv(data)

    end = time.time()
    print("End train and test in " + str(end - start) + "s")

    with open(task_path + 'acc.txt', 'w') as acc_file:
        acc_file.write('Accuracy train ' + "\t".join(map(lambda n: '%.2f'%n, 
                                                         result_dict['accs_train'])) + '\n')
        acc_file.write('Accuracy train mean: ' + str(result_dict['accs_train_mean']) + '\n'
                       'Std train mean: ' + str(result_dict['accs_train_std']) + '\n')

        acc_file.write('Accuracy test ' + "\t".join(map(lambda n: '%.2f'%n, 
                                                        result_dict['accs_test'])) + '\n')
        acc_file.write('Accuracy test mean: ' + str(result_dict['accs_test_mean']) + '\n'
                       'Std test mean: ' + str(result_dict['accs_test_std']) + '\n')

    with open(task_path + 'fitness.txt', 'w') as fit_file:
        fit_file.write(str(result_dict['fitness_mean']))

    pickle.dump(result_dict, open(task_path + 'result_dict.pkl', 'wb'))

    save = time.time()
    print("Files saved in "+ str(save - end) + "s")

    return result_dict['fitness_mean'], result_dict


def solve_baseline(task_path='./', redirect_out=True, filename='settings.json', input_settings=None):
    if input_settings is None:
        settings = json.load(open(task_path + filename, 'r'))
    else:
        settings = input_settings

    if redirect_out:
        sys.stdout = open(task_path + 'out.txt', 'w')
        sys.stderr = open(task_path + 'err.txt', 'w')

    print("Start train and test")
    start = time.time()

    if settings['data']['dataset'] == 'iris':
        data = load_iris()
    elif settings['data']['dataset'] == 'cancer':
        data = load_breast_cancer()
    elif settings['data']['dataset'] == 'digits':
        data = {}
        digits = load_digits()
        data['data'] = digits.images.reshape((len(digits.images), -1))
        data['target'] = digits.target

    x = data['data']
    y = data['target']

    if 'pca' in settings['data']['preprocessing']:
        pca = PCA(n_components=4)
        x = pca.fit_transform(x)

    if 'normalize' in settings['data']['normalization']:
        x = preprocessing.normalize(x)
        print('normalize')
    if 'minmax' in settings['data']['normalization']:
        x = preprocessing.minmax_scale(x)
        print('minmax')

    n_coding_neurons = settings['data']['n_coding_neurons']
    sigma = settings['data']['coding_sigma']

    # round_to = 2
    round_to = round_decimals(settings['network']['h'])

    print('convert')
    reverse = 'reverse' in settings['data']['conversion']
    no_last = 'no_last' in settings['data']['conversion']
    if 'receptive_fields' in settings['data']['conversion']:
        converter = ReceptiveFieldsConverter(sigma, 1.0, n_coding_neurons, round_to,
                                             reshape=False, reverse=reverse, no_last=no_last)
        data = converter.convert(x, y)
    elif 'temporal' in settings['data']['conversion']:
        converter = TemporalConverter(settings['data']['pattern_length'], round_to,
                                      reshape=False, reverse=reverse, no_last=no_last)
        data = converter.convert(x, y)

    solver = BaseLineSolver(settings)

    result_dict = solver.test_acc_cv(data)

    end = time.time()
    print("End train and test in " + str(end - start) + "s")

    with open(task_path + 'acc.txt', 'w') as acc_file:
        acc_file.write('Accuracy train ' + "\t".join(map(lambda n: '%.2f'%n,
                                                         result_dict['accs_train'])) + '\n')
        acc_file.write('Accuracy train mean: ' + str(result_dict['accs_train_mean']) + '\n'
                       'Std train mean: ' + str(result_dict['accs_train_std']) + '\n')

        acc_file.write('Accuracy test ' + "\t".join(map(lambda n: '%.2f'%n,
                                                        result_dict['accs_test'])) + '\n')
        acc_file.write('Accuracy test mean: ' + str(result_dict['accs_test_mean']) + '\n'
                       'Std test mean: ' + str(result_dict['accs_test_std']) + '\n')

    with open(task_path + 'fitness.txt', 'w') as fit_file:
        fit_file.write(str(result_dict['fitness_mean']))

    pickle.dump(result_dict, open(task_path + 'result_dict.pkl', 'wb'))

    save = time.time()
    print("Files saved in "+ str(save - end) + "s")

    return result_dict['fitness_mean'], result_dict


if __name__ == '__main__':
    if sys.argv[1]:
        solve_task(sys.argv[1])
    else:
        solve_task("../")


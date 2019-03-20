# coding: utf-8

import json
import time
import sys

from spiking_network_learning_alghorithm.patterns_mpi_new import *
from spiking_network_learning_alghorithm.converter import *

from sklearn import preprocessing
from sklearn.datasets import load_iris, load_breast_cancer


def solve_task(task_path):
    sys.stdout = open(task_path + 'out.txt', 'w')
    sys.stderr = open(task_path + 'err.txt', 'w')

    settings = json.load(open(task_path + 'settings.json', 'r'))

    print("Start train and test")
    start = time.time()

    if settings['data']['dataset'] == 'iris':
        data = load_iris()
    elif settings['data']['dataset'] == 'cancer':
        data = load_breast_cancer()

    X = data['data']
    y = data['target']

    # X = preprocessing.normalize(X)
    # if settings['data']['normalization'] == 'normalize':
    #     X = preprocessing.normalize(X)
    # elif settings['data']['normalization'] == 'minmax':
    #     X = preprocessing.minmax_scale(X)
    if 'normalize' in settings['data']['normalization']:
        X = preprocessing.normalize(X)
        print('normalize')
    if 'minmax' in settings['data']['normalization']:
        X = preprocessing.minmax_scale(X)
        print('minmax')


    n_coding_neurons = settings['data']['n_coding_neurons']
    sigma = settings['data']['coding_sigma']

    round_to = 2
    conv = Converter()
    data, max_y = conv.convert_data_to_patterns_gaussian_receptive_field(X, y, sigma, 1.0,
                                                                         n_coding_neurons, round_to)

    settings['topology']['n_input'] = len(X[0]) * n_coding_neurons
    result_dict = test_network_acc_cv(data, settings)

    print(result_dict)
    end = time.time()
    print("End train and test in "+ str(end - start) + "s")

    with open(task_path + 'acc.txt', 'w') as acc_file:
        acc_file.write('Accuracy train ' + "\t".join(map(lambda n: '%.2f'%n, result_dict['accs_train'])) + '\n')
        acc_file.write('Accuracy train mean: ' + str(result_dict['accs_train_mean']) + '\n'
                       'Std train mean: ' + str(result_dict['accs_train_std']) + '\n')

        acc_file.write('Accuracy test ' + "\t".join(map(lambda n: '%.2f'%n, result_dict['accs_test'])) + '\n')
        acc_file.write('Accuracy test mean: ' + str(result_dict['accs_test_mean']) + '\n'
                       'Std test mean: ' + str(result_dict['accs_test_std']) + '\n')

    with open(task_path + 'fitness.txt', 'w') as fit_file:
        fit_file.write(str(result_dict['fitness_mean']))

    save = time.time()
    print("Files saved in "+ str(save - end) + "s")

    return result_dict['fitness_mean']


if __name__ == '__main__':
    solve_task("./")
 
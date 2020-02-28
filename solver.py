# coding: utf-8

import json
import time
import sys

from spiking_network_learning_alghorithm.patterns import *
from spiking_network_learning_alghorithm.converter import *

from sklearn import preprocessing
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.decomposition import PCA

import pickle


def solve_task(task_path='./', redirect_out=True, filename='settings.json', input_settings=None):
    
    def round_decimals(value):
        i = 0
        while value < 1:
            value *= 10
            i += 1
        return i

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

    X = data['data']
    y = data['target']

    if 'pca' in settings['data']['preprocessing']:
        pca = PCA(n_components=4)
        X = pca.fit_transform(X)
    if 'normalize' in settings['data']['normalization']:
        X = preprocessing.normalize(X)
        print('normalize')
    if 'minmax' in settings['data']['normalization']:
        X = preprocessing.minmax_scale(X)
        print('minmax')


    n_coding_neurons = settings['data']['n_coding_neurons']
    sigma = settings['data']['coding_sigma']

    # round_to = 2
    round_to = round_decimals(settings['network']['h'])

    conv = Converter()
    data, max_y = conv.convert_data_to_patterns_gaussian_receptive_field(X, y, sigma, 1.0,
                                                                         n_coding_neurons, 
                                                                         round_to)

    settings['topology']['n_input'] = len(X[0]) * n_coding_neurons
    result_dict = test_network_acc_cv(data, settings)

    end = time.time()
    print("End train and test in "+ str(end - start) + "s")

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


    pickle.dump(result_dict, open('result_dict.pkl', 'wb'))

    save = time.time()
    print("Files saved in "+ str(save - end) + "s")


    return result_dict['fitness_mean'], result_dict


if __name__ == '__main__':
    if sys.argv[1]:
        solve_task(sys.argv[1])
    else:
        solve_task("./")
 
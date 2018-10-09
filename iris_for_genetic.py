# coding: utf-8

import json

import sys
# sys.path.append("./")
# path.append("/s/ls4/users/dartl0l/nest3/lib/python3.4/site-packages") 

from spiking_network_learning_alghorithm.patterns_new import *
from spiking_network_learning_alghorithm import converter # , plotter

from sklearn import preprocessing
from sklearn.datasets import load_iris

# from contextlib import redirect_stdout, redirect_stdout

# def solve_task_with_redirect(task_path):
#     # import contextlib.redirect_stdout
#     with open(task_path + 'out.txt', 'w') as f:
#         with redirect_stdout(f):
#             solve_task(task_path)


def solve_task(task_path):
    sys.stdout = open(task_path + 'out.txt', 'w')
    sys.stderr = open(task_path + 'err.txt', 'w')

    conv = converter.Converter()
    data = load_iris()

    X = data['data']
    y = data['target']

    X = preprocessing.normalize(X)

    settings = json.load(open(task_path + 'settings.json', 'r'))

    n_coding_neurons = settings['n_coding_neurons']
    sigma = settings['coding_sigma']

    round_to = 2
    data, max_y = conv.convert_data_to_patterns_gaussian_receptive_field(X, y, sigma, 1.0,
                                                                         n_coding_neurons, 1, round_to, True)

    settings['n_input'] = len(X[0]) * n_coding_neurons

    print("Start train and test")
    # acc, std, accs, fit, fitness = test_network_acc_cv_for_genetic(data, settings)
    result_dict = test_network_acc_cv_for_genetic(data, settings)

    print(result_dict)

    with open(task_path + 'acc.txt', 'w') as acc_file:
        acc_file.write('Accuracy train ' + "\t".join(map(lambda n: '%.2f'%n, result_dict['accs_train'])) + '\n')
        acc_file.write('Accuracy train mean: ' + str(result_dict['accs_train_mean']) + '\n'
                       'Std train mean: ' + str(result_dict['accs_train_std']) + '\n')

        acc_file.write('Accuracy test ' + "\t".join(map(lambda n: '%.2f'%n, result_dict['accs_test'])) + '\n')
        acc_file.write('Accuracy test mean: ' + str(result_dict['accs_test_mean']) + '\n'
                       'Std test mean: ' + str(result_dict['accs_test_std']) + '\n')

    with open(task_path + 'fitness.txt', 'w') as fit_file:
        fit_file.write(str(result_dict['fitness_mean']))
    return result_dict['fitness_mean']


if __name__ == '__main__':
    solve_task("./")
 
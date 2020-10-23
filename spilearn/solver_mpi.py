from .solver import *
from mpi4py import MPI


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
        solver = MPISeparateNetworkSolver(settings)
    else:
        solver = MPINetworkSolver(settings)

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
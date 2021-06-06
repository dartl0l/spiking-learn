import numpy as np

from sklearn.metrics import accuracy_score, f1_score


class Evaluation(object):
    def __init__(self, settings):
        # super(Solver, self).__init__()
        self.settings = settings
        
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


import numpy as np

from sklearn.metrics import accuracy_score, f1_score


class Evaluation:
    def __init__(self, settings):
        self.settings = settings
        self.n_neurons = self.settings['topology']['n_layer_out']
        self.start_delta = self.settings['network']['start_delta']
        self.h_time = self.settings['network']['h_time']
        
    def convert_latency(self, latency_list):
        output_array = []
        for latencies in latency_list:
            tmp_list = [np.nan] * self.n_neurons
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
        d_time = self.start_delta
        input_latency['spikes'] = np.array(input_latency['spikes'])
        input_latency['senders'] = np.array(input_latency['senders'])
        for _ in range(n_examples):
            mask = (input_latency['spikes'] > d_time) & \
                   (input_latency['spikes'] < d_time + self.h_time)
            spikes_tmp = input_latency['spikes'][mask]
            senders_tmp = input_latency['senders'][mask]
            tmp_dict = {
                        'spikes': spikes_tmp - d_time,
                        'senders': senders_tmp
                        }

            d_time += self.h_time
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


class EvaluationPool(Evaluation):
    def __init__(self, settings):
        super().__init__(settings)
        self.pool_size = self.settings['learning']['teacher_pool_size']
        self.n_neurons = self.settings['topology']['n_layer_out']

    def convert_latency(self, latency_list):
        output_array = []
        for latencies in latency_list:
            tmp_list = [np.nan] * self.n_neurons
            senders = set(latencies['senders'])
            for sender in senders:
                mask = latencies['senders'] == sender
                tmp_list[sender - 1] = latencies['spikes'][mask][0]
            output_array.append(tmp_list)
        output_array = np.array(output_array).reshape(
            len(output_array),
            int(self.n_neurons / self.pool_size),
            self.pool_size)
        output_array = np.mean(output_array, axis=2)
        return output_array


class DiehlEvaluation(Evaluation):
    
    def __init__(self, settings):
        super().__init__(settings)
        self.assignments = None
        self.h_time = self.settings['network']['h_time']

    def get_assignments(self, latencies, y):
        latencies = np.array(latencies)
        neurons_number = len(latencies[0])
        assignments = [-1] * neurons_number
        minimum_latencies_for_all_neurons = [self.h_time] * neurons_number
        for current_class in set(y):
            class_size = len(np.where(y == current_class)[0])
            if class_size == 0:
                # This class is not present in the set,
                # so no need to assign it to any neuron.
                continue
            latencies_for_this_class = np.mean(latencies[y == current_class], axis=0)
            for i in range(neurons_number):
                if latencies_for_this_class[i] < minimum_latencies_for_all_neurons[i]:
                    minimum_latencies_for_all_neurons[i] = latencies_for_this_class[i]
                    assignments[i] = current_class
        self.assignments = assignments
        return assignments

    def evaluate(self, latencies, y, latencies_for_assignments=None, y_for_assignments=None):
        if latencies_for_assignments is None:
            latencies_for_assignments = latencies
        if y_for_assignments is None:
            y_for_assignments = y
        latencies = np.array(latencies)
        y = np.array(y)
        latencies_for_assignments = np.array(latencies_for_assignments)
        y_for_assignments = np.array(y_for_assignments)

        # number_of_classes = len(set(y))
        # neurons_number = latencies.shape[1]
        assignments = self.get_assignments(latencies_for_assignments, y_for_assignments)
        class_certainty_ranks = [
            self.get_classes_rank_per_one_vector(
                latencies[i], set(y), assignments
            )
            for i in range(len(latencies))
        ]
        y_predicted = np.array(class_certainty_ranks)[:,0]
        # difference = y_predicted - y
        # correct = len(np.where(difference == 0)[0])
        # incorrect = np.where(difference != 0)[0]
        return y_predicted

    def get_classes_rank_per_one_vector(self, latency, set_of_classes, assignments):
        latency = np.array(latency)
        number_of_classes = len(set_of_classes)
        min_latencies = [0] * number_of_classes
        number_of_neurons_assigned_to_this_class = [0] * number_of_classes
        for class_number, current_class in enumerate(set_of_classes):
            number_of_neurons_assigned_to_this_class = len(np.where(assignments == current_class)[0])
            if number_of_neurons_assigned_to_this_class == 0:
                continue
            min_latencies[class_number] = np.min(
                latency[assignments == current_class]
            ) / number_of_neurons_assigned_to_this_class
        return np.argsort(min_latencies)[::1]
# coding: utf-8

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class FirstSpikeVotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, h_time) -> None:
        super().__init__()
        self.h_time = h_time
        self.classes = None
        self.assignments = None
        self.func = np.mean

    def _get_classes_rank_per_one_vector(self, latency, set_of_classes, assignments):
        latency = np.array(latency)
        number_of_classes = len(set_of_classes)
        min_latencies = [np.nan] * number_of_classes
        number_of_neurons_assigned_to_this_class = [0] * number_of_classes
        for class_number, current_class in enumerate(set_of_classes):
            number_of_neurons_assigned_to_this_class = len(
                np.where(assignments == current_class)[0]
            )
            if number_of_neurons_assigned_to_this_class == 0:
                continue
            min_latencies[class_number] = self.func(
                latency[assignments == current_class]
            )
        return np.argsort(min_latencies)[::1]

    def _get_assignments(self, latencies, y):
        latencies = np.array(latencies)
        neurons_number = len(latencies[0])
        assignments = [-1] * neurons_number
        minimum_latencies_for_all_neurons = [self.h_time] * neurons_number
        for current_class in self.classes:
            class_size = len(np.where(y == current_class)[0])
            if class_size == 0:
                continue
            latencies_for_this_class = self.func(latencies[y == current_class], axis=0)
            for i in range(neurons_number):
                if latencies_for_this_class[i] < minimum_latencies_for_all_neurons[i]:
                    minimum_latencies_for_all_neurons[i] = latencies_for_this_class[i]
                    assignments[i] = current_class
        return assignments

    def fit(self, X, y=None):
        self.classes = set(y)
        self.assignments = self._get_assignments(X, y)
        return self

    def predict(self, X):
        class_certainty_ranks = [
            self._get_classes_rank_per_one_vector(X[i], self.classes, self.assignments)
            for i in range(len(X))
        ]
        y_predicted = np.array(class_certainty_ranks)[:, 0]
        return y_predicted

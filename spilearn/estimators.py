from .evaluation import *
from .network import *
from .teacher import *
from .utils import *

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class SupervisedTemporalClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, settings, model, **kwargs) -> None:
        self.model = model
        self.settings = settings

        self.n_layer_out = kwargs.get('n_layer_out', settings['topology']['n_layer_out'])
        self.start_delta = kwargs.get('start_delta', settings['network']['start_delta'])
        self.h_time = kwargs.get('h_time', settings['network']['h_time'])
        self._network = EpochNetwork(settings, model, Teacher(settings), progress=False, **kwargs)
        self._devices_fit = None
        self._weights = None

    def fit(self, X, y):
        self._network.n_input = len(X[0])
        self._weights, _, self._devices_fit = self._network.train(X, y)
        return self

    def predict(self, X):
        output, self._devices_predict = self._network.test(X, self._weights)

        all_latency = split_spikes_and_senders(
            output, len(X),
            self.start_delta,
            self.h_time)
        out_latency = convert_latency(all_latency, self.n_layer_out)
        y_pred = predict_from_latency(out_latency)
        return y_pred.astype(int)


class SupervisedTemporalReservoirClassifier(SupervisedTemporalClassifier):

    def __init__(self, settings, model, **kwargs) -> None:
        self.model = model
        self.settings = settings

        self.n_layer_out = kwargs.get('n_layer_out', settings['topology']['n_layer_out'])
        self.start_delta = kwargs.get('start_delta', settings['network']['start_delta'])
        self.h_time = kwargs.get('h_time', settings['network']['h_time'])
        self._network = TwoLayerNetwork(settings, model, Teacher(settings), progress=False, **kwargs)
        self._devices_fit = None
        self._weights = None


class ClasswiseTemporalClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, settings, model, **kwargs) -> None:
        self.model = model
        self.settings = settings

        self.n_layer_out = kwargs.get('n_layer_out', settings['topology']['n_layer_out'])
        self.start_delta = kwargs.get('start_delta', settings['network']['start_delta'])
        self.h_time = kwargs.get('h_time', settings['network']['h_time'])

        self._network = EpochNetwork(settings, model, progress=False, **kwargs)
        self._devices_fit = None
        self._weights = None

    def fit(self, X, y):
        self._weights = []
        self._devices_fit = []
        self._network.n_input = len(X[0])
        for current_class in set(y):
            mask = y == current_class
            weights, _, devices_fit = self._network.train(X[mask], y[mask])
            self._weights.append(weights)
            self._devices_fit.append(devices_fit)
        return self

    def predict(self, X):
        full_output = []
        for weights in self._weights:
            output, self._devices_predict = self._network.test(X, weights)
            all_latency = split_spikes_and_senders(
                output, len(X),
                self.start_delta,
                self.h_time)
            out_latency = convert_latency(all_latency, self.n_layer_out)
            full_output.append(out_latency)
        y_pred = predict_from_latency(np.concatenate(full_output, axis=1))
        return y_pred.astype(int)


class UnsupervisedTemporalTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, settings, model, **kwargs) -> None:
        self.model = model
        self.settings = settings

        self.n_layer_out = kwargs.get('n_layer_out', settings['topology']['n_layer_out'])
        self.start_delta = kwargs.get('start_delta', settings['network']['start_delta'])
        self.h_time = kwargs.get('h_time', settings['network']['h_time'])
        self.reshape = kwargs.get('reshape', True)

        self._network = kwargs.get('network', EpochNetwork(settings, model, progress=False, **kwargs))
        self._devices_fit = None
        self._weights = None

    def fit(self, X, y=None):
        self._network.n_input = len(X[0])
        self._weights, _, self._devices_fit = self._network.train(X, y)
        return self

    def transform(self, X, y=None):
        output, self._devices_predict = self._network.test(X, self._weights)

        all_latency = split_spikes_and_senders(
            output, len(X),
            self.start_delta,
            self.h_time)
        out_latency = np.array(convert_latency(all_latency, self.n_layer_out))
        return out_latency.reshape(out_latency.shape[0], out_latency.shape[1], 1) if self.reshape else out_latency


class UnsupervisedConvolutionTemporalTransformer(UnsupervisedTemporalTransformer):

    def __init__(self, settings, model, **kwargs) -> None:
        self.model = model
        self.settings = settings

        self.n_layer_out = kwargs.get('n_layer_out', settings['topology']['n_layer_out'])
        self.start_delta = kwargs.get('start_delta', settings['network']['start_delta'])
        self.h_time = kwargs.get('h_time', settings['network']['h_time'])

        self._network = ConvolutionNetwork(settings, model, **kwargs)


class ReceptiveFieldsTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, n_fields, sigma2=None, k_round=2, max_x=1.0, scale=1.0,
                 reshape=True, reverse=False, no_last=False) -> None:
        self.sigma2 = sigma2
        self.max_x = max_x
        self.n_fields = n_fields
        self.k_round = k_round
        self.scale = scale
        self.reshape = reshape
        self.reverse = reverse
        self.no_last = no_last

    def _get_sigma_squared(self, min_x, max_x, n_fields):
        # as in [Yu et al. 2014]
        return (
            2/3 * (max_x - min_x) / (n_fields - 2)
        )**2

    def _get_gaussian(self, x, sigma2, mu):
        return (1 / np.sqrt(2 * sigma2 * np.pi)) * np.e ** (- (x - mu) ** 2 / (2 * sigma2))

    def fit(self, X, y=None):
        h_mu = self.max_x / (self.n_fields - 1)
        if self.sigma2 is None:
            self.sigma2 = self._get_sigma_squared(0, self.max_x, self.n_fields)
        self.max_y = np.round(self._get_gaussian(h_mu, self.sigma2, h_mu), 0)
        self.mu = np.tile(np.linspace(0, self.max_x, self.n_fields), len(X[0]))
        return self

    def transform(self, X, y=None):
        X = np.repeat(X, self.n_fields, axis=1)
        assert len(self.mu) == len(X[0])

        X = np.round(self._get_gaussian(X, self.sigma2, self.mu), self.k_round)
        if self.no_last:
            X[X == 0] = np.nan
        X = X if self.reverse else self.max_y - X
        X *= self.scale
        return X.reshape(X.shape[0], X.shape[1], 1) if self.reshape else X.reshape(X.shape[0], X.shape[1])


class TemporalPatternTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, pattern_length, k_round=2, max_x=1.0, reshape=True, reverse=False, no_last=False) -> None:
        self.pattern_length = pattern_length
        self.k_round = k_round
        self.max_x = max_x
        self.reshape = reshape
        self.reverse = reverse
        self.no_last = no_last

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.no_last:
            X[X == 0] = np.nan

        X = X if self.reverse else self.max_x - X
        X = np.round(self.pattern_length * X, self.k_round)
        return X.reshape(X.shape[0], X.shape[1], 1) if self.reshape else X.reshape(X.shape[0], X.shape[1])


class FirstSpikeVotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, h_time) -> None:
        super().__init__()
        self.h_time = h_time
        self.classes = None
        self.assignments = None

    def _get_classes_rank_per_one_vector(self, latency, set_of_classes, assignments):
        latency = np.array(latency)
        number_of_classes = len(set_of_classes)
        min_latencies = [np.nan] * number_of_classes
        number_of_neurons_assigned_to_this_class = [0] * number_of_classes
        for class_number, current_class in enumerate(set_of_classes):
            number_of_neurons_assigned_to_this_class = len(np.where(assignments == current_class)[0])
            if number_of_neurons_assigned_to_this_class == 0:
                continue
            min_latencies[class_number] = np.median(
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
            latencies_for_this_class = np.median(latencies[y == current_class], axis=0)
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
            self._get_classes_rank_per_one_vector(
                X[i], self.classes, self.assignments
            )
            for i in range(len(X))
        ]
        y_predicted = np.array(class_certainty_ranks)[:,0]
        return y_predicted


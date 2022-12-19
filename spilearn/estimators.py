from .evaluation import *
from .network import *
from .teacher import *
from .utils import *

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class TemporalClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, settings):
        self.settings = settings

        self.n_layer_out = settings['topology']['n_layer_out']
        self.start_delta = settings['network']['start_delta']
        self.h_time = settings['network']['h_time']
        self._network = EpochNetwork(settings, Teacher(settings), progress=False)
        self._evaluation = Evaluation(settings)
        
    def fit(self, X, y):
        self._weights, output_fit, self._devices_fit = self._network.train(X, y)

    def predict(self, X):
        output, self._devices_predict = self._network.test(X, self._weights)

        all_latency = split_spikes_and_senders(
            output, len(X),
            self.start_delta,
            self.h_time)
        out_latency = convert_latency(all_latency, self.n_layer_out)
        y_pred = self._evaluation.predict_from_latency(out_latency)
        return y_pred.astype(int)


class ClasswiseTemporalClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, settings):
        self.settings = settings
        self.n_layer_out = settings['topology']['n_layer_out']
        self.start_delta = settings['network']['start_delta']
        self.h_time = settings['network']['h_time']
        
        self._network = EpochNetwork(settings, progress=False)
        self._evaluation = Evaluation(settings)
        
    def fit(self, X, y):
        self._weights = []
        self._devices_fit = []
        for current_class in set(y):
            mask = y == current_class
            weights, output_fit, devices_fit = self._network.train(X[mask], y[mask])
            self._weights.append(weights)
            self._devices_fit.append(devices_fit)

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
        y_pred = self._evaluation.predict_from_latency(np.concatenate(full_output, axis=1))
        return y_pred.astype(int)


class ReceptiveFieldsTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, n_fields, sigma2, k_round=2, max_x=1.0, scale=1.0,
                 reshape=True, reverse=False, no_last=False):
        self.sigma2 = sigma2
        self.max_x = max_x
        self.n_fields = n_fields
        self.k_round = k_round
        self.scale = scale
        self.reshape = reshape
        self.reverse = reverse
        self.no_last = no_last

    def _get_gaussian(self, x, sigma2, mu):
        return (1 / np.sqrt(2 * sigma2 * np.pi)) * np.e ** (- (x - mu) ** 2 / (2 * sigma2))

    def fit(self, X, y=None):
        h_mu = self.max_x / (self.n_fields - 1)

        self.max_y = np.round(self._get_gaussian(h_mu, self.sigma2, h_mu), 0)

        self.mu = np.tile(np.linspace(0, self.max_x, self.n_fields), len(X[0]))

        return self
    
    def transform(self, X, y=None):
        X = np.repeat(X, self.n_fields, axis=1)
        assert len(self.mu) == len(X[0])

        if self.reverse:
            X = np.round(self._get_gaussian(X, self.sigma2, self.mu), self.k_round)
            if self.no_last:
                mask = X < 0.1
                X[mask] = np.nan
        else:
            X = self.max_y - np.round(self._get_gaussian(X, self.sigma2, self.mu), self.k_round)
            if self.no_last:
                mask = X > max_y - 0.09
                X[mask] = np.nan

        X *= self.scale
        if self.reshape:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        return X


class TemporalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, pattern_length, k_round, reshape=True, reverse=False, no_last=False):
        self.pattern_length = pattern_length
        self.k_round = k_round
        self.reshape = reshape
        self.reverse = reverse
        self.no_last = no_last
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.no_last:
            zero_values = X == 0
            X[zero_values] = np.nan
        if self.reverse:
            X = np.round(self.pattern_length * X, self.k_round)
        else:
            X = np.round(self.pattern_length * (1 - X), self.k_round)

        if self.reshape:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        return X


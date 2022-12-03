from spilearn.evaluation import *
from spilearn.network import *
from spilearn.teacher import *

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class TemporalClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, settings):
        teacher = Teacher(settings)
        self._network = EpochNetwork(settings, teacher, progress=False)
        self._evaluation = Evaluation(settings)
        
    def fit(self, X, y):
        self._weights, output_fit, self._devices_fit = self._network.train(X, y)

    def predict(self, X):
        output, self._devices_predict = self._network.test(X, self._weights)

        all_latency = self._evaluation.split_spikes_and_senders(output, len(X))
        out_latency = self._evaluation.convert_latency(all_latency)
        y_pred = self._evaluation.predict_from_latency(out_latency)
        return y_pred.astype(int)

    
class ReceptiveFieldsTransformer(BaseEstimator, TransformerMixin):
    """
        Class for receptive fields data conversion
    """
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


# coding: utf-8

from .evaluation import *
from .network import *
from .teacher import *
from .utils import *

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class BaseTemporalEstimator(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(
        self,
        settings,
        model,
        reshape=True,
        n_layer_out: Optional[int] = None,
        n_input: Optional[int] = None,
        epochs: Optional[int] = None,
        h_time: Optional[float] = None,
        start_delta: Optional[float] = None,
        h: Optional[float] = None,
        **kwargs,
    ) -> None:
        self.model = model
        self.settings = settings

        self.epochs = epochs or settings['learning'].get('epochs', 1)
        self.n_input = n_input or settings['topology'].get('n_input', 2)
        self.n_layer_out = n_layer_out or settings['topology'].get('n_layer_out', 2)
        self.start_delta = start_delta or settings['network'].get('start_delta', 50)
        self.h_time = h_time or settings['network'].get('h_time', 50)
        self.h = h or settings['network'].get('h', 0.01)
        self.reshape = reshape

        self._network = self._init_network(settings, model, **kwargs)
        self._devices_fit = None
        self._weights = None

    def _init_network(self, settings, model, **kwargs):
        return EpochNetwork(
            settings,
            model,
            n_layer_out=self.n_layer_out,
            n_input=self.n_input,
            epochs=self.epochs,
            spike_generator=TemporalSpikeGenerator(
                self.n_input, self.epochs, self.h_time
            ),
            h_time=self.h_time,
            start_delta=self.start_delta,
            h=self.h,
            **kwargs,
        )

    def fit(self, X, y):
        self.n_input = len(X[0])
        self._network.n_input = self.n_input
        self._network.spike_generator.n_input = self.n_input
        self._weights, _, self._devices_fit = self._network.train(X, y)
        return self

    def predict(self, X):
        output, self._devices_predict = self._network.test(X, self._weights)

        all_latency = split_spikes_and_senders(
            output, len(X), self.start_delta, self.h_time
        )
        out_latency = convert_latency(all_latency, self.n_layer_out)
        y_pred = predict_from_latency(out_latency)
        return y_pred.astype(int)

    def transform(self, X, y=None):
        output, self._devices_predict = self._network.test(X, self._weights)

        all_latency = split_spikes_and_senders(
            output, len(X), self.start_delta, self.h_time
        )
        out_latency = np.array(convert_latency(all_latency, self.n_layer_out))
        return (
            out_latency.reshape(out_latency.shape[0], out_latency.shape[1], 1)
            if self.reshape
            else out_latency
        )


class SupervisedTemporalClassifier(BaseTemporalEstimator):
    def __init__(
        self,
        settings,
        model,
        reshape=True,
        n_layer_out: Optional[int] = None,
        n_input: Optional[int] = None,
        epochs: Optional[int] = None,
        h_time: Optional[float] = None,
        start_delta: Optional[float] = None,
        h: Optional[float] = None,
        teacher_amplitude: Optional[float] = None,
        reinforce_delta: Optional[float] = None,
        reinforce_time: Optional[float] = None,
        use_min_teacher=True,
        **kwargs,
    ) -> None:
        self.teacher_amplitude = teacher_amplitude or settings['learning'].get(
            'teacher_amplitude', 1000
        )
        self.reinforce_delta = reinforce_delta or settings['learning'].get(
            'reinforce_delta', 0
        )
        self.reinforce_time = reinforce_time or settings['learning'].get(
            'reinforce_time', 0
        )
        self.use_min_teacher = use_min_teacher
        super().__init__(
            settings,
            model,
            reshape,
            n_layer_out,
            n_input,
            epochs,
            h_time,
            start_delta,
            h,
            **kwargs,
        )

    def _init_network(self, settings, model, **kwargs):
        return EpochNetwork(
            settings,
            model,
            n_layer_out=self.n_layer_out,
            n_input=self.n_input,
            epochs=self.epochs,
            spike_generator=TemporalSpikeGenerator(
                self.n_input, self.epochs, self.h_time
            ),
            teacher=Teacher(
                n_layer_out=self.n_layer_out,
                teacher_amplitude=self.teacher_amplitude,
                reinforce_delta=self.reinforce_delta,
                reinforce_time=self.reinforce_time,
                start=self.start_delta,
                h_time=self.h_time,
                h=self.h,
                use_min=self.use_min_teacher,
            ),
            h_time=self.h_time,
            start_delta=self.start_delta,
            h=self.h,
            **kwargs,
        )


class SupervisedTemporalPoolClassifier(SupervisedTemporalClassifier):
    def __init__(
        self,
        settings,
        model,
        pool_size,
        reshape=True,
        n_layer_out: Optional[int] = None,
        n_input: Optional[int] = None,
        epochs: Optional[int] = None,
        h_time: Optional[float] = None,
        start_delta: Optional[float] = None,
        h: Optional[float] = None,
        teacher_amplitude: Optional[float] = None,
        reinforce_delta: Optional[float] = None,
        reinforce_time: Optional[float] = None,
        use_min_teacher=True,
        **kwargs,
    ) -> None:
        self.pool_size = pool_size
        super().__init__(
            settings,
            model,
            reshape,
            n_layer_out,
            n_input,
            epochs,
            h_time,
            start_delta,
            h,
            teacher_amplitude,
            reinforce_delta,
            reinforce_time,
            use_min_teacher,
            **kwargs,
        )

    def _init_network(self, settings, model, **kwargs):
        return EpochNetwork(
            settings,
            model,
            n_layer_out=self.n_layer_out,
            n_input=self.n_input,
            epochs=self.epochs,
            spike_generator=TemporalSpikeGenerator(
                self.n_input, self.epochs, self.h_time
            ),
            teacher=TeacherPool(
                n_layer_out=self.n_layer_out,
                pool_size=self.pool_size,
                teacher_amplitude=self.teacher_amplitude,
                reinforce_delta=self.reinforce_delta,
                reinforce_time=self.reinforce_time,
                start=self.start_delta,
                h_time=self.h_time,
                h=self.h,
                use_min=self.use_min_teacher,
            ),
            h_time=self.h_time,
            start_delta=self.start_delta,
            h=self.h,
            **kwargs,
        )

    def predict(self, X):
        output, self._devices_predict = self._network.test(X, self._weights)

        all_latency = split_spikes_and_senders(
            output, len(X), self.start_delta, self.h_time
        )
        out_latency = convert_latency_pool(
            all_latency, self.n_layer_out, self.pool_size
        )
        y_pred = predict_from_latency_pool(out_latency)
        return y_pred.astype(int)


class SupervisedTemporalRLClassifier(BaseTemporalEstimator):
    def __init__(
        self,
        settings,
        model,
        learning_rate,
        reshape=True,
        n_layer_out: Optional[int] = None,
        n_input: Optional[int] = None,
        epochs: Optional[int] = None,
        h_time: Optional[float] = None,
        start_delta: Optional[float] = None,
        h: Optional[float] = None,
        **kwargs,
    ) -> None:
        self.learning_rate = learning_rate
        super().__init__(
            settings,
            model,
            reshape,
            n_layer_out,
            n_input,
            epochs,
            h_time,
            start_delta,
            h,
            **kwargs,
        )

    def _init_network(self, settings, model, **kwargs):
        return LiteRlNetwork(
            settings,
            model,
            learning_rate=self.learning_rate,
            n_layer_out=self.n_layer_out,
            n_input=self.n_input,
            epochs=self.epochs,
            spike_generator=TemporalSpikeGenerator(
                self.n_input, self.epochs, self.h_time
            ),
            h_time=self.h_time,
            start_delta=self.start_delta,
            h=self.h,
            **kwargs,
        )


class SupervisedTemporalRLPoolClassifier(SupervisedTemporalRLClassifier):
    def __init__(
        self,
        settings,
        model,
        pool_size,
        learning_rate,
        reshape=True,
        n_layer_out: Optional[int] = None,
        n_input: Optional[int] = None,
        epochs: Optional[int] = None,
        h_time: Optional[float] = None,
        start_delta: Optional[float] = None,
        h: Optional[float] = None,
        **kwargs,
    ) -> None:
        self.pool_size = pool_size
        super().__init__(
            settings,
            model,
            learning_rate,
            reshape,
            n_layer_out,
            n_input,
            epochs,
            h_time,
            start_delta,
            h,
            **kwargs,
        )

    def _init_network(self, settings, model, **kwargs):
        return LitePoolRlNetwork(
            settings,
            model,
            pool_size=self.pool_size,
            learning_rate=self.learning_rate,
            n_layer_out=self.n_layer_out,
            n_input=self.n_input,
            epochs=self.epochs,
            spike_generator=TemporalSpikeGenerator(
                self.n_input, self.epochs, self.h_time
            ),
            h_time=self.h_time,
            start_delta=self.start_delta,
            h=self.h,
            **kwargs,
        )

    def predict(self, X):
        output, self._devices_predict = self._network.test(X, self._weights)

        all_latency = split_spikes_and_senders(
            output, len(X), self.start_delta, self.h_time
        )
        out_latency = convert_latency_pool(
            all_latency, self.n_layer_out, self.pool_size
        )
        y_pred = predict_from_latency_pool(out_latency)
        return y_pred.astype(int)


class SupervisedTemporalReservoirClassifier(SupervisedTemporalClassifier):
    def __init__(
        self,
        settings,
        model,
        reshape=True,
        n_layer_out: Optional[int] = None,
        n_input: Optional[int] = None,
        epochs: Optional[int] = None,
        h_time: Optional[float] = None,
        start_delta: Optional[float] = None,
        h: Optional[float] = None,
        teacher_amplitude: Optional[float] = None,
        reinforce_delta: Optional[float] = None,
        reinforce_time: Optional[float] = None,
        use_min_teacher=True,
        **kwargs,
    ) -> None:
        super().__init__(
            settings,
            model,
            reshape,
            n_layer_out,
            n_input,
            epochs,
            h_time,
            start_delta,
            h,
            teacher_amplitude,
            reinforce_delta,
            reinforce_time,
            use_min_teacher,
            **kwargs,
        )

    def _init_network(self, settings, model, **kwargs):
        return TwoLayerNetwork(
            settings,
            model,
            n_layer_out=self.n_layer_out,
            n_input=self.n_input,
            epochs=self.epochs,
            spike_generator=TemporalSpikeGenerator(
                self.n_input, self.epochs, self.h_time
            ),
            teacher=Teacher(
                n_layer_out=self.n_layer_out,
                teacher_amplitude=self.teacher_amplitude,
                reinforce_delta=self.reinforce_delta,
                reinforce_time=self.reinforce_time,
                start=self.start_delta,
                h_time=self.h_time,
                h=self.h,
                use_min=self.use_min_teacher,
            ),
            h_time=self.h_time,
            start_delta=self.start_delta,
            h=self.h,
            **kwargs,
        )


class SupervisedConvolutionRLClassifier(BaseTemporalEstimator):
    def __init__(
        self,
        settings,
        model,
        kernel_size,
        stride,
        learning_rate,
        n_layer_hid,
        reshape=True,
        n_layer_out: Optional[int] = None,
        n_input: Optional[int] = None,
        epochs: Optional[int] = None,
        h_time: Optional[float] = None,
        start_delta: Optional[float] = None,
        h: Optional[float] = None,
        **kwargs,
    ) -> None:
        self.learning_rate = learning_rate
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_layer_hid = n_layer_hid
        super().__init__(
            settings,
            model,
            reshape,
            n_layer_out,
            n_input,
            epochs,
            h_time,
            start_delta,
            h,
            **kwargs,
        )

    def _init_network(self, settings, model, **kwargs):
        return ConvolutionRlNetwork(
            settings,
            model,
            learning_rate=self.learning_rate,
            n_layer_out=self.n_layer_out,
            n_layer_hid=self.n_layer_hid,
            n_input=self.n_input,
            epochs=self.epochs,
            spike_generator=TemporalSpikeGenerator(
                self.n_input, self.epochs, self.h_time
            ),
            kernel_size=self.kernel_size,
            stride=self.stride,
            h_time=self.h_time,
            start_delta=self.start_delta,
            h=self.h,
            **kwargs,
        )


class ClasswiseTemporalClassifier(BaseTemporalEstimator):
    def __init__(
        self,
        settings,
        model,
        reshape=True,
        n_layer_out: Optional[int] = None,
        n_input: Optional[int] = None,
        epochs: Optional[int] = None,
        h_time: Optional[float] = None,
        start_delta: Optional[float] = None,
        h: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            settings,
            model,
            reshape,
            n_layer_out,
            n_input,
            epochs,
            h_time,
            start_delta,
            h,
            **kwargs,
        )

    def fit(self, X, y):
        self.n_input = len(X[0])
        self._weights = []
        self._devices_fit = []
        self._network.n_input = self.n_input
        self._network.spike_generator.n_input = self.n_input
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
                output, len(X), self.start_delta, self.h_time
            )
            out_latency = convert_latency(all_latency, self.n_layer_out)
            full_output.append(out_latency)
        y_pred = predict_from_latency(np.concatenate(full_output, axis=1))
        return y_pred.astype(int)

    def transform(self, X, y=None):
        full_output = []
        for weights in self._weights:
            output, self._devices_predict = self._network.test(X, weights)
            all_latency = split_spikes_and_senders(
                output, len(X), self.start_delta, self.h_time
            )
            out_latency = convert_latency(all_latency, self.n_layer_out)
            full_output.append(out_latency)
        out_latency = np.concatenate(full_output, axis=1)
        return (
            out_latency.reshape(out_latency.shape[0], out_latency.shape[1], 1)
            if self.reshape
            else out_latency
        )


class UnsupervisedTemporalTransformer(BaseTemporalEstimator):
    def __init__(
        self,
        settings,
        model,
        reshape=True,
        n_layer_out: Optional[int] = None,
        n_input: Optional[int] = None,
        epochs: Optional[int] = None,
        h_time: Optional[float] = None,
        start_delta: Optional[float] = None,
        h: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            settings,
            model,
            reshape,
            n_layer_out,
            n_input,
            epochs,
            h_time,
            start_delta,
            h,
            **kwargs,
        )


class UnsupervisedTemporalNoiseTransformer(BaseTemporalEstimator):
    def __init__(
        self,
        settings,
        model,
        reshape=True,
        noise: Optional[float] = None,
        n_layer_out: Optional[int] = None,
        n_input: Optional[int] = None,
        epochs: Optional[int] = None,
        h_time: Optional[float] = None,
        start_delta: Optional[float] = None,
        h: Optional[float] = None,
        **kwargs,
    ) -> None:
        self.noise = noise
        super().__init__(
            settings,
            model,
            reshape,
            n_layer_out,
            n_input,
            epochs,
            h_time,
            start_delta,
            h,
            **kwargs,
        )

    def _init_network(self, settings, model, **kwargs):
        return EpochNetwork(
            settings,
            model,
            n_layer_out=self.n_layer_out,
            n_input=self.n_input,
            epochs=self.epochs,
            spike_generator=TemporalSpikeGenerator(
                self.n_input, self.epochs, self.h_time
            ),
            noise=NoiseGenerator(noise_freq=self.noise, n_input=self.n_input),
            h_time=self.h_time,
            start_delta=self.start_delta,
            h=self.h,
            **kwargs,
        )


class UnsupervisedConvolutionTemporalTransformer(UnsupervisedTemporalTransformer):
    def __init__(
        self,
        settings,
        model,
        kernel_size,
        stride,
        reshape=True,
        n_layer_out: Optional[int] = None,
        n_input: Optional[int] = None,
        epochs: Optional[int] = None,
        h_time: Optional[float] = None,
        start_delta: Optional[float] = None,
        h: Optional[float] = None,
        **kwargs,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        super().__init__(
            settings,
            model,
            reshape,
            n_layer_out,
            n_input,
            epochs,
            h_time,
            start_delta,
            h,
            **kwargs,
        )

    def _init_network(self, settings, model, **kwargs):
        return ConvolutionNetwork(
            settings,
            model,
            n_layer_out=self.n_layer_out,
            n_input=self.n_input,
            epochs=self.epochs,
            spike_generator=TemporalSpikeGenerator(
                self.n_input, self.epochs, self.h_time
            ),
            kernel_size=self.kernel_size,
            stride=self.stride,
            h_time=self.h_time,
            start_delta=self.start_delta,
            h=self.h,
            **kwargs,
        )


class UnsupervisedConvolutionTemporalNoiseTransformer(UnsupervisedTemporalTransformer):
    def __init__(
        self,
        settings,
        model,
        kernel_size,
        stride,
        reshape=True,
        noise: Optional[float] = None,
        n_layer_out: Optional[int] = None,
        n_input: Optional[int] = None,
        epochs: Optional[int] = None,
        h_time: Optional[float] = None,
        start_delta: Optional[float] = None,
        h: Optional[float] = None,
        **kwargs,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.noise = noise
        super().__init__(
            settings,
            model,
            reshape,
            n_layer_out,
            n_input,
            epochs,
            h_time,
            start_delta,
            h,
            **kwargs,
        )

    def _init_network(self, settings, model, **kwargs):
        return ConvolutionNetwork(
            settings,
            model,
            n_layer_out=self.n_layer_out,
            n_input=self.n_input,
            epochs=self.epochs,
            spike_generator=TemporalSpikeGenerator(
                self.n_input, self.epochs, self.h_time
            ),
            noise=NoiseGenerator(noise_freq=self.noise, n_input=self.n_input),
            kernel_size=self.kernel_size,
            stride=self.stride,
            h_time=self.h_time,
            start_delta=self.start_delta,
            h=self.h,
            **kwargs,
        )


class ReceptiveFieldsTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_fields,
        sigma2=None,
        k_round=2,
        max_x=1.0,
        scale=1.0,
        reshape=True,
        reverse=False,
        no_last=False,
    ) -> None:
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
        return (2 / 3 * (max_x - min_x) / (n_fields - 2)) ** 2

    def _get_gaussian(self, x, sigma2, mu):
        return (1 / np.sqrt(2 * sigma2 * np.pi)) * np.e ** (
            -((x - mu) ** 2) / (2 * sigma2)
        )

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
        return (
            X.reshape(X.shape[0], X.shape[1], 1)
            if self.reshape
            else X.reshape(X.shape[0], X.shape[1])
        )


class TemporalPatternTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        pattern_length,
        k_round=2,
        max_x=1.0,
        reshape=True,
        reverse=False,
        no_last=False,
    ) -> None:
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
        return (
            X.reshape(X.shape[0], X.shape[1], 1)
            if self.reshape
            else X.reshape(X.shape[0], X.shape[1])
        )


class TemporalPatternReversedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, pattern_length, k_round=2, max_x=1.0, reshape=True) -> None:
        self.pattern_length = pattern_length
        self.k_round = k_round
        self.max_x = max_x
        self.reshape = reshape

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[X == 0] = self.max_x
        X = np.round(self.pattern_length * X, self.k_round)
        return (
            X.reshape(X.shape[0], X.shape[1], 1)
            if self.reshape
            else X.reshape(X.shape[0], X.shape[1])
        )


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

# coding: utf-8

import nest
import numpy as np
from tqdm import tqdm

from .epoch import EpochNetwork
from ..utils import (
    convert_latency,
    predict_from_latency,
)


class LiteRlNetwork(EpochNetwork):
    def __init__(self, settings, model, learning_rate, **kwargs):
        super().__init__(settings, model, **kwargs)

        self.learning_rate_default = learning_rate
        self.learning_rate = learning_rate
        self.learning_rate_scaled = learning_rate
        self.spikes_length = 0
        self.time = 0

    def get_previous_layer(self):
        return self.input_layer

    def update_learning_rate(self, learning_rate):
        connection = nest.GetConnections(self.get_previous_layer(), target=self.layer_out)
        nest.SetStatus(connection, 'lambda', learning_rate)

    def update_learning_rate_separate(self, learning_rate, action):
        connection = nest.GetConnections(
            self.get_previous_layer(), target=self.layer_out[action]
        )
        nest.SetStatus(connection, 'lambda', learning_rate)

    def sim(self):
        nest.Run(self.h_time)

        spikes = nest.GetStatus(self.spike_detector_out, keys='events')[0]['times']
        senders = nest.GetStatus(self.spike_detector_out, keys='events')[0]['senders']
        self.spikes_length += spikes.size
        mask = spikes > self.time
        raw_latency = {'spikes': spikes[mask], 'senders': senders[mask]}
        raw_latency['spikes'] -= self.time
        return raw_latency

    def run(self, n_neurons):
        self.update_learning_rate(0.0)

        raw_latency = self.sim()
        self.time += self.h_time

        out_latency = convert_latency([raw_latency], n_neurons)
        y_pred = predict_from_latency([out_latency[0]])
        return int(y_pred[0])

    def learn(self, action, correct_class):
        self.update_learning_rate(0)
        self.update_learning_rate_separate(self.learning_rate, action)
        self.update_learning_rate_separate(self.learning_rate_scaled, correct_class)

        nest.Run(self.h_time)
        self.time += self.h_time
        return

    # TODO: move somewhere
    def ema_update(self, prev, acc, alpha=0.05):
        return acc if prev is None else (1 - alpha) * prev + alpha * acc

    def lr_scale(self, A, A0=0.75, k=4.0):
        return 1.0 if A < A0 else np.exp(-k * (A - A0))

    def simulate(
        self,
        full_time,
        spike_dict,
        epochs=1,
        teacher_dicts=None,
        normalize_weights=False,
        y=None,
    ):
        progress_bar = tqdm(
            total=epochs * self.data_len * int(self.progress)
            + epochs * int(not self.progress),
            disable=not self.progress,
        )

        assert len(self.layers[1:]) == len(self.synapse_models), "Number of layers and synapse models do not match."
        nest.Prepare()
        scale = 1
        counter = 0
        running_accuracy = None
        for _ in range(epochs):
            self.spike_generator.set_input_spikes(spike_dict=spike_dict)

            accuracy_train = 0
            if y is not None:
                nest.Run(self.start_delta)
                self.time += self.start_delta
                for i in range(self.data_len):
                    target = y[i]
                    pred = self.run(self.n_layer_out)
                    if pred == target:
                        accuracy_train += 1
                        counter += 1
                        self.learning_rate = self.learning_rate_default * scale
                    else:
                        self.learning_rate = -1 * self.learning_rate_default * scale
                    self.learning_rate_scaled = self.learning_rate_default * scale
                    self.learn(pred, target)
                    if normalize_weights and ((
                        self.normalize_step and counter % self.normalize_step == 0
                    ) or not self.normalize_step):
                        for layer, model, w_target in zip(
                            self.layers[1:], self.synapse_models, self.w_targets
                        ):
                            self.normalize(layer, model, w_target)
                    progress_bar.update()
                accuracy_train /= self.data_len
                running_accuracy = self.ema_update(running_accuracy, accuracy_train)
                scale = self.lr_scale(running_accuracy)
            else:
                nest.Run(full_time)
                progress_bar.update()

            for spikes in spike_dict:
                spikes['spike_times'] += full_time
            # counter += counter_batch
            # counter /= self.data_len

        nest.Cleanup()
        progress_bar.close()

    def train(self, x, y=None):
        self.init_network()
        self.create_layers()
        self.create_devices()

        self.connect_devices()
        self.connect_layers()

        if self.use_inhibition:
            self.connect_layers_inh()

        self.set_neuron_status(self.high_threshold_teacher)

        self.data_len = len(x)
        x = np.repeat(x, 2, axis=0)

        spike_dict = None
        full_time = None
        if self.spike_generator:
            self.spike_generator.create_devices()
            self.spike_generator.connect_devices(self.input_layer)

            spike_dict, full_time = self.spike_generator.create_spike_dict(
                dataset=x, tile=self.tile_train, delta=self.start_delta
            )

        #         if self.noise_after_pattern and self.interpattern_noise_generator:
        #             noise_dict = self.create_poisson_noise(spike_dict)
        #             self.set_poisson_noise(
        #                 noise_dict,
        #                 self.interpattern_noise_generator)
        #         elif self.poisson_layer:
        #             self.set_noise()
        teacher_dicts = None
        self.simulate(
            full_time,
            spike_dict,
            self.epochs,
            teacher_dicts,
            self.normalize_weights,
            y=y,
        )

        weights = self.save_weights(self.layers, self.synapse_models)
        output = {
            'spikes': nest.GetStatus(self.spike_detector_out, keys='events')[0][
                'times'
            ].tolist(),
            'senders': nest.GetStatus(self.spike_detector_out, keys='events')[0][
                'senders'
            ].tolist(),
        }
        devices = self.get_devices() if self.need_devices else None
        return weights, output, devices

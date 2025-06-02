# coding: utf-8

import nest
import numpy as np


class TemporalSpikeGenerator:
    def __init__(self, n_input: int, epochs: int, h_time: float) -> None:
        self.n_input = n_input
        self.h_time = h_time
        self.epochs = epochs

        self.input_generators = None

    def create_devices(self):
        self.input_generators = nest.Create(
            'spike_train_injector',
            self.n_input
        )

    def connect_devices(self, input_layer):
        nest.Connect(
            self.input_generators,
            input_layer,
            'one_to_one',
            syn_spec='static_synapse'
        )

    def set_input_spikes(self, spike_dict):
        assert len(spike_dict) == len(self.input_generators)
        nest.SetStatus(self.input_generators, spike_dict)

    def create_spike_dict(self, dataset, tile=False, delta=0.0):
        spikes = np.tile(dataset, (self.epochs, 1, 1)) if tile else dataset

        pattern_start_shape = (len(dataset[0]), 1)
        full_time = len(spikes) * self.h_time + delta

        spike_times = self.create_spike_times(
            spikes, delta, full_time, pattern_start_shape
        )

        spike_dict = [None] * len(dataset[0])
        for input_neuron in range(len(dataset[0])):
            tmp_spikes = spike_times[:, input_neuron]#.reshape(len(spikes))
            spike_dict[input_neuron] = {
                'spike_times': tmp_spikes[np.isfinite(tmp_spikes)]
            }
        return spike_dict, full_time

    def create_spike_times(self, current_spikes, start, end,
                           pattern_start_shape):
        times = np.arange(start, end, self.h_time)
        pattern_start_times = np.expand_dims(
            np.tile(times, pattern_start_shape).T, axis=2)
        assert len(current_spikes) == len(pattern_start_times)
        spike_times = np.add(current_spikes, pattern_start_times)
        return spike_times


class WeightedTemporalSpikeGenerator(TemporalSpikeGenerator):
    def __init__(self, spike_weights_scale: float, n_input: int, epochs: int, h_time: float) -> None:
        super().__init__(n_input, epochs, h_time)
        self.spike_weights_scale = spike_weights_scale

    def create_spike_weights(self, spikes, spike_weights_scale):
        weights = np.exp(-spikes / spike_weights_scale)
        return weights

    def create_spike_dict(self, dataset, tile=False, delta=0.0):
        pattern_start_shape = (len(dataset[0]), 1)
        full_time = len(dataset) * self.h_time + delta

        spike_times = self.create_spike_times(
            dataset, delta, full_time, pattern_start_shape
        )

        spike_weights = self.create_spike_weights(
            dataset, self.spike_weights_scale
        )

        spike_dict = [None] * len(dataset[0])
        for input_neuron in range(len(dataset[0])):
            tmp_spikes = spike_times[:, input_neuron]
            tmp_weights = spike_weights[:, input_neuron]
            spike_dict[input_neuron] = {
                'spike_times': tmp_spikes[np.isfinite(tmp_spikes)],
                'spike_weights': tmp_weights[np.isfinite(tmp_weights)]
            }
        return spike_dict, full_time


class FrequencySpikeGenerator(TemporalSpikeGenerator):
    def __init__(self, pattern_length: float, n_input: int, epochs: int, h_time: float):
        super().__init__(n_input, epochs, h_time)
        self.pattern_length = pattern_length

    def create_spike_dict(self, dataset, tile=False, delta=0.0):
        spikes = []
        d_time = delta

        spike_dict = [None] * len(dataset[0])
        for input_neuron in range(len(dataset[0])):
            spike_dict[input_neuron] = {'spike_times': []}

        # TODO
        # calc spike times one time and concatenate
        for _ in range(self.epochs):
            for example in dataset:
                tmp_spikes = []
                for input_neuron in example:
                    spike_dict[input_neuron]['spike_times'] \
                        += map(lambda x: x + d_time, example[input_neuron])
                    tmp_spikes.append(example[input_neuron])
                spikes.append(tmp_spikes)
                d_time += self.h_time
        return spike_dict, d_time

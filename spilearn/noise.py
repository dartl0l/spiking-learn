# coding: utf-8

import nest
import numpy as np


class NoiseGenerator():
    def __init__(
            self, noise_freq, n_input, epochs=1, 
            test_with_noise=False, noise_after_pattern=False, 
            h_time=50, start_delta=50, h=0.01
        ):

        self.n_input = n_input
        self.noise_freq = noise_freq
        self.test_with_noise = test_with_noise
        self.noise_after_pattern = noise_after_pattern

        self.h = h
        self.h_time = h_time
        self.epochs = epochs
        self.start_delta = start_delta

        self.layer = None

    def create(self):
        if self.noise_after_pattern:
            self.layer = nest.Create(
                'spike_generator',
                self.n_input
            )
        else:
            self.layer = nest.Create(
                'poisson_generator',
                self.n_input
            )

    def connect(self, input_layer):
        if self.layer:
            nest.Connect(
                self.layer,
                input_layer,
                'one_to_one',
                syn_spec='static_synapse'
            )

    def create_poisson_noise(self, spikes, tile):
        dt = 1.0 / 1000.0
        h = self.h
        h_time = self.h_time
        epochs = self.epochs
        start = self.start_delta
        p_spike = self.noise_freq * dt * h

        spikes_list = np.tile(spikes, (epochs, 1, 1)) if tile else spikes

        num_inputs = len(spikes_list[0])

        num_total_patterns = epochs * len(spikes_list)
        end = num_total_patterns * h_time + start

        start_pattern_times = np.linspace(start, end, num_total_patterns,
                                          endpoint=False)
        end_pattern_times = start_pattern_times + h_time

        start_noise_times = np.tile(
            np.amax(spikes_list, axis=1),
            epochs
        ).reshape(start_pattern_times.shape) + start_pattern_times

        num_per_pattern = int((end_pattern_times[0] - start_noise_times[0]) / h)
        times = np.linspace(
            start_noise_times,
            end_pattern_times,
            num_per_pattern,
            endpoint=False, axis=1).ravel()
        
        # np.random.random_sample is slow
        noise_dict = [None] * num_inputs
        for input_neuron in range(num_inputs):
            random_distribution_mask = \
                np.random.random_sample(len(times)) < p_spike
            noise_dict[input_neuron] = {
                'spike_times': times[random_distribution_mask]
            }
        return noise_dict

    def set(self, x, tile=False):
        if self.noise_after_pattern:
            noise_dict = self.create_poisson_noise(x, tile)
            nest.SetStatus(self.layer, noise_dict)
        else:
            nest.SetStatus(
                self.layer,
                {
                    'rate': self.noise_freq,
                    'origin': 0.0
                }
            )


class FrequencyNetworkNoiseGenerator(NoiseGenerator):
    def __init__(self, pattern_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pattern_length = pattern_length

    def create_poisson_noise(self, spikes_list):
        dt = 1.0 / 1000.0
        h = self.h
        h_time = self.h_time
        epochs = self.epochs
        start = self.start_delta
        pattern_length = self.pattern_length
        noise_frequency = self.noise_freq

        num_inputs = len(spikes_list[0])

        num_total_patterns = epochs * len(spikes_list)
        end = num_total_patterns * h_time + start

        start_pattern_times = np.linspace(start, end, num_total_patterns,
                                          endpoint=False)
        end_pattern_times = start_pattern_times + h_time

        start_noise_times = start_pattern_times + pattern_length

        num_per_pattern = int((end_pattern_times[0] - start_noise_times[0]) / h)
        times = np.linspace(
            start_noise_times,
            end_pattern_times,
            num_per_pattern,
            endpoint=False, axis=1).ravel()

        # np.random.random_sample is slow
        noise_dict = [None] * num_inputs
        for input_neuron in range(num_inputs):
            random_distribution_mask = \
                np.random.random_sample(len(times)) < noise_frequency * dt
            noise_dict[input_neuron] = {
                'spike_times': times[random_distribution_mask]
            }
        return noise_dict

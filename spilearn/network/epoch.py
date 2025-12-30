# coding: utf-8

import nest
import numpy as np
from tqdm import tqdm

from .base import Network


class EpochNetwork(Network):
    def __init__(self, settings, model, **kwargs):
        super().__init__(settings, model, **kwargs)
        self.progress = kwargs.get('progress', False)
        self.normalize_weights = kwargs.get('normalize_weights', False)
        self.normalize_step = kwargs.get('normalize_step', None)
        self.tile_train = False

    # def normalize(self, w_target=1):
    #     weights = self.save_weights(self.layers, self.synapse_models)

    #     for layer_weights in weights:
    #         for neuron in layer_weights:
    #             w = np.array(layer_weights[neuron])
    #             w_norm = w / sum(abs(w))
    #             w = w_target * w_norm
    #             layer_weights[neuron] = w
    #     self.set_weights(weights)

    def normalize(self, neurons_to_be_normalized, w_target=1):
        for neuron in neurons_to_be_normalized:
            conn = nest.GetConnections(target=neuron)
            w = np.array(conn.weight)
            w_normed = w / sum(abs(w))  # L1-norm
            conn.weight = w_target * w_normed

    def simulate(self, full_time, spike_dict, teacher_dicts=None, epochs=1):
        normalize_weights = self.normalize_weights
        progress_bar = tqdm(
            total=epochs * self.data_len * int(self.progress)
            + epochs * int(not self.progress),
            disable=not self.progress,
        )

        nest.Prepare()
        for _ in range(epochs):
            self.spike_generator.set_input_spikes(spike_dict=spike_dict)
            if teacher_dicts and self.teacher:
                self.teacher.set_teachers_input(teacher_dicts)
            if normalize_weights:
                nest.Run(self.start_delta)
                for i in range(self.data_len):
                    nest.Run(self.h_time)
                    progress_bar.update()
                    if (
                        self.normalize_step and i % self.normalize_step == 0
                    ) or not self.normalize_step:
                        for layer in self.layers[1:]:
                            self.normalize(layer)
            elif self.progress:
                nest.Run(self.start_delta)
                for _ in range(self.data_len):
                    nest.Run(self.h_time)
                    progress_bar.update()
            else:
                nest.Run(full_time)
                progress_bar.update()

            for spikes in spike_dict:
                spikes['spike_times'] += full_time
            if teacher_dicts and self.teacher:
                for teacher in teacher_dicts:
                    teacher_dicts[teacher]['amplitude_times'] += full_time
        nest.Cleanup()
        progress_bar.close()

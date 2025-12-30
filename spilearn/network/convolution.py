# coding: utf-8

import math

import nest
import numpy as np

from .epoch import EpochNetwork


class ConvolutionNetwork(EpochNetwork):
    def __init__(self, settings, model, kernel_size, stride=1, **kwargs):
        super().__init__(settings, model, **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.image_dimension = int(math.sqrt(self.n_input))
        self.n_combinations = int(
            (((self.image_dimension - self.kernel_size) / self.stride) + 1) ** 2
        )

        assert self.n_combinations > 0, (
            f'image_dimension={self.image_dimension}, \
            kernel_size={self.kernel_size}, stride={self.stride}'
        )

        self.n_combination_neurons = self.get_neuron_number() // self.n_combinations

        self.two_dimensional_image_indices = np.arange(self.n_input).reshape(
            self.image_dimension, self.image_dimension
        )

    def get_neuron_number(self):
        return self.n_layer_out

    def get_layer_conv(self):
        return self.layer_out

    def get_layer_in(self):
        return self.input_layer

    def get_indexes(self, image_row, image_column, current_combination):
        input_indexes = np.concatenate(
            self.two_dimensional_image_indices[
                image_row : image_row + self.kernel_size,
                image_column : image_column + self.kernel_size,
            ]
        )
        out_layer = self.get_layer_conv()
        input_indexes = np.array(self.get_layer_in())[input_indexes]
        output_indexes = np.array(
            out_layer[
                current_combination : current_combination + self.n_combination_neurons
            ]
        )
        return input_indexes, output_indexes

    def connect_exc(self, spec):
        current_combination = 0
        for image_row in range(
            0, self.image_dimension - self.kernel_size + 1, self.stride
        ):
            for image_column in range(
                0, self.image_dimension - self.kernel_size + 1, self.stride
            ):
                input_indexes, output_indexes = self.get_indexes(
                    image_row, image_column, current_combination
                )
                nest.Connect(
                    nest.NodeCollection(input_indexes),
                    nest.NodeCollection(output_indexes),
                    'all_to_all',
                    syn_spec=spec,
                )
                current_combination += self.n_combination_neurons

    def connect_inh(self, syn_dict, conn_spec):
        current_combination = 0
        for image_row in range(
            0, self.image_dimension - self.kernel_size + 1, self.stride
        ):
            for image_column in range(
                0, self.image_dimension - self.kernel_size + 1, self.stride
            ):
                _, output_indexes = self.get_indexes(
                    image_row, image_column, current_combination
                )

                self.interconnect_layer(
                    nest.NodeCollection(output_indexes),
                    syn_dict, conn_spec,
                )
                current_combination += self.n_combination_neurons

    def connect_layers(self):
        self.connect_exc(self.model['syn_dict_exc'])

    def connect_layers_static(self):
        self.connect_exc('static_synapse')

    def connect_layers_inh(self):
        self.connect_inh(
            self.model['syn_dict_inh'],
            self.model.get('conn_dict_inh', {'rule': 'all_to_all'})
        )

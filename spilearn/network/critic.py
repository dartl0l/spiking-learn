# coding: utf-8

import nest

from .epoch import EpochNetwork


class CriticNetwork(EpochNetwork):
    def __init__(self, settings, model, critic_neuron_count=1, **kwargs):
        super().__init__(settings, model, **kwargs)

        self.critic_neuron_count = critic_neuron_count

    def interconnect_layer(self, layer, syn_dict, conn_spec):
        for neuron_1 in layer[: -self.critic_neuron_count]:
            for neuron_2 in layer[: -self.critic_neuron_count]:
                if neuron_1 != neuron_2:
                    nest.Connect(
                        neuron_1, neuron_2, syn_spec=syn_dict, conn_spec=conn_spec
                    )

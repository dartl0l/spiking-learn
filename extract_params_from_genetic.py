import sys
# sys.path.append('./src/')
# from parameters import traits
import numpy as np


def collect_parameters(input_directory, output_directory):
    traits_set = (traits.network_traits, traits.neuron_traits, traits.synapse_traits)
    for current_traits in traits_set:
        for parameter_name in current_traits:
            current_parameter = current_traits[parameter_name]
            parameter_is_good = current_parameter['mutation_prob'] > 0 \
                              and current_parameter['details']['max'] \
                              != current_parameter['details']['min']
            if parameter_is_good:
                print(
                    '#', 'generation', parameter_name,
                    '\n#',
                    'min:', current_parameter['details']['min'],
                    'max:', current_parameter['details']['max'],
                    file=open(
                        output_directory
                        + 'parameter_'
                        + parameter_name
                        + '-good_values_for_histogram.txt',
                        'w'
                    ),
                    sep='\t'
                )

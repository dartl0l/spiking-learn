# coding: utf-8


# Network parameters
# ------------------
#   time_max: 10000000 # [ms]
#   one_vector_longtitude: 2000 # [ms] one input vector presenting duration
#   high_input_rate: 25 # [Hz], frequency coding an input vector component of 1
#   low_input_rate: 0 # [Hz], frequency coding an input vector component of 0
#   inh_rate: 100 # [Hz], frequency of inhibitory signal presented to wrong-class neurons
#   weights_watching_step: 1000 # [ms]
#   number_of_inputs: 100
#   number_of_excitatory_inputs: 100
#   number_of_inh_generators: 1 # not to be confused with inputs. These generators feed noise to wrong-class neurons
#   weight_scale: 150 # * 50 # multiply roughly by 16 in case of Maass-Markram existence
#   weight_initial_mean: 0.3
#   weight_initial_sigma: 0.01
#   initial_inhibitory_weight: 1.0 # if positive, wrong-class neurons get excitatory signal
data_traits = {
    # 'n_coding_neurons': {
    #   'details': {
    #       'min': 5,
    #       'max': 40,
    #       'mut_power': 1, # the highest possible amount of change during one mutation
    #       'mut_replace_prob': 0.25 # if mutated, the probability to be re-initialized to a random value, rather than changed by a random value proportional to mut_power
    #   },
    #   'importance_coeff': 1.0, # the trait value is multiplied by importance_coeff when calculating distance between genomes
    #   'mutation_prob': 0.1,
    #   'type': 'int'
    # },
    #
    #
    # 'coding_sigma': {
    #   'details': {
    #       'min': 0.005,
    #       'max': 0.1,
    #       'mut_power': 0.005, # the highest possible amount of change during one mutation
    #       'mut_replace_prob': 0.25 # if mutated, the probability to be re-initialized to a random value, rather than changed by a random value proportional to mut_power
    #   },
    #   'importance_coeff': 1.0, # the trait value is multiplied by importance_coeff when calculating distance between genomes
    #   'mutation_prob': 0.1,
    #   'type': 'float'
    # },
}


network_traits = {
    # 'h_time': {
    #   'details': {
    #       'min': 5,
    #       'max': 30,
    #       'mut_power': 5, # the highest possible amount of change during one mutation
    #       'mut_replace_prob': 0.25 # if mutated, the probability to be re-initialized to a random value, rather than changed by a random value proportional to mut_power
    #   },
    #   'importance_coeff': 1.0, # the trait value is multiplied by importance_coeff when calculating distance between genomes
    #   'mutation_prob': 0.1,
    #   'type': 'int'
    # },


    #'epochs': {
    #   'details': {
    #       'min': 1000,
    #       'max': 10000,
    #       'mut_power': 500, # the highest possible amount of change during one mutation
    #       'mut_replace_prob': 0.25 # if mutated, the probability to be re-initialized to a random value, rather than changed by a random value proportional to mut_power
    #   },
    #   'importance_coeff': 1.0, # the trait value is multiplied by importance_coeff when calculating distance between genomes
    #   'mutation_prob': 0.1,
    #   'type': 'int'
    #}

    # 'noise_freq': {
    #     'details': {
    #         'min': 1.0,
    #         'max': 5.0,
    #         'mut_power': 1.0, # the highest possible amount of change during one mutation
    #         'mut_replace_prob': 0.25 # if mutated, the probability to be re-initialized to a random value, rather than changed by a random value proportional to mut_power
    #     },
    #     'importance_coeff': 1.0, # the trait value is multiplied by importance_coeff when calculating distance between genomes
    #     'mutation_prob': 0.1,
    #     'type': 'float'
    # }
}


# Neuron parameters
# -----------------
#   V_m: -70.0 # [mV], initial membrane potential
#   E_L: -70.0 # [mV], leak reversal
#   C_m: 300.0 # [pF]
#   t_ref: 3.0 # [ms]
#   tau_m: 10.0 # [ms]
#   V_th: -54.0
#   tau_syn_ex: 5.0
#   tau_syn_in: 5.0
#   I_e: 0.0 # external stimulation current
#   tau_minus: 34.0 # [ms], an STDP parameter. In NEST belongs to the neuron for technical reasons.

neuron_traits = {

    # V_m is always E_L, not to be mutable, so skipped there.

    'V_th': {
        'details': {
            'min': 1.0,  # 40
            'max': 10.0,  # 10
            'mut_power': 1.0,
            'mut_replace_prob': 0.25
        },
        'importance_coeff': 1.0,
        'mutation_prob': 0.1,
        'type': 'float'
    },

    'tau_minus': {
        'details': {
            'min': 1.0, # 40
            'max': 40.0, # 10
            'mut_power': 1.0,
            'mut_replace_prob': 0.25
        },
        'importance_coeff': 1.0,
        'mutation_prob': 0.1,
        'type': 'float'
    },

    # 'C_m': {
    #     'details': {
    #         'min': 1.0, # 40
    #         'max': 25.0, # 10
    #         'mut_power': 1.0,
    #         'mut_replace_prob': 0.25
    #     },
    #     'importance_coeff': 1.0,
    #     'mutation_prob': 0.1,
    #     'type': 'float'
    # },

    'tau_m': {
        'details': {
            'min': 1.0, # 40
            'max': 25.0, # 10
            'mut_power': 1.0,
            'mut_replace_prob': 0.25
        },
        'importance_coeff': 1.0,
        'mutation_prob': 0.1,
        'type': 'float'
    },

    't_ref': {
        'details': {
            'min': 1.0, # 40
            'max': 25.0, # 10
            'mut_power': 1.0,
            'mut_replace_prob': 0.25
        },
        'importance_coeff': 1.0,
        'mutation_prob': 0.1,
        'type': 'float'
    },

}


# hidden_neuron_traits = {

#     # V_m is always E_L, not to be mutable, so skipped there.


#     'tau_minus': {
#         'details': {
#             'min': 1.0, # 40
#             'max': 50.0, # 10
#             'mut_power': 1.0,
#             'mut_replace_prob': 0.25
#         },
#         'importance_coeff': 1.0,
#         'mutation_prob': 0.1,
#         'type': 'float'
#     },

#     'C_m': {
#         'details': {
#             'min': 1.0, # 40
#             'max': 5.0, # 10
#             'mut_power': 0.2,
#             'mut_replace_prob': 0.25
#         },
#         'importance_coeff': 1.0,
#         'mutation_prob': 0.1,
#         'type': 'float'
#     },

#     'tau_m': {
#         'details': {
#             'min': 1.0, # 40
#             'max': 5.0, # 10
#             'mut_power': 0.2,
#             'mut_replace_prob': 0.25
#         },
#         'importance_coeff': 1.0,
#         'mutation_prob': 0.1,
#         'type': 'float'
#     },

#     't_ref': {
#         'details': {
#             'min': 1.0, # 40
#             'max': 10.0, # 10
#             'mut_power': 0.5,
#             'mut_replace_prob': 0.25
#         },
#         'importance_coeff': 1.0,
#         'mutation_prob': 0.1,
#         'type': 'float'
#     },

# }


# Synapse parameters
# ------------------
#   Wmax
#   lambda
#   alpha
#   mu_plus
#   mu_minus
#   tau_plus
#   model

synapse_traits = {
    
    # Wmax is always 1, not to be mutable, so skipped there.

    # 'alpha': {
    #   'details': {
    #       'min': 0.3, #0.4,
    #       'max': 1.5, #1.3,
    #       'mut_power': 0.1,
    #       'mut_replace_prob': 0.25
    #   },
    #   'importance_coeff': 1.0,
    #   'mutation_prob': 0.1,
    #   'type': 'float'
    # },
    #
    # 'lambda': {
    #   'details': {
    #       'min': 0.05,
    #       'max': 0.1,
    #       'mut_power': 0.01, # the highest possible amount of change during one mutation
    #       'mut_replace_prob': 0.25 # if mutated, the probability to be re-initialized to a random value, rather than changed by a random value proportional to mut_power
    #   },
    #   'importance_coeff': 1.0, # the trait value is multiplied by importance_coeff when calculating distance between genomes
    #   'mutation_prob': 0.1,
    #   'type': 'float'
    # },

    # 'mu_plus': {
    #     'details': {
    #         'min': 0.0,
    #         'max': 1.0,
    #         'mut_power': 0.1,
    #         'mut_replace_prob': 0.25
    #     },
    #     'importance_coeff': 1.0,
    #     'mutation_prob': 0.1,
    #     'type': 'float'
    # },

    # 'mu_minus': {
    #     'details': {
    #         'min': 0.0,
    #         'max': 1.0,
    #         'mut_power': 0.1,
    #         'mut_replace_prob': 0.25
    #     },
    #     'importance_coeff': 1.0,
    #     'mutation_prob': 0.1,
    #     'type': 'float'
    # },

    'tau_plus': {
      'details': {
          'min': 1.0, #40.0,
          'max': 25.0, #10.0,
          'mut_power': 1.0,
          'mut_replace_prob': 0.25
      },
      'importance_coeff': 1.0,
      'mutation_prob': 0.1,
      'type': 'float'
    },

    # 'model': {
    #   'details': {
    #       'set': ['stdp_synapse'],
    #       'probs': [1.0], # if mutated, the new value is chosen by a roulette according to these probs
    #       'mut_power': 0.001,
    #       'mut_replace_prob': 0.25
    #   },
    #   'importance_coeff': 0,
    #   'mutation_prob': 0,
    #   'type': 'str'
    # }
}



# hidden_synapse_traits = {
    
#     # Wmax is always 1, not to be mutable, so skipped there.
    
#     'lambda': {
#       'details': {
#           'min': 0.05,
#           'max': 0.1,
#           'mut_power': 0.001, # the highest possible amount of change during one mutation
#           'mut_replace_prob': 0.25 # if mutated, the probability to be re-initialized to a random value, rather than changed by a random value proportional to mut_power
#       },
#       'importance_coeff': 1.0, # the trait value is multiplied by importance_coeff when calculating distance between genomes
#       'mutation_prob': 0.1,
#       'type': 'float'
#     },

#     'alpha': {
#       'details': {
#           'min': 0.3, #0.4,
#           'max': 1.5, #1.3,
#           'mut_power': 0.001,
#           'mut_replace_prob': 0.25
#       },
#       'importance_coeff': 1.0,
#       'mutation_prob': 0.1,
#       'type': 'float'
#     },

#     # 'mu_plus': {
#     #     'details': {
#     #         'min': 0.0,
#     #         'max': 1.0,
#     #         'mut_power': 0.1,
#     #         'mut_replace_prob': 0.25
#     #     },
#     #     'importance_coeff': 1.0,
#     #     'mutation_prob': 0.1,
#     #     'type': 'float'
#     # },

#     # 'mu_minus': {
#     #     'details': {
#     #         'min': 0.0,
#     #         'max': 1.0,
#     #         'mut_power': 0.1,
#     #         'mut_replace_prob': 0.25
#     #     },
#     #     'importance_coeff': 1.0,
#     #     'mutation_prob': 0.1,
#     #     'type': 'float'
#     # },

#     'tau_plus': {
#       'details': {
#           'min': 1.0, #40.0,
#           'max': 50.0, #10.0,
#           'mut_power': 1.0,
#           'mut_replace_prob': 0.25
#       },
#       'importance_coeff': 1.0,
#       'mutation_prob': 0.1,
#       'type': 'float'
#     },

#     # 'model': {
#     #   'details': {
#     #       'set': ['stdp_synapse'],
#     #       'probs': [1.0], # if mutated, the new value is chosen by a roulette according to these probs
#     #       'mut_power': 0.001,
#     #       'mut_replace_prob': 0.25
#     #   },
#     #   'importance_coeff': 0,
#     #   'mutation_prob': 0,
#     #   'type': 'str'
#     # }
# }


def write_parameters_file(filename, trait_values):
    f = open(filename, 'w')
    for trait_name, trait_value in trait_values.items():
        f.write(str(trait_name) + ': ' + str(trait_value) + '\n')

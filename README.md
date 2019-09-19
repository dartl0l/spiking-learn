# spiking_network_learning_algorithm
This is "framework" for spiking neural networks machine learning based on temporal encoding using Fisher' Iris, Wisconsin Breast Bancer and skelarn Digits datasets.
You can optimize model parameters using MultiNEAT neuroevolution library.
It can be running with mpi.

Dependencies:
  * Python3
  * NEST Simulator https://www.nest-simulator.org/
  * sklearn
  * numpy
  * matplotlib
  * mpi4py
  * MultiNEAT http://multineat.com/index.html (needed for genetic.py)

To start simulation create your own py file and add

```
from spiking_network_learning_alghorithm.solver import solve_task

solve_task("./")
```


or run from command line

```
python solver path-to-folder-with-settings-file
```

here is example of settings file for Fisher's Iris Classification settings.json

```
{
    "network": {
        "num_threads": 48,
        "test_with_noise": false,
        "save_history": false,
        "h": 0.01,
        "noise_after_pattern": true,
        "noise_freq": 1.8621152583509684,
        "num_procs": 1,
        "h_time": 25.0,
        "start_delta": 50,
        "separate_networks": false
    },
    "topology": {
        "n_input": 80,
        "use_reciprocal": false,
        "two_layers": false,
        "n_layer_out": 1,
        "use_inhibition": false,
        "n_layer_hid": 100
    },
    "learning": {
        "reinforce_delta": 0.0,
        "teacher_amplitude": 1500000.0,
        "use_teacher": false,
        "epochs": 10,
        "use_fitness_func": true,
        "reinforce_time": 0.0,
        "n_splits": 5,
        "fitness_func": "f1",
        "metrics": "f1"
    },
    "data": {
        "use_valid": true,
        "n_coding_neurons": 20,
        "normalization": "normalize minmax",
        "valid_size": 0.05,
        "coding_sigma": 0.005,
        "dataset": "iris",
        "preprocessing": ""
    },
    "model": {
        "neuron_out": {
            "I_e": 0.0,
            "V_th": 1.0,
            "tau_minus": 28.801956753013656,
            "E_L": 0.0,
            "tau_m": 6.219211603049189,
            "C_m": 3.3689216719940305,
            "t_ref": 4.660600075731054,
            "V_m": -5.0,
            "V_reset": -5.0
        },
        "syn_dict_stdp": {
            "model": "stdp_synapse",
            "alpha": 0.6104427804239094,
            "tau_plus": 14.757516889367253,
            "lambda": 0.09483805783092976,
            "mu_plus": 0.0,
            "weight": {
                "distribution": "normal",
                "sigma": 0.0,
                "mu": 1.0
            },
            "Wmax": {
                "distribution": "normal",
                "sigma": 0.0,
                "mu": 1.0
            },
            "mu_minus": 0.0
        },
        "syn_dict_stdp_hid": {
            "model": "stdp_synapse",
            "alpha": 0.85,
            "tau_plus": 10.429564842488617,
            "lambda": 0.03,
            "mu_plus": 0.0,
            "weight": {
                "distribution": "normal",
                "sigma": 0.0,
                "mu": 1.0
            },
            "Wmax": {
                "distribution": "normal",
                "sigma": 0.0,
                "mu": 1.0
            },
            "mu_minus": 0.0
        },
        "neuron_hid": {
            "I_e": 0.0,
            "V_th": 1.0,
            "tau_minus": 33.7,
            "E_L": 0.0,
            "tau_m": 10.0,
            "C_m": 10.0,
            "t_ref": 3.0,
            "V_m": -5.0,
            "V_reset": -5.0
        },
        "syn_dict_inh": {
            "weight": -15.0,
            "model": "static_synapse"
        }
    }
}

```


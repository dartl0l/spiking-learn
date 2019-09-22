# spiking_network_learning_algorithm
This is "framework" for spiking neural networks machine learning based on temporal encoding using Fisher' Iris, Wisconsin Breast Bancer and skelarn Digits datasets.
You can optimize model parameters using MultiNEAT neuroevolution library.
It can be running with mpi.

### Dependencies
  * Python3
  * NEST Simulator https://www.nest-simulator.org/
  * sklearn
  * numpy
  * matplotlib
  * mpi4py
  * MultiNEAT http://multineat.com/index.html (needed for genetic.py)

## Running
To start simulation create your own py file and add

```
from spiking_network_learning_alghorithm.solver import solve_task

solve_task(path-to-folder-with-settings-file)
```


or run from command line

```
python solver path-to-folder-with-settings-file
```

here is example of settings file for Fisher's Iris Classification settings.json 

```
{
    "model": {
        "neuron_out": {
            "V_reset": -5.0,
            "E_L": 0.0,
            "I_e": 0.0,
            "C_m": 2.3041116651147604,
            "V_m": -5.0,
            "t_ref": 1.4445230588316917,
            "V_th": 1.0,
            "tau_m": 5.103678226470947,
            "tau_minus": 31.954210396157578
        },
        "syn_dict_inh": {
            "weight": -15.0,
            "model": "static_synapse"
        },
        "syn_dict_stdp_hid": {
            "weight": {
                "sigma": 0.0,
                "mu": 1.0,
                "distribution": "normal"
            },
            "mu_plus": 0.0,
            "lambda": 0.03,
            "tau_plus": 10.429564842488617,
            "mu_minus": 0.0,
            "model": "stdp_synapse",
            "Wmax": {
                "sigma": 0.0,
                "mu": 1.0,
                "distribution": "normal"
            },
            "alpha": 0.85
        },
        "neuron_hid": {
            "V_reset": -5.0,
            "E_L": 0.0,
            "I_e": 0.0,
            "C_m": 10.0,
            "V_m": -5.0,
            "t_ref": 3.0,
            "V_th": 1.0,
            "tau_m": 10.0,
            "tau_minus": 33.7
        },
        "syn_dict_stdp": {
            "weight": {
                "sigma": 0.0,
                "mu": 1.0,
                "distribution": "normal"
            },
            "mu_plus": 0.0,
            "lambda": 0.05999982627108694,
            "tau_plus": 12.122823428362608,
            "mu_minus": 0.0,
            "model": "stdp_synapse",
            "Wmax": {
                "sigma": 0.0,
                "mu": 1.0,
                "distribution": "normal"
            },
            "alpha": 0.7067821397446096
        }
    },
    "learning": {
        "n_splits": 5,
        "fitness_func": "f1",
        "use_teacher": true,
        "reinforce_delta": 0.0,
        "use_fitness_func": true,
        "teacher_amplitude": 1500000.0,
        "epochs": 10,
        "reinforce_time": 0.0,
        "metrics": "f1"
    },
    "data": {
        "coding_sigma": 0.005,
        "shuffle_train": true,
        "n_coding_neurons": 20,
        "normalization": "normalize",
        "valid_size": 0.1,
        "dataset": "iris",
        "preprocessing": "",
        "use_valid": false
    },
    "network": {
        "num_threads": 48,
        "noise_after_pattern": false,
        "h_time": 25.0,
        "noise_freq": 1.7179610719904304,
        "test_with_noise": false,
        "num_procs": 1,
        "h": 0.01,
        "separate_networks": false,
        "save_history": false,
        "start_delta": 50
    },
    "topology": {
        "use_reciprocal": false,
        "use_inhibition": true,
        "two_layers": false,
        "n_layer_hid": 100,
        "n_layer_out": 3,
        "n_input": 80
    }
}

```


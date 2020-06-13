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
from spiking_network_learning_alghorithm.solver_new import solve_task

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
            "V_reset": 0.0,
            "E_L": 0.0,
            "I_e": 0.0,
            "C_m": 1.0,
            "V_m": 0.0,
            "t_ref": 19.0,
            "V_th": 8.0,
            "tau_m": 6.0,
            "tau_minus": 31.0
        },
        "syn_dict_inh": {
            "weight": -5,
            "model": "static_synapse"
        },
        "syn_dict_stdp": {
            "weight": {
                "sigma": 0.0,
                "mu": 1.0,
                "distribution": "normal"
            },
            "mu_plus": 0.0,
            "lambda": 0.03,
            "tau_plus": 6.0,
            "mu_minus": 0.0,
            "model": "stdp_synapse",
            "Wmax": {
                "sigma": 0.0,
                "mu": 1.0,
                "distribution": "normal"
            },
            "alpha": 0.65
        },
        "neuron_out_model": "iaf_psc_exp"
    },
    "learning": {
        "n_splits": 5,
        "fitness_func": "f1",
        "use_teacher": true,
        "reinforce_delta": 0.0,
        "use_fitness_func": true,
        "teacher_amplitude": 1000.0,
        "epochs": 5,
        "reinforce_time": 0.0,
        "metrics": "f1",
        "inhibitory_teacher": false
    },
    "data": {
        "coding_sigma": 0.005,
        "shuffle_train": true,
        "n_coding_neurons": 40,
        "normalization": "normalize",
        "valid_size": 0.1,
        "dataset": "iris",
        "preprocessing": "",
        "use_valid": false,
        "shuffle_test": true,
        "frequency_coding": false,
        "conversion": "receptive_fields"
    },
    "network": {
        "num_threads": 48,
        "noise_after_pattern": false,
        "h_time": 25.0,
        "noise_freq": 0.0,
        "test_with_noise": false,
        "num_procs": 1,
        "h": 0.01,
        "separate_networks": false,
        "save_history": false,
        "start_delta": 50,
        "test_with_inhibition": true
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


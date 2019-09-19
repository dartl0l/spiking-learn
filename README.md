# spiking_network_learning_algorithm
This is "framework" for spiking neural networks machine learning based on temporal encoding using Fisher' Iris, Wisconsin Breast Bancer and skelarn Digits datasets.
You can optimize model parameters using MultiNEAT neuroevolution library.
It can be running with mpi.

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

```

Dependencies:
  * Python3
  * NEST Simulator https://www.nest-simulator.org/
  * sklearn
  * numpy
  * matplotlib
  * mpi4py
  * MultiNEAT http://multineat.com/index.html (if needed)

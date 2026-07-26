# spiking-learn

This is a framework for spiking neural network machine learning based on temporal encoding.

### Citation

Please cite my latest paper when using this repository:
https://scholar.google.com/citations?user=wiwYfQMAAAAJ&hl=en

### Dependencies

- Python 3
- [NEST Simulator](https://www.nest-simulator.org/)
- scikit-learn
- NumPy
- Matplotlib
- Hyperopt
- tqdm

## Examples

The examples require NEST and the custom `stdptanhmodule`.

```bash
python examples/iris_supervised.py
python examples/iris_unsupervised.py
```

- [`iris_supervised.py`](examples/iris_supervised.py) demonstrates supervised STDP
  classification with an sklearn pipeline.
- [`iris_unsupervised.py`](examples/iris_unsupervised.py) demonstrates unsupervised STDP
  feature learning followed by first-spike voting.

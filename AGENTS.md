# spiking-learn (`spilearn`)

Framework for machine learning with **spiking neural networks** based on **temporal / time-to-first-spike (TTFS) coding**, built on top of the [NEST simulator](https://www.nest-simulator.org/). The public API is exposed as **scikit-learn-compatible estimators** (`fit` / `predict` / `transform`), so networks compose into `sklearn` pipelines and cross-validation.

## Critical environment fact

NEST is not available in every development environment. NEST-dependent code must be tested
in an environment that provides NEST and the required custom modules.

Consequences for working here:
- **Do not assume you can import `spilearn` or run the networks locally.** Anything that imports `spilearn.network` triggers `import nest` and `nest.Install('stdptanhmodule')` (see `spilearn/network/base.py`) and will fail.
- Pure modules (`estimators/encoders/*`, parts of `utils.py`) only need `numpy`/`sklearn` and can run anywhere those are installed.
- Don't claim NEST-dependent code runs/passes unless it was actually executed in a NEST-capable environment.
- Requires a **custom NEST module `stdptanhmodule`** (provides `stdp_synapse` tanh variant). It must be built/installed on the target machine.

## Package layout (`spilearn/`)

Two layers: low-level **NEST networks** and the sklearn **estimator** wrappers around them.

- `network/` — NEST network implementations. `base.py:Network` is the base class: builds the NEST kernel, layers (`parrot_neuron` input → output neurons), devices (spike recorders, multimeter), connections, and implements `train()` / `test()`. Subclasses: `EpochNetwork` (epoch-based supervised), `TwoLayerNetwork`, `MultilayerRLNetwork`, `ContrastiveMultilayerRLNetwork`, `AdaptiveMultilayerRLNetwork`, `LiteRlNetwork`, `LiteRlPoolNetwork`, `ConvolutionNetwork`, `ConvolutionRlNetwork`, `CriticNetwork`.
- `estimators/` — sklearn-facing API (`spilearn.estimators`):
  - `encoders/` — feature → spike-latency transformers: `DoGTransformer`, `ReceptiveFieldsTransformer`, `TemporalPatternTransformer`, `TemporalPatternReversedTransformer`.
  - `decoders/` — spike output → labels: `FirstSpikeVotingClassifier`.
  - `networks/` — classifiers/transformers wrapping a `Network`. `base.py:BaseTemporalEstimator` ties an estimator to a `Network` + `TemporalSpikeGenerator`; `fit` calls `network.train`, `predict`/`transform` call `network.test` then decode latencies. Supervised variants add a `Teacher`. Includes `SupervisedTemporalClassifier`, `*RL*`, `*Pool*`, `*Reservoir*`, `*Multilayer*`, `Unsupervised*` transformers.
- `spike_generator.py` — `TemporalSpikeGenerator`: turns latency-encoded data into NEST spike-generator input.
- `teacher.py` — `Teacher`: supervisory teacher spikes for supervised STDP.
- `noise.py` — `NoiseGenerator`.
- `converter.py` — data ↔ spike-time conversions.
- `simulation.py` — `Simulation` / `ReinforcementSimulation` (RL loop with `gym`).
- `reward.py`, `genetic.py` (hyperopt-style search), `evaluation.py`, `plotter.py`, `andrews_curve.py` — helpers, metrics, plotting.

Exports are curated in the `__init__.py` of `spilearn`, `spilearn.estimators`, and `spilearn.network` — **add new public classes to the relevant `__init__.py` and `__all__`**.

## Config conventions

- Estimators and networks take a legacy `settings` dict **plus** a `model` dict (NEST neuron/synapse parameters).
- `settings` is **deprecated as the source of truth**: estimators now accept explicit keyword args (`n_input`, `n_layer_out`, `epochs`, `h_time`, `start_delta`, `h`, `teacher_amplitude`, …) that **override** the corresponding `settings`/`model` values via the `kwarg or settings[...].get(...)` pattern. Prefer explicit kwargs in new code; keep the `settings` fallback working.
- The `model` dict keys (e.g. `neuron_out`, `syn_dict_exc`, `syn_dict_inh`, `syn_dict_stdp`, `*_model`) map directly onto NEST `Create`/`SetStatus` calls.
- Public usage examples live in `examples/`; keep cluster-specific launchers and search
  configurations outside the library repository.

## Conventions

- Python 3, files start with `# coding: utf-8`.
- 4-space indent; type hints with `Optional[...]` on constructor args; deep-copy mutable `model` before mutating NEST params.
- Version lives in `spilearn/__init__.py` (`__version__`); `setup.py` reads it.

## Gaps / watch-outs

- NEST and `stdptanhmodule` must be installed separately; Python dependencies are listed in
  `requirements.txt`.
- Tests are lightweight standalone Python files under `tests/`; run NEST-dependent tests in
  a NEST-capable environment.
- `tasks/` is git-ignored (scratch task specs, e.g. adaptive-threshold work).

import os
import sys
import copy
import json
import math
import pickle

import numpy as np

from struct import unpack

from scipy.stats import mode

from sklearn.model_selection import (
    cross_val_score,
    RepeatedStratifiedKFold,
    train_test_split,
)
from sklearn.metrics import f1_score, accuracy_score

from ray import train, tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def get_labeled_data(picklename, MNIST_data_path, bTrain=True):
    """Read input-vector (image) and target class (label, 0-9) and return
    it as list of tuples.
    """
    picklename = MNIST_data_path + picklename
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename, 'rb'))
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(MNIST_data_path + 'train-images.idx3-ubyte', 'rb')
            labels = open(MNIST_data_path + 'train-labels.idx1-ubyte', 'rb')
        else:
            images = open(MNIST_data_path + 't10k-images.idx3-ubyte', 'rb')
            labels = open(MNIST_data_path + 't10k-labels.idx1-ubyte', 'rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]
        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        for i in range(N):
            if i % 1000 == 0:
                print('i: %i' % i)
            x[i] = [
                [unpack('>B', images.read(1))[0] for unused_col in range(cols)]
                for unused_row in range(rows)
            ]
            y[i] = unpack('>B', labels.read(1))[0]
        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open('%s.pickle' % picklename, 'wb'))
    return data


def run_train_test(
    params,
    X,
    y,
    model_in,
    create_pipe,
    X_test=None,
    y_test=None,
    **_,
):
    model = copy.deepcopy(model_in)

    try:
        pipe = create_pipe(model, params)

        pipe.fit(X, y)
        y_test_pred = pipe.predict(X_test)
        train.report({'metric': accuracy_score(y_test, y_test_pred)})
    except ValueError as exc:
        if 'All-NaN slice encountered' not in str(exc):
            raise
        train.report({'metric': 0.0})


def run_cv(
    params,
    X,
    y,
    model_in,
    create_pipe,
    X_test=None,
    y_test=None,
    subset_fractions=(0.1, 0.3, 1.0),
    n_splits=5,
    n_repeats_final=3,
    n_jobs=5,
    base_seed=42,
):
    """Multi-fidelity CV objective for ASHA, with subset-size as the fidelity.

    Evaluates the config on an increasing stratified subset of the data
    (``subset_fractions``) and ``train.report``s after each rung, so ASHA can
    kill weak configs cheaply (most die on the smallest subset). The final
    (full-data) rung uses ``RepeatedStratifiedKFold`` (``n_repeats_final``) to
    average over several splits — de-biasing the score so the chosen config
    generalizes (closer to a full-dataset run) instead of overfitting one split.

    Each rung calls ``cross_val_score(n_jobs=...)`` as before, so per-rung fold
    parallelism is preserved; reporting happens between rungs.
    """
    model = copy.deepcopy(model_in)

    try:
        pipe = create_pipe(model, params)
    except Exception:
        train.report(
            {"metric": 0.0, "mean_accuracy": 0.0, "std_accuracy": 0.0,
             "subset_fraction": 1.0}
        )
        return

    X = np.asarray(X)
    y = np.asarray(y)
    n_rungs = len(subset_fractions)

    for i, frac in enumerate(subset_fractions):
        is_final = (i == n_rungs - 1) or (frac >= 1.0)
        if frac >= 1.0:
            Xs, ys = X, y
        else:
            Xs, _, ys, _ = train_test_split(
                X, y, train_size=frac, stratify=y, random_state=base_seed
            )
        n_rep = n_repeats_final if is_final else 1
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_rep, random_state=base_seed
        )
        try:
            scores = cross_val_score(
                pipe, Xs, ys, cv=cv, scoring='accuracy', n_jobs=n_jobs
            )
            m, s = float(scores.mean()), float(scores.std())
        except ValueError as exc:
            if 'All-NaN slice encountered' not in str(exc):
                raise
            m, s = 0.0, 0.0
        train.report(
            {
                "metric": m - 0.5 * s,
                "mean_accuracy": m,
                "std_accuracy": s,
                "subset_fraction": float(frac),
            }
        )


def optimize_ray(
    X,
    y,
    func,
    create_pipe,
    space,
    model,
    exp_name,
    new_trial=True,
    max_evals=100,
    X_test=None,
    y_test=None,
    max_concurrent=11,
    subset_fractions=(0.1, 0.3, 1.0),
    n_splits=5,
    n_repeats_final=3,
    n_jobs=5,
):
    ray_path = os.path.expanduser('~/ray_results')
    exp_dir = os.path.join(ray_path, exp_name)

    trainable = tune.with_parameters(
        func,
        X=X,
        y=y,
        model_in=model,
        create_pipe=create_pipe,
        X_test=X_test,
        y_test=y_test,
        subset_fractions=subset_fractions,
        n_splits=n_splits,
        n_repeats_final=n_repeats_final,
        n_jobs=n_jobs,
    )

    if tune.Tuner.can_restore(exp_dir) and not new_trial:
        tuner = tune.Tuner.restore(
            exp_dir,
            trainable=trainable,
            resume_errored=True,
        )
    else:
        search_alg = OptunaSearch(seed=42)

        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_concurrent)
        tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(
                num_samples=max_evals,
                search_alg=search_alg,
                # subset-size fidelity: one report per rung, so ASHA prunes on
                # the cheap (small-subset) rungs and only survivors hit full data
                scheduler=ASHAScheduler(
                    time_attr='training_iteration',
                    max_t=len(subset_fractions),
                    grace_period=1,
                    reduction_factor=3,
                ),
                metric='metric',
                mode='max',
            ),
            run_config=train.RunConfig(
                name=exp_name,
            ),
            param_space=space,
        )
    return tuner.fit()


def run_experiment(
    exp_name_prefix,
    space,
    create_pipe,
    model,
    X,
    y,
    argv=None,
    default_max_evals=200,
    select_metric='mean_accuracy',
    select_mode='max',
    subset_fractions=(0.1, 0.3, 1.0),
    n_splits=5,
    n_repeats_final=3,
    n_jobs=5,
):
    """Shared launcher for Ray Tune experiments.

    An experiment script only defines `space`, `create_pipe`, its `model` dict and
    the data, then calls this. All the common boilerplate lives here instead of
    being copy-pasted per script:
      - parse CLI args, run/resume the sweep via ``optimize_ray``,
      - dump ``<path>/model.json`` and ``<path>/best_space.json``,
      - print the best config + score.

    CLI args (passed by the sbatch script):
        <path> <new_trial 0|1> [max_evals] [synapse_model]
      path          -> exp_name = exp_name_prefix + path; outputs go to ./<path>/
      new_trial     -> 1 = fresh sweep, 0 = resume
      synapse_model -> optional override of model['syn_dict_exc']['synapse_model']
    """
    argv = sys.argv if argv is None else argv
    path = argv[1]
    new_trial = bool(int(argv[2]))
    max_evals = int(argv[3]) if len(argv) > 3 else default_max_evals
    synapse_model = argv[4] if len(argv) > 4 else None

    if synapse_model:
        model['syn_dict_exc']['synapse_model'] = synapse_model

    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'model.json'), 'w') as fh:
        json.dump(model, fh, indent=4)

    exp_name = exp_name_prefix + path
    results = optimize_ray(
        X=X,
        y=y,
        func=run_cv,
        create_pipe=create_pipe,
        space=space,
        model=model,
        exp_name=exp_name,
        new_trial=new_trial,
        max_evals=max_evals,
        subset_fractions=subset_fractions,
        n_splits=n_splits,
        n_repeats_final=n_repeats_final,
        n_jobs=n_jobs,
    )

    # Select the best config among trials that reached the FULL-DATA rung, so the
    # reported best is a real full-dataset score and not a small-subset fluke from
    # a trial ASHA pruned early.
    df = results.get_dataframe()
    final_frac = max(subset_fractions)
    full = df
    if 'subset_fraction' in df.columns:
        reached = df[df['subset_fraction'] >= final_frac]
        if len(reached):
            full = reached
    if select_mode == 'min':
        best_row = full.loc[full[select_metric].idxmin()]
    else:
        best_row = full.loc[full[select_metric].idxmax()]
    best_params = {
        c[len('config/'):]: best_row[c]
        for c in best_row.index
        if c.startswith('config/')
    }
    print('best', select_metric, '=', best_row[select_metric])
    print(best_params)
    with open(os.path.join(path, 'best_space.json'), 'w') as fh:
        json.dump(best_params, fh, indent=4, cls=NpEncoder)
    return results


def round_decimals(value):
    i = 0
    while value < 1:
        value *= 10
        i += 1
    return i


def print_settings(settings):
    for key in settings:
        if isinstance(settings[key], dict):
            print(key)
            for parameter in settings[key]:
                if isinstance(settings[key][parameter], dict):
                    print('\t' + parameter)
                    for parameter2 in settings[key][parameter]:
                        print(
                            '\t\t'
                            + parameter2
                            + ' : '
                            + str(settings[key][parameter][parameter2])
                        )
                else:
                    print('\t' + parameter + ' : ' + str(settings[key][parameter]))
        else:
            print('\t' + key + ' : ' + str(settings[key]))


def split_spikes_and_senders(input_latency, n_examples, start_delta, h_time):
    output_latency = []
    d_time = start_delta
    input_latency['spikes'] = np.array(input_latency['spikes'])
    input_latency['senders'] = np.array(input_latency['senders'])
    for _ in range(n_examples):
        mask = (input_latency['spikes'] > d_time) & (
            input_latency['spikes'] < d_time + h_time
        )
        spikes_tmp = input_latency['spikes'][mask]
        senders_tmp = input_latency['senders'][mask]
        tmp_dict = {'spikes': spikes_tmp - d_time, 'senders': senders_tmp}

        d_time += h_time
        output_latency.append(tmp_dict)
    return output_latency


def convert_latency(latency_list, n_neurons, offset=1):
    output_array = []
    for latencies in latency_list:
        tmp_list = [np.nan] * n_neurons
        senders_array = np.array(latencies['senders'])
        spikes_array = np.array(latencies['spikes'])
        senders = set(senders_array)
        for sender in senders:
            mask = senders_array == sender
            tmp_list[int(sender) - offset] = spikes_array[mask][0]
        output_array.append(tmp_list)
    return output_array


def convert_latency_pool_mean(latency_list, n_neurons, pool_size):
    output_array = convert_latency(latency_list, n_neurons)
    output_array = np.array(output_array).reshape(
        len(output_array), int(n_neurons / pool_size), pool_size
    )
    output_array = np.mean(output_array, axis=2)
    return output_array


def predict_from_latency(latency_list, func=np.nanargmin):
    latency_list = np.array(latency_list)
    mask = np.logical_not(np.all(np.isnan(latency_list), axis=1))
    prediction = np.zeros(len(latency_list))
    prediction[mask] = func(latency_list[mask], axis=1)
    return prediction


def convert_latency_pool(latency_list, n_neurons, pool_size):
    output_array = convert_latency(latency_list, n_neurons)
    output_array = np.array(output_array).reshape(
        len(output_array), int(n_neurons / pool_size), pool_size
    )
    return output_array


def predict_from_latency_pool(latency_list, func=np.nanargmin):
    latency_list = np.array(latency_list)
    mask = np.isnan(latency_list)
    mask = np.all(mask, axis=1)
    if np.any(mask):
        prediction = np.full(len(latency_list), -1, dtype=int)
        valid = np.logical_not(mask).all(axis=1)
        if np.any(valid):
            predictions = func(latency_list[valid], axis=1)
            prediction[valid] = mode(predictions, axis=1).mode.flatten()
        return prediction

    predictions = func(latency_list, axis=1)
    prediction = mode(predictions, axis=1).mode.flatten()
    return prediction


def predict_from_earliest_latency_pool(latency_list, func=np.argmin):
    latency_list = np.array(latency_list)
    class_latency = np.min(
        np.where(np.isnan(latency_list), np.inf, latency_list),
        axis=2,
    )
    valid = np.isfinite(class_latency).any(axis=1)
    prediction = np.full(len(latency_list), -1, dtype=int)
    prediction[valid] = func(class_latency[valid], axis=1)
    return prediction


def convert_layer_weights(weights):
    out_weights = {}
    for neuron_id in weights:
        out_weights[neuron_id] = np.zeros(len(weights[neuron_id]))
        for i, input_id in enumerate(weights[neuron_id]):
            out_weights[neuron_id][i] = weights[neuron_id][input_id]
    return out_weights


def convert_weights(weights):
    out_weights = []
    for layer_weights in weights:
        out_weights.append(convert_layer_weights(layer_weights))
    return out_weights


def fitness_func_time(latency_list, Y):
    fit_list = []

    for latency, y in zip(latency_list, Y):
        latency_of_desired_neuron = latency.pop(y)

        fit = -1 * latency_of_desired_neuron
        fit_list.append(fit)

    fitness_score = np.mean(fit_list)
    if np.isnan(fitness_score):
        fitness_score = 0
    return fitness_score


def fitness_func_sigma(latency_list, Y):
    def sigmoid(x, alpha):
        return 1 / (1 + np.exp(-2 * alpha * x))

    fit_list = []
    for latency, y in zip(latency_list, Y):
        latency_of_desired_neuron = latency.pop(y)
        fit = 1
        for lat in latency:
            fit *= sigmoid(lat - latency_of_desired_neuron, 0.1)
        fit_list.append(fit)
    fitness_score = np.mean(fit_list)
    if np.isnan(fitness_score):
        fitness_score = 0
    return fitness_score, fit_list


def fitness_func_exp(latency_list, Y):
    fit_list = []
    for latency, y in zip(latency_list, Y):
        latency_of_desired_neuron = latency.pop(y)
        fit = 1
        for lat in latency:
            fit -= math.exp(latency_of_desired_neuron - lat)
        fit_list.append(fit)
    fitness_score = np.mean(fit_list)
    if np.isnan(fitness_score):
        fitness_score = 0
    return fitness_score, fit_list

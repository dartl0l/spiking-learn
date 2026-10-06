import copy
import json
import os
import sys
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    cross_val_score,
    train_test_split,
)
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

from .spec import Model, Params, PathInput, RunMode, SearchPipeFactory, SelectMode


class NpEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def run_train_test(
    params: Params,
    X: Any,
    y: Any,
    model_in: Model,
    create_pipe: SearchPipeFactory,
    X_test: Optional[Any] = None,
    y_test: Optional[Any] = None,
    **_: Any,
) -> None:
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


def run_cv(params: Params, X: Any, y: Any, model_in: Model, create_pipe: SearchPipeFactory,
           X_test: Optional[Any] = None, y_test: Optional[Any] = None,
           subset_fractions: Tuple[float, ...] = (0.1, 0.3, 1.0),
           n_splits: int = 5, n_repeats_final: int = 3, n_jobs: int = 5,
           base_seed: int = 42, groups: Optional[Any] = None,
           scorer: Union[str, Callable[..., float]] = 'accuracy') -> None:
    """Multi-fidelity CV objective for ASHA, with subset-size as the fidelity.

    Evaluates the config on an increasing stratified subset of the data
    (``subset_fractions``) and ``train.report``s after each rung, so ASHA can
    kill weak configs cheaply (most die on the smallest subset). The final
    (full-data) rung uses ``RepeatedStratifiedKFold`` (``n_repeats_final``) to
    average over several splits — de-biasing the score so the chosen config
    generalizes (closer to a full-dataset run) instead of overfitting one split.

    Each rung calls ``cross_val_score(n_jobs=...)`` as before, so per-rung fold
    parallelism is preserved; reporting happens between rungs.
    When ``groups`` are provided, folds use ``StratifiedGroupKFold`` instead.
    """
    model = copy.deepcopy(model_in)
    pipe = create_pipe(model, params)
    X = np.asarray(X)
    y = np.asarray(y)
    groups = None if groups is None else np.asarray(groups)
    n_rungs = len(subset_fractions)
    for i, frac in enumerate(subset_fractions):
        is_final = (i == n_rungs - 1) or (frac >= 1.0)
        if frac >= 1.0:
            Xs, ys, group_subset = X, y, groups
        else:
            indices, _ = train_test_split(
                np.arange(len(y)), train_size=frac, stratify=y,
                random_state=base_seed,
            )
            Xs, ys = X[indices], y[indices]
            group_subset = None if groups is None else groups[indices]
        n_rep = n_repeats_final if is_final else 1
        if group_subset is None:
            cv = RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=n_rep, random_state=base_seed,
            )
        else:
            cv = []
            for repeat in range(n_rep):
                splitter = StratifiedGroupKFold(
                    n_splits=n_splits, shuffle=True,
                    random_state=base_seed + repeat,
                )
                cv.extend(splitter.split(Xs, ys, group_subset))
        score_kwargs = {
            'cv': cv,
            'scoring': scorer,
            'n_jobs': n_jobs,
            'error_score': 'raise',
        }
        if group_subset is not None:
            score_kwargs['groups'] = group_subset
        scores = cross_val_score(pipe, Xs, ys, **score_kwargs)
        if not np.isfinite(scores).all():
            raise ValueError('scorer returned non-finite scores')
        m, s = float(scores.mean()), float(scores.std())
        report = {'metric': m - 0.5 * s, 'subset_fraction': float(frac)}
        report.update(mean_score=m, std_score=s)
        train.report(report)


def optimize_ray(X: Any, y: Any, func: Callable[..., None], create_pipe: SearchPipeFactory,
                 space: Params, model: Model, exp_name: str,
                 new_trial: bool = True, max_evals: int = 100,
                 X_test: Optional[Any] = None, y_test: Optional[Any] = None,
                 max_concurrent: int = 11,
                 subset_fractions: Tuple[float, ...] = (0.1, 0.3, 1.0),
                 n_splits: int = 5, n_repeats_final: int = 3,
                 n_jobs: int = 5, groups: Optional[Any] = None,
                 cpus_per_trial: Optional[int] = None,
                 storage_path: Optional[PathInput] = None,
                 scorer: Union[str, Callable[..., float]] = 'accuracy',
                 time_budget_s: Optional[float] = None) -> tune.ResultGrid:
    ray_path = (os.path.expanduser('~/ray_results') if storage_path is None
                else os.fspath(storage_path))
    exp_dir = os.path.join(ray_path, exp_name)
    trainable = tune.with_parameters(
        func, X=X, y=y, model_in=model, create_pipe=create_pipe,
        X_test=X_test, y_test=y_test, subset_fractions=subset_fractions,
        n_splits=n_splits, n_repeats_final=n_repeats_final, n_jobs=n_jobs,
        groups=groups, scorer=scorer,
    )
    if cpus_per_trial is not None:
        trainable = tune.with_resources(trainable, {'cpu': cpus_per_trial})
    if tune.Tuner.can_restore(exp_dir) and not new_trial:
        tuner = tune.Tuner.restore(
            exp_dir, trainable=trainable, resume_errored=True,
        )
    else:
        search_alg = OptunaSearch(seed=42)
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_concurrent)
        tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(
                num_samples=max_evals, search_alg=search_alg,
                scheduler=ASHAScheduler(
                    time_attr='training_iteration',
                    max_t=len(subset_fractions), grace_period=1,
                    reduction_factor=3,
                ),
                metric='metric', mode='max',
                time_budget_s=time_budget_s,
            ),
            run_config=train.RunConfig(name=exp_name, storage_path=ray_path),
            param_space=space,
        )
    return tuner.fit()


def _run_search_backend(*, exp_name_prefix: str, space: Params,
                        create_pipe: SearchPipeFactory, model: Model, X: Any, y: Any,
                        run_id: str, mode: RunMode, max_evals: int,
                        select_metric: str, select_mode: SelectMode,
                        subset_fractions: Tuple[float, ...], n_splits: int,
                        n_repeats_final: int, n_jobs: int,
                        groups: Optional[Any], max_concurrent: int,
                        cpus_per_trial: Optional[int] = None,
                        storage_path: Optional[PathInput] = None,
                        output_path: Optional[PathInput] = None,
                        scorer: Union[str, Callable[..., float]] = 'accuracy',
                        time_budget_s: Optional[float] = None) -> tune.ResultGrid:
    """Run the explicit experiments API without parsing process arguments."""
    output_path = os.fspath(output_path)
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'model.json'), 'w') as fh:
        json.dump(model, fh, indent=4, cls=NpEncoder)
    results = optimize_ray(
        X=X, y=y, func=run_cv, create_pipe=create_pipe, space=space,
        model=model, exp_name=exp_name_prefix + run_id,
        new_trial=mode == 'fresh', max_evals=max_evals,
        subset_fractions=subset_fractions, n_splits=n_splits,
        n_repeats_final=n_repeats_final, n_jobs=n_jobs, groups=groups,
        max_concurrent=max_concurrent, cpus_per_trial=cpus_per_trial,
        storage_path=storage_path, scorer=scorer,
        time_budget_s=time_budget_s,
    )
    aliases = {
        'mean_score': 'mean_accuracy',
        'std_score': 'std_accuracy',
        'mean_accuracy': 'mean_score',
        'std_accuracy': 'std_score',
    }

    def metric_value(result: Any) -> float:
        metrics = result.metrics
        if select_metric in metrics:
            return metrics[select_metric]
        return metrics.get(aliases.get(select_metric, ''), np.nan)

    final_frac = max(subset_fractions)
    valid = [
        result for result in results
        if getattr(result, 'error', None) is None
        and np.isfinite(metric_value(result))
    ]
    completed = [
        result for result in valid
        if result.metrics.get('subset_fraction', 0) >= final_frac
    ]
    pool = completed or valid
    if not pool:
        raise ValueError(f'no finite result for select_metric={select_metric!r}')
    best = (min if select_mode == 'min' else max)(pool, key=metric_value)
    params = best.config
    with open(os.path.join(output_path, 'best_space.json'), 'w') as fh:
        json.dump(params, fh, indent=4, cls=NpEncoder)
    return results


def run_experiment(exp_name_prefix: str, space: Params, create_pipe: SearchPipeFactory,
                   model: Model, X: Any, y: Any,
                   argv: Optional[Sequence[str]] = None,
                   default_max_evals: int = 200,
                   select_metric: str = 'mean_score',
                   select_mode: SelectMode = 'max',
                   subset_fractions: Tuple[float, ...] = (0.1, 0.3, 1.0),
                   n_splits: int = 5, n_repeats_final: int = 3,
                   n_jobs: int = 5, groups: Optional[Any] = None,
                   max_concurrent: int = 11,
                   storage_path: Optional[PathInput] = None,
                   output_path: Optional[PathInput] = None) -> tune.ResultGrid:
    """Shared launcher for Ray Tune experiments.

    An experiment script only defines `space`, `create_pipe`, its `model` dict and
    the data, then calls this. All the common boilerplate lives here instead of
    being copy-pasted per script:
      - parse CLI args, run/resume the sweep via ``optimize_ray``,
      - dump ``<path>/model.json`` and ``<path>/best_space.json``.

    CLI args (passed by the sbatch script):
        <path> <new_trial 0|1> [max_evals]
      path          -> exp_name = exp_name_prefix + path; outputs go to ./<path>/
      new_trial     -> 1 = fresh sweep, 0 = resume
    """
    argv = sys.argv if argv is None else argv
    path = argv[1]
    new_trial = bool(int(argv[2]))
    max_evals = int(argv[3]) if len(argv) > 3 else default_max_evals
    return _run_search_backend(
        exp_name_prefix=exp_name_prefix, space=space, create_pipe=create_pipe,
        model=model, X=X, y=y, run_id=path,
        mode='fresh' if new_trial else 'resume', max_evals=max_evals,
        select_metric=select_metric, select_mode=select_mode,
        subset_fractions=subset_fractions, n_splits=n_splits,
        n_repeats_final=n_repeats_final, n_jobs=n_jobs, groups=groups,
        max_concurrent=max_concurrent, storage_path=storage_path,
        output_path=path if output_path is None else output_path,
    )

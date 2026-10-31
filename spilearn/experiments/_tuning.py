import copy
import hashlib
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
from ray.tune.callback import Callback
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


def _write_json(path, payload):
    temporary = os.fspath(path) + '.tmp'
    with open(temporary, 'w') as stream:
        json.dump(payload, stream, indent=2, cls=NpEncoder, allow_nan=False)
    os.replace(temporary, path)


class _TrialOutcomes(Callback):
    """Persist real Ray trial identities even when no metric is reported."""

    def __init__(self, path):
        self.path = os.fspath(path)
        self.trials = {}
        _write_json(self.path, self.trials)

    def _record(self, trial, **updates):
        try:
            path = trial.local_path
        except RuntimeError:
            # Ray does not initialize a directory for every pending trial.
            path = None
        path = os.path.realpath(path) if path is not None else None
        record = self.trials.setdefault(trial.trial_id, {
            'trial_id': trial.trial_id, 'config': trial.config,
            'pruned': False,
        })
        record['path'] = path
        record.update(updates)
        _write_json(self.path, self.trials)

    def on_trial_start(self, iteration, trials, trial, **info):
        self._record(trial, status='started', started=True)

    def on_trial_complete(self, iteration, trials, trial, **info):
        self._record(trial, status='terminated')

    def on_trial_result(self, iteration, trials, trial, result, **info):
        metrics = {}
        for name in ('subset_fraction', 'mean_score', 'std_score', 'metric'):
            value = result.get(name)
            metrics[name] = float(value) if value is not None and np.isfinite(value) else None
        self._record(trial, reported=True, metrics=metrics)

    def on_trial_error(self, iteration, trials, trial, **info):
        self._record(trial, status='failed')

    def mark_pruned(self, trial):
        self._record(trial, pruned=True)

    def on_experiment_end(self, trials, **info):
        for trial in trials:
            self._record(trial, status=trial.status)


class _RecordedASHA(ASHAScheduler):
    """Keep ASHA decisions unchanged and record genuine pre-final STOPs."""

    def __init__(self, recorder, final_subset, **kwargs):
        super().__init__(**kwargs)
        self.recorder = recorder
        self.final_subset = final_subset

    def on_trial_result(self, tune_controller, trial, result):
        decision = super().on_trial_result(tune_controller, trial, result)
        if decision == self.STOP and result.get('subset_fraction', 0) < self.final_subset:
            self.recorder.mark_pruned(trial)
        return decision


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
                 time_budget_s: Optional[float] = None,
                 outcome_recorder: Optional[_TrialOutcomes] = None) -> tune.ResultGrid:
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
        scheduler_args = dict(time_attr='training_iteration',
                              max_t=len(subset_fractions), grace_period=1,
                              reduction_factor=3)
        scheduler = (ASHAScheduler(**scheduler_args) if outcome_recorder is None
                     else _RecordedASHA(outcome_recorder,
                                        final_subset=max(subset_fractions), **scheduler_args))
        tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(
                num_samples=max_evals, search_alg=search_alg,
                scheduler=scheduler,
                metric='metric', mode='max',
                time_budget_s=time_budget_s,
            ),
            run_config=train.RunConfig(name=exp_name, storage_path=ray_path,
                                       callbacks=[outcome_recorder] if outcome_recorder else None),
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
    best_path = os.path.join(output_path, 'best_space.json')
    if os.path.exists(best_path):
        os.unlink(best_path)
    # Legacy checkpoints lack a verified pruning history. Keep their restore
    # path, but do not advertise exact pruning counts for resumed searches.
    recorder = (_TrialOutcomes(os.path.join(output_path, 'trial_events.json'))
                if mode == 'fresh' else None)
    digest = hashlib.sha256()
    for values in (X, y, groups):
        if values is not None:
            array = np.asarray(values)
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
    provenance = {
        'experiment': exp_name_prefix + run_id, 'run_id': run_id,
        'data_fingerprint': digest.hexdigest(), 'requested_samples': max_evals,
        'select_metric': select_metric, 'select_mode': select_mode,
        'subset_fractions': subset_fractions, 'n_splits': n_splits,
        'n_repeats_final': n_repeats_final, 'cv_seed': 42, 'n_jobs': n_jobs,
        'max_concurrent': max_concurrent, 'cpus_per_trial': cpus_per_trial,
        'time_budget_s': time_budget_s, 'optuna_seed': 42,
        'asha': {'max_t': len(subset_fractions), 'grace_period': 1, 'reduction_factor': 3},
    }
    _write_json(os.path.join(output_path, 'provenance.json'), provenance)
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
        outcome_recorder=recorder,
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
            value = metrics[select_metric]
        else:
            value = metrics.get(aliases.get(select_metric, ''), np.nan)
        return np.nan if value is None else value

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
    best = ((min if select_mode == 'min' else max)(completed, key=metric_value)
            if completed else None)
    events = dict(recorder.trials) if recorder is not None else {}
    by_path = {event['path']: event for event in events.values() if event.get('path')}
    result_by_id = {}
    for result in results:
        metrics = result.metrics or {}
        path = os.path.realpath(result.path) if getattr(result, 'path', None) else None
        event = by_path.get(path, {})
        trial_id = event.get('trial_id', metrics.get('trial_id'))
        if trial_id is None and recorder is not None:
            if ('subset_fraction' not in metrics and getattr(result, 'error', None) is None
                    and any(not item.get('started') for item in events.values())):
                # Concrete pending trial IDs/configs already come from the
                # experiment-end callback; this report-free row adds no data.
                continue
            raise ValueError('Ray trial identity is missing from result and recorded events')
        key = trial_id if trial_id is not None else object()
        events.setdefault(key, {'trial_id': trial_id, 'path': path,
                                'config': result.config})
        result_by_id[key] = result
    outcomes = []
    winner = None
    for key, event in events.items():
        result = result_by_id.get(key)
        metrics = (result.metrics or {}) if result is not None else event.get('metrics', {})
        fraction = metrics.get('subset_fraction') or 0
        error = getattr(result, 'error', None)
        if error is not None or event.get('status') in ('failed', 'ERROR'):
            status, reason = 'failed', str(error) if error is not None else 'trial_error'
        elif fraction >= final_frac:
            status, reason = 'completed', None
        elif recorder is None:
            status, reason = 'unknown', 'pruning_history_unavailable'
        elif event.get('pruned'):
            status, reason = 'pruned', 'asha_stop'
        else:
            status = 'incomplete'
            reason = 'no_report' if 'subset_fraction' not in metrics else 'budget_or_interruption'
        mean = metrics.get('mean_score', metrics.get('mean_accuracy'))
        std = metrics.get('std_score', metrics.get('std_accuracy'))
        selected = metric_value(result) if result is not None else np.nan
        outcome = {
            'trial_id': event['trial_id'], 'path': event.get('path'),
            'status': status, 'reason': reason,
            'subset_fraction': fraction, 'config': event['config'],
            'mean_score': float(mean) if mean is not None and np.isfinite(mean) else None,
            'std_score': float(std) if std is not None and np.isfinite(std) else None,
            'selection_value': float(selected) if np.isfinite(selected) else None,
        }
        outcomes.append(outcome)
        if result is best and best is not None:
            winner = outcome
    counts = {status: sum(item['status'] == status for item in outcomes)
              for status in ('completed', 'pruned', 'failed', 'incomplete')}
    if recorder is None:
        counts.update(pruned=None, incomplete=None,
                      unknown=sum(item['status'] == 'unknown' for item in outcomes))
    _write_json(os.path.join(output_path, 'search_summary.json'), {
        **provenance, 'created_trials': len(outcomes), 'counts': counts,
        'launched_trials': (sum(bool(event.get('started')) for event in events.values())
                            if recorder is not None else None),
        'outcome_coverage': 'fresh_trace' if recorder is not None else 'unavailable_resume_history',
        'trials': outcomes, 'best': winner,
    })
    if best is None:
        raise ValueError(f'no finite result at final-subset for select_metric={select_metric!r}')
    _write_json(best_path, best.config)
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

"""Generic experiment orchestration."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from ray.tune import ResultGrid
from sklearn.model_selection import cross_val_score

from ._tuning import _run_search_backend
from .spec import EvaluationSpec, PathInput, RunMode, SearchSpec

SAFE_PATH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def require_path(value: str) -> str:
    if not isinstance(value, str) or not SAFE_PATH.fullmatch(value):
        raise ValueError("unsafe PATH")
    return value


def run_search(spec: SearchSpec, *, mode: RunMode, run_id: str,
               output_path: PathInput, storage_path: PathInput,
               max_evals: Optional[int] = None, override: bool = False) -> ResultGrid:
    run_id = require_path(run_id)
    output_path = Path(output_path)
    storage_path = Path(storage_path)
    remote = storage_path / require_path(spec.prefix + run_id)
    if mode not in ("fresh", "resume"):
        raise ValueError("mode must be fresh or resume")
    if override and mode == "resume":
        raise ValueError("override and resume are mutually exclusive")

    max_evals = max_evals if max_evals is not None else spec.default_max_evals
    model = spec.base_model()
    X, y, groups = spec.load_data()
    space = spec.search_space()
    if override:
        output_resolved = output_path.resolve()
        remote_resolved = remote.resolve()
        if output_resolved == remote_resolved or remote_resolved in output_resolved.parents:
            raise ValueError("fresh output cannot be inside Ray path")
        if remote.exists():
            shutil.rmtree(remote)
        for path in (output_path / "model.json", output_path / "best_space.json",
                     output_path / "provenance.json"):
            path.unlink(missing_ok=True)

    return _run_search_backend(
        exp_name_prefix=spec.prefix,
        space=space,
        create_pipe=spec.create_pipe,
        model=copy.deepcopy(model),
        X=X,
        y=y,
        run_id=run_id,
        mode=mode,
        max_evals=max_evals,
        select_metric=spec.select_metric,
        select_mode=spec.select_mode,
        subset_fractions=tuple(spec.subset_fractions),
        n_splits=spec.n_splits,
        n_repeats_final=spec.n_repeats_final,
        n_jobs=spec.n_jobs,
        groups=groups,
        max_concurrent=spec.max_concurrent,
        cpus_per_trial=spec.cpus_per_trial,
        scorer=spec.scorer,
        time_budget_s=spec.time_budget_s,
        storage_path=storage_path,
        output_path=output_path,
    )


def load_best(path: PathInput) -> Tuple[Any, str]:
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def run_evaluation(spec: EvaluationSpec, *, mode: RunMode, run_id: str,
                   output_path: PathInput, best_space_path: PathInput) -> Dict[str, Any]:
    run_id = require_path(run_id)
    output_path = Path(output_path)
    best_space_path = Path(best_space_path).resolve()
    if mode not in ("fresh", "resume"):
        raise ValueError("mode must be fresh or resume")
    if mode == "fresh" and output_path.exists():
        raise FileExistsError("fresh output already exists")

    params, best_sha256 = load_best(best_space_path)
    model_seeds = tuple(spec.model_seeds)
    summary_path = output_path / "summary.json"
    seed_paths = [output_path / "model_seed_{}.json".format(seed) for seed in model_seeds]

    output_path.mkdir(parents=True, exist_ok=True)
    rows = []
    X = y = splits = None
    for model_seed, seed_path in zip(model_seeds, seed_paths):
        if mode == "resume" and seed_path.exists():
            saved = json.loads(seed_path.read_text())
            if saved.get("best_space_sha256") != best_sha256:
                raise ValueError("resume output was created for a different best space")
            rows.append({key: saved[key] for key in (
                "model_seed", "fold_accuracies", "mean_accuracy", "std_accuracy"
            )})
            continue

        if splits is None:
            X, y = spec.load_data()
            splits = list(spec.cv.split(X, y))
        np.random.seed(model_seed)
        pipe = spec.create_pipe(
            copy.deepcopy(spec.base_model()), copy.deepcopy(params), model_seed
        )
        scores = cross_val_score(
            pipe, X, y, cv=splits, scoring="accuracy", n_jobs=1, error_score="raise"
        )
        row = {
            "model_seed": model_seed,
            "fold_accuracies": scores.tolist(),
            "mean_accuracy": float(scores.mean()),
            "std_accuracy": float(scores.std()),
        }
        rows.append(row)
        with seed_path.open("w", encoding="utf-8") as handle:
            json.dump(dict(row, best_space_sha256=best_sha256, selected_params=params), handle)
    if splits is None and not rows:
        X, y = spec.load_data()
        splits = list(spec.cv.split(X, y))
    report = {
        "protocol": "cross_validation",
        "model_seeds": list(model_seeds),
        "n_splits": len(splits) if splits is not None else len(rows[0]["fold_accuracies"]),
        "scoring": "accuracy",
        "best_space_path": str(best_space_path),
        "best_space_sha256": best_sha256,
        "selected_params": params,
        "rows": rows,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle)
    return report

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Tuple, Union

from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator

Model = Dict[str, Any]
Params = Mapping[str, Any]
PathInput = Union[str, os.PathLike]
SearchData = Tuple[Any, Any, Optional[Any]]
EvaluationData = Tuple[Any, Any]
BaseModelFactory = Callable[[], Model]
SearchPipeFactory = Callable[[Model, Params], BaseEstimator]
EvaluationPipeFactory = Callable[[Model, Params, int], BaseEstimator]
RunMode = Literal['fresh', 'resume']
SelectMode = Literal['min', 'max']

@dataclass
class SearchSpec:
    prefix: str
    base_model: BaseModelFactory
    load_data: Callable[[], SearchData]
    search_space: Callable[[], Model]
    create_pipe: SearchPipeFactory
    select_metric: str
    select_mode: SelectMode
    subset_fractions: Tuple[float, ...]
    n_splits: int
    n_repeats_final: int
    default_max_evals: int
    n_jobs: int
    max_concurrent: int
    cpus_per_trial: int
    # Higher scores are better; search ranks mean score minus half a standard deviation.
    scorer: Union[str, Callable[[BaseEstimator, Any, Any], float]] = 'accuracy'
    time_budget_s: Optional[float] = None

@dataclass
class EvaluationSpec:
    base_model: BaseModelFactory
    load_data: Callable[[], EvaluationData]
    create_pipe: EvaluationPipeFactory
    cv: BaseCrossValidator
    model_seeds: Tuple[int, ...] = (42,)

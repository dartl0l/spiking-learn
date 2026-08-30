"""Standalone checks for the opt-in outer holdout Ray stage."""

import json
import os
import tempfile
from unittest.mock import patch

import numpy as np
import ray
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from spilearn import utils


def _pipe(model, params):
    pipe = Pipeline([("dummy", model)])
    return pipe.set_params(**params)


def _run_stage(root, exp_name, new_trial=0):
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    X_test = np.array([[4], [5]])
    y_test = np.array([1, 0])
    original_expanduser = os.path.expanduser

    def expanduser(value):
        return os.path.join(root, "ray_results") if value == "~/ray_results" else original_expanduser(value)

    with patch.object(utils.os.path, "expanduser", expanduser):
        utils._run_outer_test_stage(
            exp_name,
            root,
            new_trial,
            {"dummy__strategy": "most_frequent"},
            "mean_test_score",
            0.5,
            X,
            y,
            DummyClassifier(),
            _pipe,
            (X_test, y_test),
        )


def main():
    ray.init(num_cpus=1, include_dashboard=False, ignore_reinit_error=True)
    with tempfile.TemporaryDirectory() as root:
        exp_name = "outer_test"
        artifact = os.path.join(root, "outer_test.json")
        _run_stage(root, exp_name)
        with open(artifact) as artifact_file:
            saved = json.load(artifact_file)
        assert saved["outer_train_count"] == 4
        assert saved["outer_test_count"] == 2
        assert {"outer_test_accuracy", "outer_test_macro_f1"} <= saved.keys()

        # A valid artifact must bypass Ray, even if the prior stage exists.
        with patch.object(utils, "_run_outer_test", side_effect=AssertionError("should not run")):
            _run_stage(root, exp_name)

        try:
            _run_stage(root, exp_name, new_trial=1)
        except FileExistsError:
            pass
        else:
            raise AssertionError("fresh run reused a valid outer-test artifact")

        # A completed stage can regenerate a missing artifact without scheduling a trial.
        os.remove(artifact)
        with patch.object(utils.tune.Tuner, "fit", side_effect=AssertionError("should not fit")):
            _run_stage(root, exp_name)
        assert os.path.exists(artifact)

        with open(artifact, "w") as artifact_file:
            artifact_file.write("not json")
        with patch.object(utils.tune.Tuner, "fit", side_effect=AssertionError("should not fit")):
            try:
                _run_stage(root, exp_name)
            except ValueError:
                pass
            else:
                raise AssertionError("malformed artifact did not fail closed")

        collision_name = "outer_collision"
        collision_dir = os.path.join(root, "ray_results", f"{collision_name}_outer_test")
        os.makedirs(collision_dir)
        with open(os.path.join(collision_dir, "marker"), "w") as marker:
            marker.write("occupied")
        try:
            _run_stage(root, collision_name, new_trial=1)
        except FileExistsError:
            pass
        else:
            raise AssertionError("opt-in fresh stage collision did not fail")

    ray.shutdown()


if __name__ == "__main__":
    main()

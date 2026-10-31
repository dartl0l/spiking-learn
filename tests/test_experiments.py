import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints
from unittest.mock import Mock, patch

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold

from spilearn.experiments import EvaluationSpec, SearchSpec, run_evaluation, run_search
from spilearn.experiments import _runner
import spilearn.experiments._tuning as utils


class ExperimentsTest(unittest.TestCase):

    def search_spec(self):
        return SearchSpec(
            prefix='search_',
            base_model=lambda: {'syn_dict_exc': {}},
            load_data=lambda: ([], [], None),
            search_space=lambda: {},
            create_pipe=lambda model, params: None,
            select_metric='metric',
            select_mode='max',
            subset_fractions=(1.0,),
            n_splits=1,
            n_repeats_final=1,
            default_max_evals=2,
            n_jobs=1,
            max_concurrent=3,
            cpus_per_trial=1,
        )

    def evaluation_spec(self):
        return EvaluationSpec(
            base_model=lambda: {},
            load_data=lambda: (np.array([[0], [1], [2], [3]]), np.array([0, 0, 1, 1])),
            create_pipe=lambda model, params, seed: DummyClassifier(strategy='most_frequent'),
            cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
        )

    def test_public_type_hints_resolve(self):
        for target in (
            SearchSpec, EvaluationSpec, run_search, _runner.load_best,
            run_evaluation, utils.run_train_test, utils.run_cv,
            utils.optimize_ray, utils.run_experiment,
        ):
            with self.subTest(target=target):
                get_type_hints(target)

    def test_safe_path(self):
        self.assertEqual(_runner.require_path('run-1.0'), 'run-1.0')
        with self.assertRaises(ValueError): _runner.require_path('../run')

    def test_search_forwards_explicit_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); seen = {}
            def backend(**kwargs):
                seen.update(kwargs)
                return [SimpleNamespace(error=None, metrics={'subset_fraction': 1.0, 'metric': .5})]
            with patch.object(_runner, '_run_search_backend', backend):
                run_search(self.search_spec(), mode='fresh', run_id='run', output_path=root / 'output', storage_path=root / 'storage')
            self.assertEqual(seen['output_path'], root / 'output')
            self.assertEqual(seen['storage_path'], root / 'storage')
        self.assertEqual(seen['max_concurrent'], 3)
        self.assertEqual(seen['cpus_per_trial'], 1)
        self.assertEqual(seen['scorer'], 'accuracy')
        self.assertIsNone(seen['time_budget_s'])

    def test_search_forwards_time_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); seen = {}

            def backend(**kwargs):
                seen.update(kwargs)
                return [SimpleNamespace(error=None, metrics={'subset_fraction': 1.0, 'metric': .5})]

            with patch.object(_runner, '_run_search_backend', backend):
                run_search(
                    replace(self.search_spec(), time_budget_s=7.5),
                    mode='fresh', run_id='run',
                    output_path=root / 'output', storage_path=root / 'storage',
                )

            self.assertEqual(seen['time_budget_s'], 7.5)

    def test_search_forwards_scorer(self):
        scorer = lambda estimator, X_val, y_val: -1.0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); seen = {}
            def backend(**kwargs):
                seen.update(kwargs)
                return [SimpleNamespace(error=None, metrics={'subset_fraction': 1.0, 'metric': .5})]
            with patch.object(_runner, '_run_search_backend', backend):
                run_search(
                    replace(self.search_spec(), scorer=scorer), mode='fresh', run_id='run',
                    output_path=root / 'output', storage_path=root / 'storage',
                )
            self.assertIs(seen['scorer'], scorer)

    def test_evaluation_rejects_an_unsafe_run_id_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                run_evaluation(self.evaluation_spec(), mode='fresh', run_id='../run', output_path=root / 'output', best_space_path=root / 'best_space.json')
            self.assertFalse((root / 'output').exists())

    def test_search_override_rejects_output_inside_ray_path(self):
        for nested in (False, True):
            with self.subTest(nested=nested), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                storage = root / 'storage'; remote = storage / 'search_run'
                output = remote / 'local' if nested else remote
                output.mkdir(parents=True)
                ray_sentinel = remote / 'ray-sentinel'; ray_sentinel.write_text('keep')
                user_sentinel = output / 'user-sentinel'; user_sentinel.write_text('keep')
                backend = Mock()
                with self.assertRaises(ValueError), patch.object(_runner, '_run_search_backend', backend):
                    run_search(self.search_spec(), mode='fresh', run_id='run',
                               output_path=output, storage_path=storage, override=True)
                backend.assert_not_called()
                self.assertTrue(ray_sentinel.exists())
                self.assertTrue(user_sentinel.exists())

    def test_search_fresh_preserves_and_override_clears_owned_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root / 'output'; storage = root / 'storage'
            output.mkdir(); remote = storage / 'search_run'; remote.mkdir(parents=True)
            for name in ('model.json', 'best_space.json', 'provenance.json'):
                (output / name).write_text('stale')
            unrelated = output / 'evaluation.json'; unrelated.write_text('keep')
            sentinel = storage / 'search_other'; sentinel.mkdir(); (sentinel / 'keep').write_text('keep')
            with patch.object(_runner, '_run_search_backend', return_value=[]):
                run_search(self.search_spec(), mode='fresh', run_id='run', output_path=output, storage_path=storage)
            self.assertTrue(remote.exists())
            self.assertTrue(all((output / name).read_text() == 'stale' for name in ('model.json', 'best_space.json', 'provenance.json')))
            with patch.object(_runner, '_run_search_backend', return_value=[]):
                run_search(self.search_spec(), mode='fresh', run_id='run', output_path=output, storage_path=storage, override=True)
            self.assertFalse(remote.exists())
            self.assertFalse(any((output / name).exists() for name in ('model.json', 'best_space.json', 'provenance.json')))
            self.assertTrue(unrelated.exists())
            self.assertTrue((sentinel / 'keep').exists())
            with patch.object(_runner, '_run_search_backend', return_value=[]) as backend:
                run_search(self.search_spec(), mode='resume', run_id='run', output_path=output, storage_path=storage)
            self.assertEqual(backend.call_args.kwargs['mode'], 'resume')
            with self.assertRaises(ValueError):
                run_search(self.search_spec(), mode='resume', run_id='run', output_path=output, storage_path=storage, override=True)

    def test_search_backend_selects_current_and_historical_score_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'; output.mkdir()
            class Results(list):
                def get_best_result(self, metric, mode):
                    raise AssertionError('selection must use reported score fields')

            results = Results([
                SimpleNamespace(
                    error=None,
                    metrics={'trial_id': 'current', 'subset_fraction': 1.0,
                             'metric': .6, 'mean_score': .6},
                    config={'winner': 'current'},
                ),
                SimpleNamespace(
                    error=None,
                    metrics={'trial_id': 'historical', 'subset_fraction': 1.0,
                             'metric': .4, 'mean_accuracy': .4},
                    config={'winner': 'historical'},
                ),
            ])
            kwargs = dict(exp_name_prefix='search_', space={}, create_pipe=lambda model, params: None,
                          model={'syn_dict_exc': {}}, X=[], y=[], run_id='run', mode='fresh', max_evals=1,
                          select_metric='metric', subset_fractions=(.2, 1.0), n_splits=1,
                          n_repeats_final=1, n_jobs=1, groups=None, max_concurrent=1,
                          storage_path=Path(directory) / 'storage', output_path=output)
            for metric, mode, winner in (
                ('metric', 'max', 'current'),
                ('mean_score', 'min', 'historical'),
                ('mean_accuracy', 'max', 'current'),
            ):
                with self.subTest(metric=metric, mode=mode), patch.object(utils, 'optimize_ray', return_value=results):
                    utils._run_search_backend(
                        **dict(kwargs, select_metric=metric, select_mode=mode)
                    )
                self.assertEqual(json.loads((output / 'best_space.json').read_text()), {'winner': winner})

    def test_search_backend_does_not_alias_present_nonfinite_score(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'; output.mkdir()
            results = [
                SimpleNamespace(
                    error=None,
                    metrics={'trial_id': 'invalid', 'subset_fraction': 1.0,
                             'mean_score': np.nan, 'mean_accuracy': .9},
                    config={'winner': 'invalid'},
                ),
                SimpleNamespace(
                    error=None,
                    metrics={'trial_id': 'historical', 'subset_fraction': 1.0,
                             'mean_accuracy': .5},
                    config={'winner': 'historical'},
                ),
            ]
            with patch.object(utils, 'optimize_ray', return_value=results):
                utils._run_search_backend(
                    exp_name_prefix='search_', space={}, create_pipe=lambda model, params: None,
                    model={'syn_dict_exc': {}}, X=[], y=[], run_id='run', mode='fresh', max_evals=1,
                    select_metric='mean_score', select_mode='max', subset_fractions=(.2, 1.0),
                    n_splits=1, n_repeats_final=1, n_jobs=1, groups=None, max_concurrent=1,
                    storage_path=Path(directory) / 'storage', output_path=output,
                )
            self.assertEqual(json.loads((output / 'best_space.json').read_text()), {'winner': 'historical'})

    def test_search_backend_rejects_missing_finite_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'; output.mkdir()
            results = [SimpleNamespace(
                error=None,
                metrics={'trial_id': 'invalid', 'subset_fraction': 1.0, 'mean_score': np.nan},
                config={'winner': 'invalid'},
            )]
            with patch.object(utils, 'optimize_ray', return_value=results), self.assertRaisesRegex(
                ValueError, "no finite result at final-subset for select_metric='mean_score'",
            ):
                utils._run_search_backend(
                    exp_name_prefix='search_', space={}, create_pipe=lambda model, params: None,
                    model={'syn_dict_exc': {}}, X=[], y=[], run_id='run', mode='fresh', max_evals=1,
                    select_metric='mean_score', select_mode='max', subset_fractions=(1.0,),
                    n_splits=1, n_repeats_final=1, n_jobs=1, groups=None, max_concurrent=1,
                    storage_path=Path(directory) / 'storage', output_path=output,
                )

    def test_search_backend_selects_full_fidelity_result(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            output.mkdir()

            class Results(list):
                def get_best_result(self, metric, mode):
                    raise AssertionError('full-fidelity result should be selected')

            results = Results([
                SimpleNamespace(error=None, metrics={'trial_id': 'partial', 'subset_fraction': .2, 'metric': .9}, config={'winner': 'partial'}),
                SimpleNamespace(error=None, metrics={'trial_id': 'full', 'subset_fraction': 1.0, 'metric': .5}, config={'winner': 'full'}),
            ])
            with patch.object(utils, 'optimize_ray', return_value=results):
                utils._run_search_backend(
                    exp_name_prefix='search_', space={}, create_pipe=lambda model, params: None,
                    model={'syn_dict_exc': {}}, X=[], y=[], run_id='run', mode='fresh', max_evals=1,
                    select_metric='metric', select_mode='max', subset_fractions=(.2, 1.0),
                    n_splits=1, n_repeats_final=1, n_jobs=1, groups=None, max_concurrent=1,
                    storage_path=Path(directory) / 'storage', output_path=output,
                )
            self.assertEqual(json.loads((output / 'best_space.json').read_text()), {'winner': 'full'})

    def test_utils_storage_and_output_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); storage = root / 'storage'; captured = {}

            class Tuner:
                @staticmethod
                def can_restore(path):
                    captured['restore_check'] = path
                    return False

                @staticmethod
                def restore(path, **kwargs):
                    captured['restore'] = (path, kwargs)
                    return SimpleNamespace(fit=lambda: 'restored')

                def __init__(self, *args, **kwargs):
                    captured['tuner'] = (args, kwargs)

                def fit(self):
                    return 'fresh'

            def run_config(**kwargs):
                captured['run_config'] = kwargs
                return kwargs

            fake_tune = SimpleNamespace(with_parameters=lambda func, **kwargs: 'trainable', Tuner=Tuner,
                                        TuneConfig=lambda **kwargs: kwargs)
            fake_train = SimpleNamespace(RunConfig=run_config)
            with patch.object(utils, 'tune', fake_tune), patch.object(utils, 'train', fake_train), \
                 patch.object(utils, 'OptunaSearch', lambda **kwargs: 'search'), \
                 patch.object(utils, 'ConcurrencyLimiter', lambda *args, **kwargs: 'limited'), \
                 patch.object(utils, 'ASHAScheduler', lambda **kwargs: 'scheduler'):
                self.assertEqual(
                    utils.optimize_ray([], [], None, None, {}, {}, 'experiment', storage_path=storage),
                    'fresh',
                )
                self.assertEqual(captured['run_config']['storage_path'], os.fspath(storage))
                self.assertEqual(
                    utils.optimize_ray(
                        [], [], None, None, {}, {}, 'budgeted', max_evals=-1, time_budget_s=7.5,
                    ),
                    'fresh',
                )
                self.assertEqual(captured['tuner'][1]['tune_config']['num_samples'], -1)
                self.assertEqual(captured['tuner'][1]['tune_config']['time_budget_s'], 7.5)
                Tuner.can_restore = staticmethod(lambda path: True)
                self.assertEqual(
                    utils.optimize_ray(
                        [], [], None, None, {}, {}, 'experiment', new_trial=False, storage_path=storage,
                    ),
                    'restored',
                )
                self.assertEqual(captured['restore'][0], os.fspath(storage / 'experiment'))
                Tuner.can_restore = staticmethod(lambda path: False)
                utils.optimize_ray([], [], None, None, {}, {}, 'default')
                self.assertEqual(captured['run_config']['storage_path'], os.path.expanduser('~/ray_results'))

            class Results(list):
                def get_best_result(self, metric, mode):
                    return SimpleNamespace(config={'x': 1})

            results = Results([
                SimpleNamespace(
                    error=None,
                    metrics={'trial_id': 'storage-check', 'subset_fraction': 1.0, 'mean_score': .5},
                    config={'x': 1},
                ),
            ])
            output = root / 'output'
            with patch.object(utils, 'optimize_ray', return_value=results) as optimize:
                utils.run_experiment('prefix_', {}, lambda model, params: None, {'syn_dict_exc': {}}, [], [], argv=['run', 'id', '1'], output_path=output, storage_path=storage)
            self.assertTrue((output / 'model.json').exists())
            self.assertTrue((output / 'best_space.json').exists())
            self.assertEqual(optimize.call_args.kwargs['storage_path'], storage)
            legacy = root / 'legacy'
            with patch.object(utils, 'optimize_ray', return_value=results) as optimize:
                utils.run_experiment(
                    'prefix_', {}, lambda model, params: None, {'syn_dict_exc': {}}, [], [],
                    argv=['run', str(legacy), '1'],
                )
            self.assertTrue((legacy / 'model.json').exists())
            self.assertIsNone(optimize.call_args.kwargs['storage_path'])

    def test_evaluation_same_revision_resume_uses_saved_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            best = root / 'best.json'
            best.write_text(json.dumps({'alpha': 1}))
            self.assertEqual(self.evaluation_spec().model_seeds, (42,))
            calls = []

            def create_pipe(model, params, seed):
                calls.append(seed)
                return DummyClassifier(strategy='most_frequent')

            cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
            spec = replace(
                self.evaluation_spec(), create_pipe=create_pipe, cv=cv, model_seeds=(11, 13)
            )
            with patch.object(cv, 'split', wraps=cv.split) as split:
                first = run_evaluation(spec, mode='fresh', run_id='run', output_path=root / 'output', best_space_path=best)
                second = run_evaluation(spec, mode='resume', run_id='run', output_path=root / 'output', best_space_path=best)
            self.assertEqual(split.call_count, 1)
        self.assertEqual(first, second)
        self.assertEqual(calls, [11, 13])
        self.assertEqual(first['model_seeds'], [11, 13])
        self.assertEqual(first['n_splits'], 2)

    def test_evaluation_resume_rebuilds_summary_for_changed_seed_list(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            best = root / 'best.json'
            best.write_text(json.dumps({'alpha': 1}))
            calls = []

            def create_pipe(model, params, seed):
                calls.append(seed)
                return DummyClassifier(strategy='most_frequent')

            first = replace(self.evaluation_spec(), create_pipe=create_pipe, model_seeds=(11, 13))
            resumed = replace(first, model_seeds=(11,))
            run_evaluation(first, mode='fresh', run_id='run', output_path=root / 'output', best_space_path=best)
            report = run_evaluation(resumed, mode='resume', run_id='run', output_path=root / 'output', best_space_path=best)
        self.assertEqual(calls, [11, 13])
        self.assertEqual(report['model_seeds'], [11])
        self.assertEqual([row['model_seed'] for row in report['rows']], [11])

    def test_evaluation_resume_runs_only_missing_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            best = root / 'best.json'
            best.write_text(json.dumps({'alpha': 1}))
            output = root / 'output'
            calls = []
            interrupted = True

            def create_pipe(model, params, seed):
                calls.append(seed)
                nonlocal interrupted
                if seed == 13 and interrupted:
                    interrupted = False
                    raise RuntimeError('interrupted')
                return DummyClassifier(strategy='most_frequent')

            spec = replace(self.evaluation_spec(), create_pipe=create_pipe, model_seeds=(11, 13))
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                run_evaluation(spec, mode='fresh', run_id='run', output_path=output, best_space_path=best)
            self.assertTrue((output / 'model_seed_11.json').exists())
            report = run_evaluation(spec, mode='resume', run_id='run', output_path=output, best_space_path=best)
            uninterrupted = run_evaluation(
                replace(self.evaluation_spec(), model_seeds=(11, 13)), mode='fresh', run_id='clean',
                output_path=root / 'clean', best_space_path=best,
            )
        self.assertEqual(calls, [11, 13, 13])
        self.assertEqual(report, uninterrupted)

    def test_evaluation_resume_rejects_changed_best_space(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            best = root / 'best.json'
            best.write_text(json.dumps({'alpha': 1}))
            output = root / 'output'
            spec = replace(self.evaluation_spec(), model_seeds=(11,))
            run_evaluation(spec, mode='fresh', run_id='run', output_path=output, best_space_path=best)
            best.write_text(json.dumps({'alpha': 2}))
            with self.assertRaises(ValueError):
                run_evaluation(spec, mode='resume', run_id='run', output_path=output, best_space_path=best)
            (output / 'summary.json').unlink()
            with self.assertRaises(ValueError):
                run_evaluation(spec, mode='resume', run_id='run', output_path=output, best_space_path=best)


    def test_search_load_failure_does_not_write_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = SearchSpec(
                prefix='search_',
                base_model=lambda: {'syn_dict_exc': {}},
                load_data=lambda: (_ for _ in ()).throw(ValueError('bad data')),
                search_space=lambda: {},
                create_pipe=lambda model, params: None,
                select_metric='metric',
                select_mode='max',
                subset_fractions=(1.0,),
                n_splits=1,
                n_repeats_final=1,
                default_max_evals=2,
                n_jobs=1,
                max_concurrent=3,
                cpus_per_trial=1,
            )
            with self.assertRaises(ValueError):
                run_search(spec, mode='fresh', run_id='run', output_path=root / 'output', storage_path=root / 'storage')
            self.assertFalse((root / 'output').exists())


if __name__ == '__main__': unittest.main()

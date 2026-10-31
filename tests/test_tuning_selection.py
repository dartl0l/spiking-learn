"""Focused checks for full-stage selection and truthful trial outcomes."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from spilearn.experiments import _tuning


def result(trial_id, fraction, mean, *, error=None, **metrics):
    return SimpleNamespace(
        path=f'/ray/{trial_id}', error=error,
        config={'decoder': 'vote', 'value': trial_id},
        metrics={'trial_id': trial_id, 'subset_fraction': fraction,
                 'mean_score': mean, 'std_score': .01, **metrics})


def search(output, **kwargs):
    arguments = dict(
        exp_name_prefix='test-', space={}, create_pipe=lambda model, params: None,
        model={}, X=np.zeros((4, 1)), y=np.array([0, 0, 1, 1]),
        run_id='run', mode='fresh', max_evals=5,
        select_metric='mean_score', select_mode='max',
        subset_fractions=(.2, 1.), n_splits=2, n_repeats_final=1,
        n_jobs=1, groups=None, max_concurrent=1, output_path=output)
    arguments.update(kwargs)
    return _tuning._run_search_backend(**arguments)


class FullStageSelectionTests(unittest.TestCase):
    def test_partial_only_persists_outcomes_and_removes_stale_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'best_space.json').write_text('{"old": true}')
            with patch.object(_tuning, 'optimize_ray', return_value=[result('partial', .2, .99)]):
                with self.assertRaisesRegex(ValueError, 'final'):
                    search(output)
            self.assertFalse((output / 'best_space.json').exists())
            summary = json.loads((output / 'search_summary.json').read_text())
            self.assertIsNone(summary['best'])
            self.assertEqual(summary['counts']['incomplete'], 1)

    def test_full_stage_mean_wins_and_decoder_is_exported(self):
        trials = [result('partial', .2, .99, metric=.99),
                  result('full-a', 1., .8, metric=.8),
                  result('full-b', 1., .9, metric=.7)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(_tuning, 'optimize_ray', return_value=trials):
                search(output)
            self.assertEqual(json.loads((output / 'best_space.json').read_text()),
                             trials[2].config)
            summary = json.loads((output / 'search_summary.json').read_text())
            self.assertEqual(summary['best']['trial_id'], 'full-b')
            self.assertEqual(summary['best']['mean_score'], .9)
            self.assertEqual(summary['counts']['completed'], 2)

    def test_generic_selection_honors_metric_and_mode(self):
        trials = [result('full-a', 1., .8, loss=.1),
                  result('full-b', 1., .9, loss=.2)]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(_tuning, 'optimize_ray', return_value=trials):
                search(directory, select_metric='loss', select_mode='min')
            best = json.loads((Path(directory) / 'best_space.json').read_text())
            self.assertEqual(best['value'], 'full-a')

    def test_identity_and_outcomes_include_no_report_trials(self):
        trials = [result('done', 1., .8), result('pruned', .2, .7),
                  result('deadline', .2, .6), result('failed', .2, .5, error=RuntimeError('failure')),
                  SimpleNamespace(path='/ray/no-report', error=None,
                                  metrics={}, config={'decoder': 'vote'})]

        def optimize(**kwargs):
            recorder = kwargs['outcome_recorder']
            for trial_result in trials:
                trial_id = Path(trial_result.path).name
                trial = SimpleNamespace(trial_id=trial_id, local_path=trial_result.path,
                                        config=trial_result.config)
                recorder.on_trial_start(0, [], trial)
                if trial_id == 'pruned':
                    recorder.mark_pruned(trial)
            return trials

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(_tuning, 'optimize_ray', side_effect=optimize):
                search(directory)
            summary = json.loads((Path(directory) / 'search_summary.json').read_text())
            self.assertEqual(summary['counts'],
                             {'completed': 1, 'pruned': 1, 'failed': 1, 'incomplete': 2})
            no_report = next(item for item in summary['trials']
                             if item['trial_id'] == 'no-report')
            self.assertEqual(no_report['reason'], 'no_report')

    def test_legacy_resume_reaches_restore_without_inventing_pruning(self):
        trials = [result('restored-full', 1., .8), result('old-partial', .2, .9)]
        with tempfile.TemporaryDirectory() as directory:
            storage = Path(directory) / 'ray'
            with patch.object(_tuning.tune, 'with_parameters', return_value='trainable'), \
                 patch.object(_tuning.tune, 'Tuner') as tuner:
                tuner.can_restore.return_value = True
                tuner.restore.return_value.fit.return_value = trials
                _tuning.run_experiment(
                    'legacy-', {}, lambda model, params: None, {}, [], [],
                    argv=['runner', 'run', '0', '2'], subset_fractions=(.2, 1.),
                    storage_path=storage, output_path=directory)
                tuner.restore.assert_called_once_with(
                    str(storage / 'legacy-run'), trainable='trainable', resume_errored=True)
                tuner.assert_not_called()
            summary = json.loads((Path(directory) / 'search_summary.json').read_text())
            self.assertEqual(summary['best']['trial_id'], 'restored-full')
            self.assertIsNone(summary['counts']['pruned'])
            self.assertIsNone(summary['counts']['incomplete'])
            self.assertEqual(summary['counts']['unknown'], 1)
            self.assertEqual(summary['outcome_coverage'], 'unavailable_resume_history')

    def test_pending_identity_does_not_require_a_directory_or_report(self):
        class Pending:
            trial_id = 'pending'
            status = 'PENDING'
            config = {'decoder': 'vote'}

            @property
            def local_path(self):
                raise RuntimeError('trial_dir_name is not set')

        def optimize(**kwargs):
            kwargs['outcome_recorder'].on_experiment_end([Pending()])
            return [result('full', 1., .8),
                    SimpleNamespace(path=None, error=None, metrics={}, config={})]

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(_tuning, 'optimize_ray', side_effect=optimize):
                search(directory)
            summary = json.loads((Path(directory) / 'search_summary.json').read_text())
            pending = next(item for item in summary['trials'] if item['trial_id'] == 'pending')
            self.assertIsNone(pending['path'])
            self.assertEqual(pending['status'], 'incomplete')
            self.assertEqual(pending['reason'], 'no_report')

    def test_scheduler_delegates_and_records_only_partial_stops(self):
        recorder = SimpleNamespace(mark_pruned=Mock())
        scheduler = _tuning._RecordedASHA(recorder, final_subset=1., max_t=2)
        trial = SimpleNamespace(trial_id='synthetic')
        with patch.object(_tuning.ASHAScheduler, 'on_trial_result',
                          return_value=_tuning.ASHAScheduler.STOP) as delegate:
            self.assertEqual(scheduler.on_trial_result(None, trial, {'subset_fraction': .2}),
                             _tuning.ASHAScheduler.STOP)
            scheduler.on_trial_result(None, trial, {'subset_fraction': 1.})
        self.assertEqual(delegate.call_count, 2)
        recorder.mark_pruned.assert_called_once_with(trial)


if __name__ == '__main__':
    unittest.main()

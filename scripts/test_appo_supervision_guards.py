"""Protocol binding and interrupted-run completion checks; no simulator/GPU."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from prepare_appo_supervision import (
    SPLITS, canonical_hash, validate_collection_binding, validate_expert_rows,
)
import run_appo_supervision as runner


class SupervisionGuards(unittest.TestCase):
    def setUp(self):
        self.cases = [{'id': split, 'split': split, 'seed': i, 'variant': 'doom_basic',
                       'spec': {'task': 'shooting', 'scenario': 'basic', 'frame_skip': 4, 'max_steps': 75}}
                      for i, split in enumerate(SPLITS)]
        self.primary = {'checkpoint_repository': 'expert/basic', 'checkpoint_revision': 'fixed',
                        'checkpoint_sha256': 'a' * 64, 'controller': 'greedy', 'epsilon': .1, 'sampling_seed': 17}
        self.protocol = {'case_sha256': 'c' * 64, 'case_counts': dict.fromkeys(SPLITS, 1),
                         'primary': self.primary, 'initialization': {'weights_sha256': 'hash'},
                         'training': {'arms': [{'name': n} for n in ('hard_s17', 'soft_s17', 'hard_s29')],
                                      'policy_pool_weights': {}}, 'evaluation': {}}
        policy = {'engine': 'sample_factory_appo', 'render_adapter': 'mirrored_native',
                  'controller': 'greedy', 'epsilon': .1, 'sampling_seed': 17,
                  'model': {'model': 'expert/basic', 'revision': 'fixed', 'checkpoint_sha256': 'a' * 64},
                  'checkpoint_sha256': {'model': 'a' * 64}}
        self.collection = {'finished': True, 'limit': 0, 'cases_sha256': 'c' * 64,
                           'config_sha256': 'p' * 64, 'selected_cases': list(SPLITS),
                           'selected_cases_sha256': canonical_hash(self.cases),
                           'episodes': 5, 'selected_episodes': 5, 'selected_before_limit': 5,
                           'split_episodes': dict.fromkeys(SPLITS, 1), 'split_records': dict.fromkeys(SPLITS, 1),
                           'policy': policy, 'continuation_policy_id': canonical_hash(policy),
                           'episode_sha256': 'e' * 64}
        self.manifest = {'episode_sha256': 'e' * 64,
                         'continuation_policy_id': self.collection['continuation_policy_id'],
                         'collection_policy': policy, 'policy_target_kind': 'expert_action'}
        self.rows = [{'split': case['split'], 'expert': copy.deepcopy(policy['model']),
                      'metadata': {'task': 'shooting', 'episode_id': case['id'], 'spec': case['spec'],
                                   'continuation_policy_id': self.collection['continuation_policy_id'],
                                   'policy_target_kind': 'expert_action', 'decision_index': 0}}
                     for case in self.cases]

    def test_full_registered_collection_and_rows_pass(self):
        validate_collection_binding(self.collection, self.protocol, self.cases, 'p' * 64)
        validate_expert_rows(self.rows, self.manifest, self.collection, self.cases, 'expert_action')

    def test_partial_splits_with_limit_zero_are_rejected(self):
        changed = copy.deepcopy(self.collection)
        changed['selected_cases'] = list(SPLITS[:-1])
        changed['episodes'] = changed['selected_episodes'] = changed['selected_before_limit'] = 4
        changed['selected_cases_sha256'] = canonical_hash(self.cases[:-1])
        changed['split_episodes'].pop('ood')
        with self.assertRaisesRegex(ValueError, 'full registered'):
            validate_collection_binding(changed, self.protocol, self.cases, 'p' * 64)

    def test_other_protocol_cases_or_checkpoint_are_rejected(self):
        for key in ('config_sha256', 'cases_sha256'):
            changed = copy.deepcopy(self.collection)
            changed[key] = 'b' * 64
            with self.assertRaises(ValueError):
                validate_collection_binding(changed, self.protocol, self.cases, 'p' * 64)
        changed = copy.deepcopy(self.collection)
        changed['policy']['model']['checkpoint_sha256'] = 'b' * 64
        changed['policy']['checkpoint_sha256']['model'] = 'b' * 64
        changed['continuation_policy_id'] = canonical_hash(changed['policy'])
        with self.assertRaisesRegex(ValueError, 'registered checkpoint'):
            validate_collection_binding(changed, self.protocol, self.cases, 'p' * 64)

    def test_missing_episode_duplicate_index_and_wrong_target_source_fail(self):
        for rows in (self.rows[:-1], self.rows + [copy.deepcopy(self.rows[0])]):
            with self.assertRaises(ValueError):
                validate_expert_rows(rows, self.manifest, self.collection, self.cases, 'expert_action')
        changed = copy.deepcopy(self.rows)
        changed[0]['expert']['checkpoint_sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'different expert'):
            validate_expert_rows(changed, self.manifest, self.collection, self.cases, 'expert_action')

    def test_complete_jobs_require_exact_unique_set_and_zero_codes(self):
        expected = runner.expected_job_names(self.protocol['training']['arms'])
        self.assertEqual(len(expected), 21)
        records = [{'name': name, 'returncode': 0} for name in expected]
        self.assertTrue(runner.completion_status(expected, records, [])['finished'])
        failed = copy.deepcopy(records)
        failed[0]['returncode'] = 1
        for variant in (records[:-1], records + [records[0]], failed,
                        records + [{'name': 'unregistered', 'returncode': 0}]):
            self.assertFalse(runner.completion_status(expected, variant, [])['finished'])
        self.assertFalse(runner.completion_status(expected, records, [], aborted=True)['finished'])
        self.assertFalse(runner.completion_status(expected, records, ['worker failed'])['finished'])

    def test_baseexception_writes_incomplete_manifest_before_reraising(self):
        # Exercise the actual runner exception/finally path, while suppressing
        # preflight IO and raising before any threads, subprocesses or GPUs run.
        for interruption in (KeyboardInterrupt(), SystemExit(7)):
            with self.subTest(interruption=type(interruption).__name__), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / 'experiment'
                argv = ['run_appo_supervision', '--datasets', tmp, '--init-checkpoint', tmp,
                        '--output', str(output)]
                dataset = {'collection_manifest': {}, 'expert_manifest': {},
                           'split_sha256': dict.fromkeys(SPLITS, 'hash')}
                with patch.object(runner.sys, 'argv', argv), \
                     patch.object(runner, 'load_protocol', return_value=(self.protocol, self.cases, 'p' * 64)), \
                     patch.object(runner, 'file_sha256', return_value='hash'), \
                     patch.object(runner, 'read_unified_dataset', return_value=([], dataset, [], {})), \
                     patch.object(runner, 'validate_collection_binding'), \
                     patch.object(runner, 'validate_expert_rows'), \
                     patch.object(runner.concurrent.futures, 'ThreadPoolExecutor', side_effect=interruption):
                    with self.assertRaises(type(interruption)):
                        runner.main()
                manifest = json.loads((output / 'experiment.json').read_text())
                self.assertFalse(manifest['finished'])
                self.assertTrue(manifest['aborted'])
                self.assertEqual(len(manifest['missing_jobs']), 21)
                self.assertIn(type(interruption).__name__, manifest['errors'][0])

    def test_runner_reads_repo_configs_independent_of_cwd(self):
        # configs/unified_games_v1_cases.jsonl used to be opened relative to the
        # process cwd, so launching the runner from outside the repository root
        # failed with FileNotFoundError after output directories were created.
        interruption = KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'experiment'
            argv = ['run_appo_supervision', '--datasets', tmp, '--init-checkpoint', tmp,
                    '--output', str(output)]
            dataset = {'collection_manifest': {}, 'expert_manifest': {},
                       'split_sha256': dict.fromkeys(SPLITS, 'hash')}
            previous_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with patch.object(runner.sys, 'argv', argv), \
                     patch.object(runner, 'load_protocol', return_value=(self.protocol, self.cases, 'p' * 64)), \
                     patch.object(runner, 'file_sha256', return_value='hash'), \
                     patch.object(runner, 'read_unified_dataset', return_value=([], dataset, [], {})), \
                     patch.object(runner, 'validate_collection_binding'), \
                     patch.object(runner, 'validate_expert_rows'), \
                     patch.object(runner.concurrent.futures, 'ThreadPoolExecutor', side_effect=interruption):
                    with self.assertRaises(KeyboardInterrupt):
                        runner.main()
            finally:
                os.chdir(previous_cwd)


if __name__ == '__main__':
    unittest.main()

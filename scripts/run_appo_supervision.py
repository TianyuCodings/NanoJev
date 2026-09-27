#!/usr/bin/env python3
"""Run the frozen APPO Basic supervision comparison with one GPU per arm."""
import argparse
import concurrent.futures
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from train_unified_games import file_sha256, read_unified_dataset
from prepare_appo_supervision import load_protocol, validate_collection_binding, validate_expert_rows


def expected_job_names(arms):
    names = [arm['name'] for arm in arms]
    if len(names) != len(set(names)) or set(names) & {'frozen_sft', 'random'}:
        raise ValueError('Training arm names must be unique and distinct from baseline names')
    expected = ['random_fresh', 'random_fresh_replay']
    for name in [*names, 'frozen_sft']:
        if name != 'frozen_sft':
            expected.append(name + '_train')
        expected.extend(name + suffix for suffix in ('_fresh', '_fresh_replay', '_regression', '_regression_replay'))
    if len(expected) != len(set(expected)):
        raise ValueError('Training arm names produce colliding job names')
    return sorted(expected)


def completion_status(expected, records, errors, aborted=False):
    counts = Counter(record['name'] for record in records)
    missing = sorted(set(expected) - set(counts))
    unexpected = sorted(set(counts) - set(expected))
    repeated = sorted(name for name, count in counts.items() if count != 1)
    failed = [record['name'] for record in records
              if type(record.get('returncode')) is not int or record['returncode'] != 0]
    return {'finished': not (errors or aborted or missing or unexpected or repeated or failed),
            'aborted': aborted, 'expected_jobs': expected, 'missing_jobs': missing,
            'unexpected_jobs': unexpected, 'repeated_jobs': repeated, 'failed_jobs': failed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path,
                    default=Path(__file__).resolve().parent.parent / 'configs/appo_basic_supervision_v1.json')
    parser.add_argument('--datasets', type=Path, required=True)
    parser.add_argument('--init-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3', help='Three training arms and one frozen baseline; do not include occupied GPUs')
    args = parser.parse_args()
    protocol, cases, protocol_sha256 = load_protocol(args.protocol)
    cfg = protocol['training']
    expected_jobs = expected_job_names(cfg['arms'])
    gpus = args.gpus.split(',')
    if len(gpus) != len(cfg['arms']) + 1 or len(gpus) != len(set(gpus)) or args.output.exists():
        parser.error('Provide distinct GPUs for every training arm and the baseline, and a new output directory')
    if file_sha256(args.init_checkpoint / 'best.safetensors') != protocol['initialization']['weights_sha256']:
        raise ValueError('Wrong SFT initialization checkpoint')
    for target in ('hard', 'soft'):
        records, manifest, _, _ = read_unified_dataset(args.datasets / target)
        collection = manifest['collection_manifest']
        validate_collection_binding(collection, protocol, cases, protocol_sha256)
        validate_expert_rows(records, manifest['expert_manifest'], collection, cases,
                             'expert_action' if target == 'hard' else 'expert_distribution')
        if set(manifest['split_sha256']) != {'train', 'dev', 'calibration', 'test', 'ood'}:
            raise ValueError('Training corpus must cover all five split hashes')
        for split, expected in manifest['split_sha256'].items():
            if file_sha256(args.datasets / target / (split + '.jsonl')) != expected:
                raise ValueError('Training corpus changed')
    args.output.mkdir(parents=True)
    (args.output / 'logs').mkdir()
    weights = args.output / 'policy_pool_weights.json'
    weights.write_text(json.dumps(cfg['policy_pool_weights'], indent=2) + '\n')
    py = sys.executable
    script_dir = Path(__file__).resolve().parent
    records = []

    def execute(name, command, gpu=None):
        environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1',
                           TOKENIZERS_PARALLELISM='false')
        if gpu is not None:
            environment['CUDA_VISIBLE_DEVICES'] = gpu
        started = time.time()
        with (args.output / 'logs' / (name + '.log')).open('x') as handle:
            result = subprocess.run(command, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        record = {'name': name, 'command': command, 'gpu': gpu, 'returncode': result.returncode,
                  'seconds': time.time() - started}
        records.append(record)
        print(json.dumps(record), flush=True)
        if result.returncode:
            raise RuntimeError(f'{name} failed; inspect its log')

    def evaluate(name, checkpoint, gpu):
        command = [py, str(script_dir / 'unified_game_pipeline.py'), 'rollout',
                   '--cases', protocol['case_file'], '--output', str(args.output / (name + '_fresh.jsonl')),
                   '--engine', 'checkpoint' if checkpoint else 'random',
                   '--controller', 'greedy', '--epsilon', str(protocol['evaluation']['epsilon']),
                   '--seed', str(protocol['evaluation']['sampling_seed']), '--splits', 'test,ood',
                   '--env-batch', '16', '--batch-questions', '16', '--max-length', str(cfg['max_length'])]
        if checkpoint:
            command.extend(['--checkpoint', str(checkpoint)])
        execute(name + '_fresh', command, gpu)
        execute(name + '_fresh_replay', [py, str(script_dir / 'replay_unified_episodes.py'),
                '--episodes', str(args.output / (name + '_fresh.jsonl')),
                '--output', str(args.output / (name + '_fresh_replay.json'))])
        if checkpoint:
            regression = list(command)
            regression[regression.index('--cases') + 1] = str(args.output / 'regression_cases.jsonl')
            regression[regression.index('--output') + 1] = str(args.output / (name + '_regression.jsonl'))
            regression[regression.index('--epsilon') + 1] = '0.15'
            execute(name + '_regression', regression, gpu)
            execute(name + '_regression_replay', [py, str(script_dir / 'replay_unified_episodes.py'),
                    '--episodes', str(args.output / (name + '_regression.jsonl')),
                    '--output', str(args.output / (name + '_regression_replay.json'))])

    # Retain the exact original regression cases; only Basic has a new cohort.
    old_cases = [json.loads(line) for line in (script_dir.parent / 'configs/unified_games_v1_cases.jsonl').read_text().splitlines() if line.strip()]
    regression = [case for case in old_cases if case['split'] in ('test', 'ood') and case['spec'].get('scenario') != 'basic']
    (args.output / 'regression_cases.jsonl').write_text(''.join(json.dumps(case, sort_keys=True) + '\n' for case in regression))

    def train(arm, gpu):
        name = arm['name']
        checkpoint = args.output / name
        target = 'hard' if arm['target'] == 'expert_argmax' else 'soft'
        command = [py, str(script_dir / 'train_unified_games.py'), '--stage', 'sft', '--loss', 'ce',
                   '--input', str(args.datasets / target), '--init-checkpoint', str(args.init_checkpoint),
                   '--output-dir', str(checkpoint), '--policy-pool-weights', str(weights), '--seed', str(arm['seed']),
                   '--precision', cfg['precision'], '--disable-native-triton']
        for key in ('steps', 'head_steps', 'batch_questions', 'microbatch_questions', 'max_microbatch_tokens',
                    'max_length', 'eval_every', 'backbone_lr', 'head_lr', 'weight_decay'):
            command.extend(['--' + key.replace('_', '-'), str(cfg[key])])
        if cfg['gradient_checkpointing']:
            command.append('--gradient-checkpointing')
        execute(name + '_train', command, gpu)
        evaluate(name, checkpoint, gpu)

    manifest = {'protocol_sha256': protocol_sha256, 'protocol': protocol,
                'init_weights_sha256': protocol['initialization']['weights_sha256'],
                'finished': False, 'expected_jobs': expected_jobs,
                'checkpoint_uploads': False, 'new_api_calls': False}
    path = args.output / 'experiment.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    errors, aborted = [], False
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus) + 1) as pool:
            futures = [pool.submit(train, arm, gpu) for arm, gpu in zip(cfg['arms'], gpus)]
            futures.extend([pool.submit(evaluate, 'frozen_sft', args.init_checkpoint, gpus[-1]),
                            pool.submit(evaluate, 'random', None, None)])
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    errors.append(str(error))
    except BaseException as error:
        aborted = True
        errors.append(f'{type(error).__name__}: {error}')
        raise
    finally:
        manifest.update(completion_status(expected_jobs, records, errors, aborted), jobs=records, errors=errors)
        path.write_text(json.dumps(manifest, indent=2) + '\n')
    if not manifest['finished']:
        raise SystemExit('; '.join(errors) or 'Expected job set did not complete successfully')


if __name__ == '__main__':
    main()

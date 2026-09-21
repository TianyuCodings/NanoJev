#!/usr/bin/env python3
"""Real BTC snapshots: joint 1/5/15-minute observed direction questions, no synthetic data."""
import argparse
from bisect import bisect_left
from collections import Counter
import csv
from datetime import datetime, timezone
import io
import math
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import digest, dump, write_splits

SOURCE = 'https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data'
FIELDS = ('midpoint', 'spread') + tuple(f'{s}_notional_{i}' for s in ('bids', 'asks') for i in range(5))


def utc(text):
    value = datetime.fromisoformat(text.replace('Z', '+00:00'))
    if value.tzinfo is None:
        raise ValueError('Timestamp must explicitly specify a timezone')
    return value.astimezone(timezone.utc).timestamp()


def iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec='microseconds')


def read_snapshots(path):
    path = Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if Path(n).name == 'BTC_1min.csv']
            if len(names) != 1:
                raise ValueError('Archive must contain exactly one BTC_1min.csv')
            with archive.open(names[0]) as f:
                return parse_csv(io.TextIOWrapper(f, encoding='utf-8-sig'))
    with path.open(encoding='utf-8-sig', newline='') as f:
        return parse_csv(f)


def parse_csv(stream):
    reader = csv.DictReader(stream)
    if not set(('system_time',) + FIELDS) <= set(reader.fieldnames or ()):
        raise ValueError('Missing required raw snapshot fields')
    rows = []
    previous = float('-inf')
    for line, row in enumerate(reader, 2):
        t = utc(row['system_time'])
        if t <= previous:
            raise ValueError(f'Duplicate/out-of-order timestamp at CSV line {line}; no silent sorting')
        values = {k: float(row[k]) for k in FIELDS}
        if not all(math.isfinite(x) for x in values.values()):
            raise ValueError(f'Nonfinite input at CSV line {line}')
        if values['midpoint'] <= 0 or values['spread'] < 0 or values['spread'] >= 2 * values['midpoint']:
            raise ValueError(f'Invalid midpoint/spread at CSV line {line}')
        if any(values[k] < 0 for k in FIELDS if 'notional' in k):
            raise ValueError(f'Negative book notional at CSV line {line}')
        rows.append({'t': t, 'line': line, **values})
        previous = t
    if not rows:
        raise ValueError('Empty source data')
    return rows


def validate_config(config):
    keys = ('availability_lag_seconds', 'target_tolerance_seconds',
            'max_gap_seconds', 'embargo_seconds', 'flat_threshold_bps')
    for key in keys:
        v = config[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f'Invalid {key}')
    horizons = config['horizons_seconds']
    if (not isinstance(horizons, list) or not horizons or
            any(type(h) is not int or h <= 0 for h in horizons) or horizons != sorted(set(horizons))):
        raise ValueError('Horizons must be unique increasing positive integer seconds')
    if config['max_gap_seconds'] <= 0:
        raise ValueError('Max gap must be positive')
    if config['availability_lag_seconds'] < 60:
        raise ValueError('This adapter conservatively requires a full 60-second availability lag')
    if type(config['lookback_rows']) is not int or config['lookback_rows'] < 1:
        raise ValueError('lookback_rows must be a positive integer')
    if [s['name'] for s in config['splits']] != ['train', 'dev', 'calibration', 'test', 'ood']:
        raise ValueError('Expected chronological train/dev/calibration/test/ood blocks')
    previous = float('-inf')
    for split in config['splits']:
        lo, hi = utc(split['start']), utc(split['end'])
        if lo < previous or hi <= lo:
            raise ValueError('Invalid/overlapping split boundaries')
        if lo % 86400 or hi % 86400:
            raise ValueError('Day-group splits must use UTC midnight boundaries')
        previous = hi


def classify(return_bps, threshold):
    # Tolerance only protects floating-point representation at exact boundaries.
    return 'up' if return_bps > threshold + 1e-9 else 'down' if return_bps < -threshold - 1e-9 else 'flat'


def input_state(history, decision_time, availability_lag=60):
    """Only historical input: neither target rows nor labels are accepted here."""
    last = history[-1]
    mid = last['midpoint']
    spread_bps = last['spread'] / mid * 10000
    bid = sum(last[f'bids_notional_{i}'] for i in range(5))
    ask = sum(last[f'asks_notional_{i}'] for i in range(5))
    imbalance = (bid - ask) / (bid + ask) if bid + ask else 0.0
    # All values and normalizations use the last completed snapshot/history only.
    lookbacks = sorted({1, 4, 9, len(history) - 1})
    past_returns = ', '.join(
        f'{last["t"] - history[-1-offset]["t"]:.3f}s:{(mid / history[-1-offset]["midpoint"] - 1) * 10000:.6f}bps'
        for offset in lookbacks if 0 < offset < len(history))
    return (f'Coinbase BTC order-book snapshot series. Decision time UTC: {iso(decision_time)}. '
            f'Provider 1-minute rows become available {availability_lag:g}s after their system timestamp. '
            'Only completed historical snapshots are used. '
            f'Latest completed midpoint: {mid:.8f}; spread: {last["spread"]:.8f} ({spread_bps:.6f} bps). '
            f'Top-five bid/ask book-notional imbalance: {imbalance:.8f}. '
            f'Past {len(history)} rows: latest-versus-past midpoint returns (age:change) [{past_returns}].')


def construct(raw, config, raw_sha):
    validate_config(config)
    times = [r['t'] for r in raw]
    lag, horizons = config['availability_lag_seconds'], config['horizons_seconds']
    lookback, tolerance = config['lookback_rows'], config['target_tolerance_seconds']
    threshold = config['flat_threshold_bps']
    blocks = [(s['name'], utc(s['start']), utc(s['end'])) for s in config['splits']]
    rows, dropped = [], Counter()
    for i, now in enumerate(raw):
        decision = now['t'] + lag
        block = next((b for b in blocks if b[1] <= decision < b[2]), None)
        if block is None:
            dropped['outside_declared_splits'] += 1
            continue
        split, lo, hi = block
        if i < lookback - 1:
            dropped['insufficient_history'] += 1
            continue
        # Search by clock time, never by price, outcome, or a hindsight-optimal horizon.
        targets = {h: bisect_left(times, now['t'] + h, lo=i + 1) for h in horizons}
        if any(j >= len(raw) or times[j] - times[i] > h + tolerance for h, j in targets.items()):
            dropped['missing_target_in_tolerance'] += 1
            continue
        j = max(targets.values())
        start = i - lookback + 1
        support_start, support_end = times[start], times[j] + lag
        if support_start < lo + config['embargo_seconds'] or support_end >= hi - config['embargo_seconds']:
            dropped['purged_boundary_or_embargo'] += 1
            continue
        if any(times[k] - times[k - 1] > config['max_gap_seconds'] for k in range(start + 1, j + 1)):
            dropped['gap_in_feature_or_label_window'] += 1
            continue
        source_day = datetime.fromtimestamp(now['t'], timezone.utc).date().isoformat()
        sid = f'coinbase:BTC:1min:{int(round(now["t"] * 1000000))}'
        questions, gold, audits = {}, {}, {}
        for horizon, target_index in targets.items():
            future = raw[target_index]
            ret = (future['midpoint'] / now['midpoint'] - 1) * 10000
            qid = f'midpoint_direction_{horizon}s'
            questions[qid] = {'type': 'choice',
                'instructions': (f'Predict the direction of the first completed provider 1-minute midpoint snapshot '
                    f'available at least {horizon:g} seconds after the decision time (maximum extra wait {tolerance:g} seconds), '
                    f'relative to the latest completed midpoint in the state. Use a symmetric {threshold:g} basis-point flat band.'),
                'criteria': {'down': f'Return strictly below -{threshold:g} bps.',
                             'flat': f'Return between -{threshold:g} and +{threshold:g} bps, inclusive.',
                             'up': f'Return strictly above +{threshold:g} bps.'}}
            gold[qid] = classify(ret, threshold)
            audits[qid] = {'label_csv_line': future['line'], 'label_source_time': iso(future['t']),
                'label_available_at': iso(future['t'] + lag), 'horizon_seconds': horizon,
                'realized_horizon_seconds': future['t'] - now['t'], 'reference_midpoint': now['midpoint'],
                'future_midpoint': future['midpoint'], 'return_bps': ret, 'flat_threshold_bps': threshold}
        rows.append({'id': sid + ':multi_v2', 'state_id': sid,
            'family_id': 'coinbase_btc_multi_horizon_direction_v2', 'split': split,
            'state': input_state(raw[start:i + 1], decision, lag), 'questions': questions,
            'gold': gold, 'gold_probs': {}, 'gold_probs_kind': {},
            'gold_label_kind': {qid: 'observed_outcome' for qid in questions},
            'metadata': {'source': SOURCE, 'license': 'CC0-1.0 (publisher declaration)',
                'source_group_id': f'coinbase:utc-day:{source_day}',
                'template_id': 'trading:multi_horizon:v2', 'source_sha256': raw_sha,
                'source_csv_line': now['line'], 'feature_first_csv_line': raw[start]['line'],
                'decision_time': iso(decision),
                'latest_source_time': iso(now['t']), 'feature_support_start': iso(support_start),
                'feature_available_at': iso(decision), 'label_available_at': iso(support_end),
                'support_end': iso(support_end), 'label_audit': audits,
                'target_semantics': 'single_observed_outcome; conditional_distribution_unknown',
                'ood_definition': 'last_chronological_block; distribution_shift_not_proven' if split == 'ood' else None}})
    return rows, dict(dropped)


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, default=root / 'raw/BTC_1min.csv.zip')
    p.add_argument('--config', type=Path, default=root / 'config.json')
    p.add_argument('--output', type=Path, default=root / 'data')
    a = p.parse_args()
    import json
    config = json.loads(a.config.read_text())
    raw = read_snapshots(a.raw)
    rows, dropped = construct(raw, config, digest(a.raw))
    stats = write_splits(rows, a.output)
    if any(v['rows'] == 0 for v in stats.values()):
        raise ValueError('A required split is empty; review coverage/config')
    dump(a.output / 'manifest.json', {'adapter': 'coinbase-kaggle-multi-horizon-v2',
        'raw_sha256': digest(a.raw), 'raw_rows': len(raw), 'raw_first_time': iso(raw[0]['t']),
        'raw_last_time': iso(raw[-1]['t']), 'source': SOURCE,
        'configuration': config, 'dropped': dropped, 'splits': stats,
        'label_counts': {s: {qid: dict(Counter(r['gold'][qid] for r in rows if r['split'] == s))
                           for qid in rows[0]['questions']} for s in stats},
        'unique_decision_times': len(rows), 'horizons_are_correlated_questions_not_independent_samples': True,
        'no_imputation': True, 'no_fitted_scaler': True, 'no_synthetic_rows': True})
    print(stats)


if __name__ == '__main__':
    main()

"""Shared deterministic I/O. Python 3.10+, standard library only."""
import hashlib
import json
from pathlib import Path

SPLITS = ('train', 'dev', 'calibration', 'test', 'ood')
ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + '\n', encoding='utf-8')


def write_splits(rows, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for split in SPLITS:
        subset = [r for r in rows if r['split'] == split]
        path = directory / (split + '.jsonl')
        path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False,
                                          separators=(',', ':')) + '\n' for r in subset), encoding='utf-8')
        result[split] = {'rows': len(subset), 'questions': sum(len(r['questions']) for r in subset),
                         'sha256': digest(path)}
    return result

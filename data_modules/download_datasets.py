#!/usr/bin/env python3
"""Download optional NanoJev-native records from pinned public HF snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent
SPLITS = ('train', 'dev', 'calibration', 'test', 'ood')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch(url, path, expected):
    if path.exists():
        if digest(path) != expected:
            raise ValueError(f'Existing file differs; refusing to overwrite: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'NanoJev-optional-datasets/2'})
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.part', delete=False) as handle:
            temporary = Path(handle.name)
            with urllib.request.urlopen(request, timeout=90) as response:
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    handle.write(block)
        if digest(temporary) != expected:
            raise ValueError(f'Download checksum mismatch: {path.name}')
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', choices=('all', 'blackjack', 'trading'), default='all')
    parser.add_argument('--include-raw', action='store_true', help='Include original CC0 BTC ZIP for reproduction/tests')
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    registry = json.loads((ROOT / 'datasets.json').read_text())
    for module, config in registry.items():
        if args.module not in ('all', module):
            continue
        base = f"https://huggingface.co/datasets/{config['repo_id']}/resolve/{config['revision']}/"
        with tempfile.TemporaryDirectory(prefix='nanojev-manifest-') as tmp:
            manifest_path = Path(tmp) / 'SHA256SUMS.json'
            fetch(base + 'SHA256SUMS.json', manifest_path, config['manifest_sha256'])
            manifest = json.loads(manifest_path.read_text())
        names = [f'{module}/data/{split}.jsonl' for split in SPLITS]
        names.append(f'{module}/data/manifest.json')
        if module == 'trading' and args.include_raw:
            names.append('trading/raw/BTC_1min.csv.zip')
        # Only fixed known paths are selected; remote manifest keys are not output paths.
        for name in names:
            fetch(base + name, args.output / name, manifest[name])
            print('Verified', name, flush=True)


if __name__ == '__main__':
    main()

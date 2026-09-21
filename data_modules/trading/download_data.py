#!/usr/bin/env python3
"""Download the publisher's publicly accessible CC0 data; no API key or paid feed."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import digest

URL = 'https://www.kaggle.com/api/v1/datasets/download/martinsn/high-frequency-crypto-limit-order-book-data/BTC_1min.csv?datasetVersionNumber=1'


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=root / 'raw/BTC_1min.csv.zip')
    a = p.parse_args()
    expected = json.loads((root / 'provenance.json').read_text())
    if a.output.exists():
        if digest(a.output) != expected['download_sha256']:
            raise ValueError('Existing file differs; refusing to overwrite it')
        print('Verified existing real-data archive:', a.output)
        return
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.output.with_suffix(a.output.suffix + '.part')
    request = urllib.request.Request(URL, headers={'User-Agent': 'NanoJev-real-data-adapter/1.0'})
    with urllib.request.urlopen(request, timeout=90) as response, temporary.open('wb') as f:
        shutil.copyfileobj(response, f)
    if not zipfile.is_zipfile(temporary):
        raise ValueError('Source did not return ZIP data (possibly access policy changed)')
    with zipfile.ZipFile(temporary) as z:
        names = [n for n in z.namelist() if Path(n).name == 'BTC_1min.csv']
        if len(names) != 1:
            raise ValueError('Unexpected archive contents')
        import hashlib
        h = hashlib.sha256()
        with z.open(names[0]) as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != expected['csv_sha256']:
            raise ValueError('Downloaded source CSV differs from pinned v1; review before using')
    # Also pin transport bytes. A differently repacked ZIP needs explicit provenance review.
    if digest(temporary) != expected['download_sha256']:
        raise ValueError('CSV matches, but ZIP packaging differs; review provenance before replacing')
    temporary.replace(a.output)
    print('Downloaded and verified:', a.output)


if __name__ == '__main__':
    main()

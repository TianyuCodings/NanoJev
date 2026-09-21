"""Pinned downloads reject corruption and preserve existing local files."""
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from download_datasets import fetch


class DownloadTests(unittest.TestCase):
    def test_verified_download_and_cached_reuse(self):
        payload = b'one native record\n'
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'data/train.jsonl'
            with patch('download_datasets.urllib.request.urlopen', return_value=io.BytesIO(payload)) as request:
                fetch('https://example.test/train.jsonl', target, expected)
                fetch('https://example.test/train.jsonl', target, expected)
            self.assertEqual(target.read_bytes(), payload)
            request.assert_called_once()
            self.assertEqual(list(target.parent.glob('*.part')), [])

    def test_corruption_never_becomes_a_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'data/train.jsonl'
            with patch('download_datasets.urllib.request.urlopen', return_value=io.BytesIO(b'bad')):
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    fetch('https://example.test/train.jsonl', target, '0' * 64)
            self.assertFalse(target.exists())
            self.assertEqual(list(target.parent.glob('*.part')), [])

    def test_existing_changed_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'train.jsonl'
            target.write_bytes(b'user experiment')
            with patch('download_datasets.urllib.request.urlopen') as request:
                with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
                    fetch('https://example.test/train.jsonl', target, '0' * 64)
                request.assert_not_called()
            self.assertEqual(target.read_bytes(), b'user experiment')


if __name__ == '__main__':
    unittest.main()

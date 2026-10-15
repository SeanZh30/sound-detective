"""Download only our six ESC-50 classes from a pinned upstream revision."""
import csv
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
import time
from training.features import CLASSES

ROOT = Path(__file__).resolve().parents[1]
REVISION = '33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6'
BASE = f'https://raw.githubusercontent.com/karolpiczak/ESC-50/{REVISION}'

def download(url):
    for attempt in range(4):
        try:
            with urlopen(url, timeout=45) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)

def main():
    directory = ROOT / 'data/raw/audio'
    directory.mkdir(parents=True, exist_ok=True)
    metadata = download(f'{BASE}/meta/esc50.csv').decode()
    rows = [r for r in csv.DictReader(io.StringIO(metadata)) if r['category'] in CLASSES]
    assert len(rows) == 240
    def save(row):
        path = directory / row['filename']
        if not path.exists():
            content = download(f'{BASE}/audio/{row["filename"]}')
            if content[:4] != b'RIFF':
                raise ValueError(f'Invalid WAV: {path.name}')
            temporary = path.with_suffix('.partial')
            temporary.write_bytes(content)
            temporary.replace(path)
        return {**row, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    with ThreadPoolExecutor(max_workers=8) as executor:
        manifest = list(executor.map(save, rows))
    manifest_path = ROOT / 'data/raw/manifest.json'
    temporary = manifest_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(manifest_path)
    (ROOT / 'training/results/data-provenance.json').write_text(json.dumps({
        'dataset': 'ESC-50 six-class subset', 'revision': REVISION,
        'source': BASE, 'recordings': manifest,
    }, indent=2) + '\n')
    print(f'Ready: {len(rows)} WAVs, revision {REVISION}', flush=True)

if __name__ == '__main__':
    main()

"""Restore the 18 public ESC-50 demonstration WAVs into data/samples/audio."""
import json
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://raw.githubusercontent.com/karolpiczak/ESC-50/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6/audio/'
for manifest in ['dataset.json', 'test-clips.json']:
    for item in json.loads((ROOT / 'data/samples' / manifest).read_text()):
        target = ROOT / 'data/samples' / item['file']
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            urlretrieve(BASE + target.name, target)
        print(target.name, target.stat().st_size)

"""Verified original UrbanSound8K download and shared global/segment features."""
import csv
import hashlib
import json
import os
from pathlib import Path
import tarfile
from concurrent.futures import ProcessPoolExecutor
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/raw/urbansound8k'
CACHE = ROOT / 'data/cache/hw3-features.npz'
OUT = ROOT / 'training/hw3/results'
URL = 'https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz?download=1'
MD5 = '9aa69802bbf37fb986f71ec1483a196e'
VERSION = 'us8k-22050-5s-logmel64-mfcc20-global-four-ordered-v1'

def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def extract(audio, sr):
    import librosa
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Invalid waveform')
    # Unlike the interactive HW2 upload validator, retain every official short clip.
    audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
    audio = librosa.util.fix_length(audio, size=110250)
    mel = librosa.feature.melspectrogram(y=audio, sr=22050, n_fft=1024, hop_length=256, n_mels=64)
    logmel = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=20)
    def stats(m, c):
        return np.concatenate([m.mean(1), m.std(1), c.mean(1), c.std(1)])
    indices = np.array_split(np.arange(logmel.shape[1]), 4)
    return np.concatenate([stats(logmel, mfcc)] + [stats(logmel[:, i], mfcc[:, i]) for i in indices]).astype('float32')

def one(row):
    import soundfile as sf
    path = DATA / 'UrbanSound8K/audio' / ('fold' + row['fold']) / row['slice_file_name']
    audio, sr = sf.read(path)
    return extract(audio, sr), digest(path), len(audio) / sr

def main():
    DATA.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    archive = DATA / 'UrbanSound8K.tar.gz'
    if not archive.exists():
        partial = archive.with_suffix('.gz.partial')
        if not partial.exists():
            urllib.request.urlretrieve(URL, partial)
        if digest(partial, 'md5') != MD5:
            raise ValueError('Incomplete/corrupt archive: remove partial and rerun download')
        partial.replace(archive)
    if digest(archive, 'md5') != MD5:
        raise ValueError('Archive checksum mismatch')
    marker = DATA / '.extracted'
    if not marker.exists():
        with tarfile.open(archive) as tar:
            # Only regular dataset files; reject traversal and symbolic links.
            for member in tar:
                if member.isfile() and member.name.startswith('UrbanSound8K/') and '..' not in Path(member.name).parts:
                    tar.extract(member, DATA, filter='data')
        marker.write_text(MD5)
    rows = list(csv.DictReader((DATA / 'UrbanSound8K/metadata/UrbanSound8K.csv').open()))
    rows.sort(key=lambda r: (int(r['fold']), r['slice_file_name']))
    assert len(rows) == 8732
    source_folds = {}
    for r in rows:
        source_folds.setdefault(r['fsID'], set()).add(r['fold'])
    assert all(len(v) == 1 for v in source_folds.values()), 'Source crosses folds'
    metadata_hash = digest(DATA / 'UrbanSound8K/metadata/UrbanSound8K.csv')
    if CACHE.exists():
        with np.load(CACHE) as c:
            assert c['version'].item() == VERSION and c['metadata_sha256'].item() == metadata_hash
        print('Verified existing feature cache', flush=True)
        return
    features, manifest = [], []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, (row, result) in enumerate(zip(rows, pool.map(one, rows, chunksize=16))):
            x, sha, duration = result
            features.append(x); manifest.append({**row, 'sha256': sha, 'duration': duration})
            if (i + 1) % 500 == 0:
                print(f'Features: {i+1}/8732', flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, X=np.stack(features), y=np.array([int(r['classID']) for r in rows]),
                        fold=np.array([int(r['fold']) for r in rows]), version=VERSION,
                        metadata_sha256=metadata_hash, filenames=np.array([r['slice_file_name'] for r in rows]))
    (OUT / 'provenance.json').write_text(json.dumps({'url': URL, 'archive_md5': MD5,
        'metadata_sha256': metadata_hash, 'feature_version': VERSION, 'recordings': manifest}, indent=2)+'\n')
    print('Saved all 8732 features and per-file provenance', flush=True)

if __name__ == '__main__':
    main()

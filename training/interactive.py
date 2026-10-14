"""Isolated, real MLP training worker for the web UI (one process per run)."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from sklearn.preprocessing import StandardScaler
from training.features import CLASSES, FEATURE_VERSION, extract_features
from training.model import SoundMLP
from training.train import scores

ROOT = Path(__file__).resolve().parents[1]

def write_status(path, status):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(status, allow_nan=False) + '\n')
    temporary.replace(path)

def run(directory, root=ROOT):
    path = directory / 'status.json'
    status = json.loads(path.read_text())
    def publish(**updates):
        status.update(updates)
        write_status(path, status)
    try:
        torch.set_num_threads(2)
        torch.manual_seed(7400)
        config = status['config']
        publish(status='preparing', message='Loading training data and features…')
        manifest = root / 'data/raw/manifest.json'
        if not manifest.exists():
            publish(message='Downloading the public 240-clip training subset (first run only)…')
            from training.prepare import main as prepare
            prepare()
        rows = json.loads(manifest.read_text())
        names = [r['filename'] for r in rows]
        cache = root / 'data/cache/features.npz'
        X = None
        if cache.exists():
            try:
                with np.load(cache) as saved:
                    if saved['version'].item() == FEATURE_VERSION and saved['filenames'].tolist() == names:
                        X = saved['X']
            except (ValueError, OSError, EOFError, KeyError):
                X = None
        if X is None:
            values = []
            for i, row in enumerate(rows):
                values.append(extract_features(*sf.read(root / 'data/raw/audio' / row['filename'])))
                publish(message=f'Extracting audio features: {i + 1}/{len(rows)}')
            X = np.stack(values)
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_name('features.partial.npz')
            np.savez(temporary, X=X, filenames=names, version=FEATURE_VERSION)
            temporary.replace(cache)
        masks = {name: np.array([int(r['fold']) in folds for r in rows]) for name, folds in [('train', [1, 2, 3]), ('validation', [4]), ('test', [5])]}
        source_sets = [{r['src_file'] for r, keep in zip(rows, mask) if keep} for mask in masks.values()]
        assert not (source_sets[0] & source_sets[1] or source_sets[0] & source_sets[2] or source_sets[1] & source_sets[2])
        y = np.array([CLASSES.index(r['category']) for r in rows])
        scaler = StandardScaler().fit(X[masks['train']])
        xs = {name: torch.tensor(scaler.transform(X[mask]), dtype=torch.float32) for name, mask in masks.items()}
        ys = {name: torch.tensor(y[mask]) for name, mask in masks.items()}
        model = SoundMLP(config['hidden'], config['dropout'])
        optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        best, best_loss = None, float('inf')
        publish(status='running', message='Training MLP on folds 1–3.', counts={n: int(m.sum()) for n, m in masks.items()})
        for epoch in range(1, config['epochs'] + 1):
            model.train()
            for index in torch.randperm(len(xs['train'])).split(24):
                optimizer.zero_grad()
                loss = criterion(model(xs['train'][index]), ys['train'][index])
                if not torch.isfinite(loss):
                    raise ValueError('Training diverged. Try a smaller learning rate.')
                loss.backward()
                optimizer.step()
            model.eval()
            entry = {'epoch': epoch}
            with torch.no_grad():
                for name in ['train', 'validation']:
                    logits = model(xs[name])
                    entry[f'{name}_loss'] = float(criterion(logits, ys[name]))
                    entry[f'{name}_accuracy'] = float((logits.argmax(1) == ys[name]).float().mean())
            if not all(np.isfinite(v) for v in entry.values()):
                raise ValueError('Training diverged. Try a smaller learning rate.')
            if entry['validation_loss'] < best_loss:
                best_loss = entry['validation_loss']
                best = copy.deepcopy(model.state_dict())
                status['best_epoch'] = epoch
            status['history'].append(entry)
            publish(epoch=epoch, message=f'Epoch {epoch}/{config["epochs"]}')
        model.load_state_dict(best)
        model.eval()
        publish(status='evaluating', message='Evaluating the validation-selected checkpoint on the test fold…')
        metrics = {}
        with torch.no_grad():
            for name in masks:
                logits = model(xs[name])
                metrics[name] = {**scores(ys[name], logits.argmax(1)), 'loss': float(criterion(logits, ys[name]))}
        model_path = root / 'backend/model/runs' / (status['id'] + '.pt')
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': best, 'hidden': config['hidden'], 'dropout': config['dropout'],
                    'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'classes': CLASSES,
                    'feature_version': FEATURE_VERSION}, model_path)
        publish(status='completed', message='Training complete. Best validation checkpoint is ready for predictions.', metrics=metrics)
    except Exception as error:
        publish(status='failed', message=f'Training failed: {error}')
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.directory, args.root)

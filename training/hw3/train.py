"""Paired, predeclared 10-fold experiment; test never participates in selection."""
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
from training.hw3.model import TemporalMLP
from training.hw3.prepare import CACHE, VERSION

HERE = Path(__file__).resolve().parent

def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        logits, gate = model(x, return_gate=True)
        loss = torch.nn.functional.cross_entropy(logits, y).item()
        probabilities = logits.softmax(1).numpy()
    pred = probabilities.argmax(1)
    return {'loss': loss, 'accuracy': float((pred == y.numpy()).mean()),
            'macro_f1': float(f1_score(y.numpy(), pred, labels=range(10), average='macro', zero_division=0)),
            'confusion_matrix': confusion_matrix(y.numpy(), pred, labels=range(10)).tolist()}, probabilities, gate.numpy().ravel()

def main():
    cfg = json.loads((HERE / 'config.json').read_text())
    config_hash = hashlib.sha256((HERE / 'config.json').read_bytes()).hexdigest()
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    saved = np.load(CACHE)
    assert saved['version'].item() == VERSION
    X, y, folds = saved['X'], saved['y'], saved['fold']
    (HERE / 'checkpoints').mkdir(exist_ok=True)
    for fold in cfg['test_folds']:
        val = fold % 10 + 1
        masks = {'train': (folds != fold) & (folds != val), 'validation': folds == val, 'test': folds == fold}
        scaler = StandardScaler().fit(X[masks['train']])
        xs = {k: torch.tensor(scaler.transform(X[m]), dtype=torch.float32) for k, m in masks.items()}
        ys = {k: torch.tensor(y[m], dtype=torch.long) for k, m in masks.items()}
        for variant in cfg['models']:
            target = HERE / 'results' / f'{variant}-fold{fold}.json'
            if target.exists():
                assert json.loads(target.read_text())['config_sha256'] == config_hash
                continue
            seed = cfg['seed'] + fold
            torch.manual_seed(seed)
            model = TemporalMLP(variant, cfg['hidden'], cfg['dropout'])
            # Fixed and gated share exactly the same initial branch/head weights.
            if variant == 'gated':
                torch.manual_seed(seed)
                reference = TemporalMLP('fixed', cfg['hidden'], cfg['dropout'])
                model.load_state_dict(reference.state_dict(), strict=False)
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg['learning_rate'],
                momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
            permutation = torch.Generator().manual_seed(seed + 10000)
            torch.manual_seed(seed + 20000)  # Paired dropout RNG as well as batches.
            best_loss, history = float('inf'), []
            start = time.perf_counter()
            for epoch in range(1, cfg['epochs'] + 1):
                model.train()
                for idx in torch.randperm(len(ys['train']), generator=permutation).split(cfg['batch_size']):
                    optimizer.zero_grad()
                    logits = model(xs['train'][idx])
                    loss = torch.nn.functional.cross_entropy(logits, ys['train'][idx])
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Nonfinite loss: {variant} fold {fold} epoch {epoch}')
                    loss.backward()
                    optimizer.step()
                entry = {'epoch': epoch}
                for name in ['train', 'validation']:
                    metrics, _, _ = evaluate(model, xs[name], ys[name])
                    for key in ['loss', 'accuracy', 'macro_f1']:
                        entry[name + '_' + key] = metrics[key]
                history.append(entry)
                if entry['validation_loss'] < best_loss:
                    best_loss = entry['validation_loss']; best_epoch = epoch
                    best = copy.deepcopy(model.state_dict())
            seconds = time.perf_counter() - start
            model.load_state_dict(best)
            report = {'variant': variant, 'test_fold': fold, 'validation_fold': val, 'seed': seed,
                      'config_sha256': config_hash, 'counts': {k: int(m.sum()) for k,m in masks.items()},
                      'best_epoch': best_epoch, 'seconds': seconds, 'parameters': sum(p.numel() for p in model.parameters()),
                      'history': history, 'metrics': {}}
            for name in masks:
                metrics, probabilities, gate = evaluate(model, xs[name], ys[name])
                report['metrics'][name] = metrics
                if name == 'test':
                    report['test_predictions'] = [{'filename': str(filename), 'actual': int(label),
                        'predicted': int(p.argmax()), 'probabilities': p.tolist(), 'local_gate': float(a)}
                        for filename,label,p,a in zip(saved['filenames'][masks[name]], y[masks[name]], probabilities, gate)]
            # Diagnostic only: reverse local segments at test time, never select on it.
            if variant != 'baseline':
                raw = X[masks['test']].copy()
                raw[:,168:] = raw[:,168:].reshape(-1,4,168)[:,::-1].reshape(-1,672)
                reverse_x = torch.tensor(scaler.transform(raw), dtype=torch.float32)
                report['reversed_segments_test'] = evaluate(model, reverse_x, ys['test'])[0]
            torch.save({'state_dict': best, 'variant': variant, 'hidden': cfg['hidden'], 'dropout': cfg['dropout'],
                'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'feature_version': VERSION,
                'test_fold': fold, 'validation_fold': val, 'config_sha256': config_hash},
                HERE / 'checkpoints' / f'{variant}-fold{fold}.pt')
            target.write_text(json.dumps(report, indent=2)+'\n')
            print(f'{variant} fold={fold} selected={best_epoch} test={report["metrics"]["test"]["accuracy"]:.4f} seconds={seconds:.1f}', flush=True)

if __name__ == '__main__':
    main()

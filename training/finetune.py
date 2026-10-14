"""Partial fine-tuning of AudioSet-pretrained Cnn6; mirrored in notebook 02."""
import copy
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from training.features import CLASSES, LABELS
from training.pretrained import load_audioset, waveform32k, PRETRAIN_VERSION, PRETRAIN_MD5, PRETRAIN_URL
from training.train import scores

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts'
SEED = 7400

def main():
    # SECTION: setup
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(False)  # MPS kernels can vary across systems.
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = load_audioset().to(device)
    initial_tail = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k.startswith(('conv_block4.', 'fc1.'))}
    rows = json.loads((ROOT / 'data/manifest.json').read_text())
    y = torch.tensor([CLASSES.index(r['category']) for r in rows])
    masks = {name: np.array([int(r['fold']) in folds for r in rows]) for name, folds in [('train', [1, 2, 3]), ('validation', [4]), ('test', [5])]}
    source_sets = [{r['src_file'] for r, keep in zip(rows, masks[name]) if keep} for name in masks]
    assert not (source_sets[0] & source_sets[1] or source_sets[0] & source_sets[2] or source_sets[1] & source_sets[2])
    print('Device:', device, '| fine-tuned parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
    # SECTION: cache
    # The first three blocks and their BN statistics never change. Cache them once.
    # Test representations are extracted independently; no fitting or statistics use them.
    cache = ROOT / 'data/panns-prefix.pt'
    signature = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest() + PRETRAIN_MD5
    if cache.exists():
        saved = torch.load(cache, map_location='cpu', weights_only=True)
        assert saved['signature'] == signature and saved['version'] == PRETRAIN_VERSION
        features = saved['features']
    else:
        batches = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(rows), 8):
                waves = np.stack([waveform32k(*sf.read(ROOT / 'data/audio' / r['filename'])) for r in rows[start:start + 8]])
                batches.append(model.prefix(torch.from_numpy(waves).to(device)).cpu())
                print(f'Frozen-prefix extraction: {min(start + 8, len(rows))}/{len(rows)}', flush=True)
        features = torch.cat(batches)
        torch.save({'features': features, 'signature': signature, 'version': PRETRAIN_VERSION}, cache)
    xs = {name: features[mask] for name, mask in masks.items()}
    ys = {name: y[mask] for name, mask in masks.items()}
    # SECTION: train
    # Fixed in advance: one schedule, not selected by observed test performance.
    optimizer = torch.optim.AdamW([
        {'params': list(model.conv_block4.parameters()) + list(model.fc1.parameters()), 'lr': 1e-4},
        {'params': model.fc_audioset.parameters(), 'lr': 1e-3},
    ], weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    history, best, best_loss, stale = [], None, float('inf'), 0
    def evaluate(name):
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model.tail(batch.to(device)).cpu() for batch in xs[name].split(8)])
        return logits, float(criterion(logits, ys[name]))
    for epoch in range(1, 21):
        model.train()  # Dropout on; BN running statistics remain frozen.
        for index in torch.randperm(len(xs['train'])).split(8):
            optimizer.zero_grad()
            logits = model.tail(xs['train'][index].to(device))
            loss = criterion(logits, ys['train'][index].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            optimizer.step()
        entry = {'epoch': epoch}
        for name in ['train', 'validation']:
            logits, loss = evaluate(name)
            entry[f'{name}_loss'] = loss
            entry[f'{name}_accuracy'] = float((logits.argmax(1) == ys[name]).float().mean())
        history.append(entry)
        print(entry, flush=True)
        if entry['validation_loss'] < best_loss - 1e-6:
            best_loss, stale = entry['validation_loss'], 0
            best = {'epoch': epoch, 'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        else:
            stale += 1
        if stale >= 5:
            break
    # SECTION: evaluate
    # Restore validation-selected weights BEFORE evaluating the untouched test labels.
    model.load_state_dict(best['state_dict'])
    report = {'model': 'PANNs Cnn6 (AudioSet), partially fine-tuned', 'classes': CLASSES,
        'labels': LABELS, 'seed': SEED, 'device': str(device), 'feature_version': PRETRAIN_VERSION,
        'pretrained_source': PRETRAIN_URL, 'pretrained_md5': PRETRAIN_MD5,
        'trainable_layers': ['conv_block4', 'fc1', 'fc_audioset (new six-class head)'],
        'frozen': 'spectrogram/logmel, bn0, conv blocks 1–3, all BN running statistics',
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'config': {'epoch': best['epoch'], 'max_epochs': 20, 'patience': 5, 'batch_size': 8,
                   'backbone_lr': 1e-4, 'head_lr': 1e-3, 'weight_decay': 0.01, 'optimizer': 'AdamW'},
        'counts': {name: int(mask.sum()) for name, mask in masks.items()}, 'history': history,
        'selection': 'One fixed schedule; minimum fold-4 validation loss; fold 5 excluded from selection.',
        'metrics': {}, 'test_predictions': [],
        'pretrained_weight_changes': {k: float((best['state_dict'][k] - v).abs().max()) for k, v in initial_tail.items() if v.is_floating_point()}}
    assert report['pretrained_weight_changes']['conv_block4.conv1.weight'] > 0
    assert report['pretrained_weight_changes']['fc1.weight'] > 0
    for name in masks:
        logits, loss = evaluate(name)
        report['metrics'][name] = {**scores(ys[name], logits.argmax(1)), 'loss': loss}
        if name == 'test':
            for row, probs in zip([r for r in rows if int(r['fold']) == 5], logits.softmax(1).tolist()):
                report['test_predictions'].append({'filename': row['filename'], 'actual': row['category'],
                    'predicted': CLASSES[int(np.argmax(probs))], 'probabilities': probs})
    # SECTION: save
    torch.save({'state_dict': best['state_dict'], 'classes': CLASSES,
                'feature_version': PRETRAIN_VERSION}, OUT / 'sound-panns.pt')
    (OUT / 'pretrained-metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ['train', 'validation']:
        for ax, metric in zip(axes, ['loss', 'accuracy']):
            ax.plot([h['epoch'] for h in history], [h[f'{name}_{metric}'] for h in history], label=name)
            ax.set(xlabel='Epoch', ylabel=metric)
    for ax in axes:
        ax.axvline(best['epoch'], ls='--', color='gray', label='selected checkpoint'); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / 'pretrained-learning-curves.png', dpi=160); plt.close(fig)
    print(json.dumps(report['metrics'], indent=2))

if __name__ == '__main__':
    main()

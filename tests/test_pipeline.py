import io
import json
import numpy as np
import soundfile as sf
import torch
from fastapi.testclient import TestClient
from backend.app import app, ROOT
from training.features import CLASSES, extract_features
from training.model import SoundMLP

client = TestClient(app)

def test_real_inference_matches_saved_test_evaluation():
    report = json.loads((ROOT / 'training/results/metrics.json').read_text())
    clips = json.loads((ROOT / 'data/samples/test-clips.json').read_text())
    expected = {row['filename']: row for row in report['test_predictions']}
    for clip in clips:
        response = client.post('/api/predict/clip', json={'clip_id': clip['id']})
        assert response.status_code == 200
        result = response.json()
        assert result['split'] == 'test'
        probs = [p['probability'] for p in result['probabilities']]
        np.testing.assert_allclose(probs, expected[clip['id'] + '.wav']['probabilities'], atol=1e-6)
        assert abs(sum(probs) - 1) < 1e-6
        assert result['category'] in CLASSES

def test_upload_uses_audio_not_filename():
    clip = json.loads((ROOT / 'data/samples/test-clips.json').read_text())[0]
    data = (ROOT / 'data/samples' / clip['file']).read_bytes()
    a = client.post('/api/predict/upload', files={'file': ('dog.wav', data, 'audio/wav')})
    b = client.post('/api/predict/upload', files={'file': ('rain.wav', data, 'audio/wav')})
    assert a.status_code == b.status_code == 200
    assert a.json() == b.json()

def test_bad_uploads_and_path_traversal_are_rejected():
    assert client.post('/api/predict/clip', json={'clip_id': '../../README'}).status_code == 404
    assert client.post('/api/predict/upload', files={'file': ('bad.wav', b'not audio')}).status_code == 422
    assert client.post('/api/predict/upload', files={'file': ('big.wav', b'x' * (5 * 1024 * 1024 + 1))}).status_code == 413
    silent = io.BytesIO(); sf.write(silent, np.zeros(22050), 22050, format='WAV')
    assert client.post('/api/predict/upload', files={'file': ('silent.wav', silent.getvalue())}).status_code == 422

def test_manifest_has_disjoint_sources_and_balanced_splits():
    rows = json.loads((ROOT / 'training/results/data-provenance.json').read_text())['recordings']
    groups = [{r['src_file'] for r in rows if int(r['fold']) in folds} for folds in [{1, 2, 3}, {4}, {5}]]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    for fold in range(1, 6):
        for category in CLASSES:
            assert sum(int(r['fold']) == fold and r['category'] == category for r in rows) == 8

def test_backpropagation_updates_parameters_and_reduces_loss():
    torch.manual_seed(7400)
    model = SoundMLP(32, 0)
    x = torch.randn(12, 168); y = torch.arange(12) % 6
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = torch.nn.CrossEntropyLoss()
    initial = float(loss_fn(model(x), y).detach())
    for _ in range(30):
        optimizer.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); optimizer.step()
    assert float(loss_fn(model(x), y).detach()) < initial / 2

def test_finetuned_inference_matches_evaluation_and_weights_changed():
    report = json.loads((ROOT / 'training/results/pretrained-metrics.json').read_text())
    assert report['pretrained_weight_changes']['conv_block4.conv1.weight'] > 0
    assert report['pretrained_weight_changes']['fc1.weight'] > 0
    clips = json.loads((ROOT / 'data/samples/test-clips.json').read_text())
    expected = {row['filename']: row for row in report['test_predictions']}
    for clip in clips:
        response = client.post('/api/predict/clip', json={'clip_id': clip['id'], 'model': 'panns'})
        assert response.status_code == 200
        result = response.json()
        assert result['model'] == 'panns'
        probs = [p['probability'] for p in result['probabilities']]
        # Training used MPS and cached prefixes; serving runs the complete network on CPU.
        np.testing.assert_allclose(probs, expected[clip['id'] + '.wav']['probabilities'], atol=2e-4)
    data = (ROOT / 'data/samples' / clips[0]['file']).read_bytes()
    upload = client.post('/api/predict/upload', data={'model': 'panns'}, files={'file': ('unrelated.wav', data)})
    reference = client.post('/api/predict/clip', json={'clip_id': clips[0]['id'], 'model': 'panns'})
    assert upload.json()['probabilities'] == reference.json()['probabilities']
    assert client.post('/api/predict/clip', json={'clip_id': clips[0]['id'], 'model': 'unknown'}).status_code == 422

def test_pretrained_batchnorm_stays_frozen_during_training():
    from training.pretrained import Cnn6Transfer
    model = Cnn6Transfer()
    model.freeze_prefix(); model.train()
    assert model.training
    assert all(not m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    assert not model.conv_block3.conv1.weight.requires_grad
    assert model.conv_block4.conv1.weight.requires_grad

def test_sample_audio_and_attribution_are_served_from_data_directory():
    clip = json.loads((ROOT / 'data/samples/dataset.json').read_text())[0]
    response = client.get('/' + clip['file'])
    assert response.status_code == 200
    assert response.content == (ROOT / 'data/samples' / clip['file']).read_bytes()
    assert client.get('/audio/missing.wav').status_code == 404
    assert client.get('/ESC50-LICENSE.txt').content == (ROOT / 'data/samples/ESC50-LICENSE.txt').read_bytes()

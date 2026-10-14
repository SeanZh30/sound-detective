"""Local inference service. Run from the repository root with uvicorn."""
from functools import lru_cache
import io
import json
from pathlib import Path
from typing import Literal
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from training.features import CLASSES, LABELS, FEATURE_VERSION, extract_features
from training.model import SoundMLP

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title='Sound Detective', version='3.0.0')
MAX_BYTES = 5 * 1024 * 1024
torch.set_num_threads(2)
ModelName = Literal["mlp", "panns"]

@lru_cache
def load_model():
    path = ROOT / 'artifacts/sound-mlp.pt'
    if not path.exists():
        raise HTTPException(503, 'Model missing. Run python -m training.train first.')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['classes'] != CLASSES or checkpoint['feature_version'] != FEATURE_VERSION:
        raise HTTPException(503, 'Model and feature extractor versions do not match.')
    model = SoundMLP(checkpoint['hidden'], checkpoint['dropout'])
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    return model, np.array(checkpoint['mean'], dtype=np.float32), np.array(checkpoint['scale'], dtype=np.float32)

@lru_cache
def load_pretrained_model():
    from training.pretrained import Cnn6Transfer, PRETRAIN_VERSION
    path = ROOT / 'artifacts/sound-panns.pt'
    if not path.exists():
        raise HTTPException(503, 'Fine-tuned model missing. Run notebook 02 first.')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['classes'] != CLASSES or checkpoint['feature_version'] != PRETRAIN_VERSION:
        raise HTTPException(503, 'Pretrained model version does not match.')
    model = Cnn6Transfer()
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    return model

def predict(content, model_name='mlp'):
    if len(content) > MAX_BYTES:
        raise HTTPException(413, 'WAV must be smaller than 5 MB.')
    if content[:4] != b'RIFF' or content[8:12] != b'WAVE':
        raise HTTPException(422, 'Please provide a valid PCM WAV file.')
    try:
        with sf.SoundFile(io.BytesIO(content)) as audio:
            if audio.channels > 2 or not 8000 <= audio.samplerate <= 192000:
                raise ValueError('Use mono/stereo WAV audio at 8–192 kHz.')
            if not 0.25 <= len(audio) / audio.samplerate <= 10:
                raise ValueError('Use a WAV recording between 0.25 and 10 seconds.')
            samples, sr = audio.read(dtype='float32'), audio.samplerate
            if model_name == 'panns':
                from training.pretrained import waveform32k
                features = waveform32k(samples, sr)
            else:
                features = extract_features(samples, sr)
    except (ValueError, RuntimeError) as error:
        raise HTTPException(422, str(error)) from error
    with torch.inference_mode():
        if model_name == 'panns':
            logits = load_pretrained_model()(torch.from_numpy(features).unsqueeze(0))
        else:
            model, mean, scale = load_model()
            logits = model(torch.from_numpy((features - mean) / scale).unsqueeze(0))
        probabilities = logits.softmax(1)[0].tolist()
    index = int(np.argmax(probabilities))
    return {'model': model_name, 'category': CLASSES[index], 'label': LABELS[index],
            'probabilities': [{'category': c, 'label': label, 'probability': p}
                              for c, label, p in zip(CLASSES, LABELS, probabilities)]}

@app.get('/api/health')
def health():
    load_model()
    return {'status': 'ready', 'feature_version': FEATURE_VERSION}

@app.get('/api/metrics')
def metrics():
    path = ROOT / 'artifacts/metrics.json'
    if not path.exists():
        raise HTTPException(503, 'Training report not found.')
    report = json.loads(path.read_text())
    pretrained_path = ROOT / 'artifacts/pretrained-metrics.json'
    if pretrained_path.exists():
        report['pretrained'] = json.loads(pretrained_path.read_text())
    return report

class ClipRequest(BaseModel):
    clip_id: str
    model: ModelName = "mlp"

@app.post('/api/predict/clip')
def predict_clip(request: ClipRequest):
    clips = json.loads((ROOT / 'src/dataset.json').read_text())
    extra = ROOT / 'src/test-clips.json'
    if extra.exists():
        clips += json.loads(extra.read_text())
    clip = next((clip for clip in clips if clip['id'] == request.clip_id), None)
    if not clip:
        raise HTTPException(404, 'Unknown recording.')
    return {**predict((ROOT / 'public' / clip['file']).read_bytes(), request.model),
            'split': 'test' if int(clip['fold']) == 5 else 'training'}

@app.post('/api/predict/upload')
def predict_upload(file: UploadFile = File(...), model: ModelName = Form("mlp")):
    content = file.file.read(MAX_BYTES + 1)
    return {**predict(content, model), 'split': 'user upload (not part of the reported evaluation)'}

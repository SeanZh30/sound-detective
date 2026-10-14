"""PANNs Cnn6 adapted from qiuqiangkong/audioset_tagging_cnn (MIT).
Upstream revision d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4.
See third_party/PANNs-LICENSE. Original AudioSet weights: Zenodo 3987831.
Changes: six-way logits; explicit frozen prefix and trainable final blocks.
"""
import hashlib
from pathlib import Path
from urllib.request import urlopen
import librosa
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

PRETRAIN_URL = 'https://zenodo.org/records/3987831/files/Cnn6_mAP%3D0.343.pth?download=1'
PRETRAIN_MD5 = 'e25e26b84585b14c7754c91e48efc9be'
PRETRAIN_VERSION = 'panns-cnn6-audioset-32k-v1'
ROOT = Path(__file__).resolve().parents[1]

class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.avg_pool2d(F.relu_(self.bn1(self.conv1(x))), (2, 2))

class Cnn6Transfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.spectrogram_extractor = Spectrogram(n_fft=1024, hop_length=320,
            win_length=1024, window='hann', center=True, pad_mode='reflect', freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=32000, n_fft=1024,
            n_mels=64, fmin=50, fmax=14000, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock5x5(1, 64)
        self.conv_block2 = ConvBlock5x5(64, 128)
        self.conv_block3 = ConvBlock5x5(128, 256)
        self.conv_block4 = ConvBlock5x5(256, 512)
        self.fc1 = nn.Linear(512, 512)
        self.fc_audioset = nn.Linear(512, 6)

    def freeze_prefix(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.startswith(('conv_block4.', 'fc1.', 'fc_audioset.'))
        # Keep ALL BN running statistics fixed; six-class data is too small.
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def prefix(self, waveform):
        x = self.logmel_extractor(self.spectrogram_extractor(waveform))
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for block in [self.conv_block1, self.conv_block2, self.conv_block3]:
            x = block(x)
        return x

    def tail(self, x):
        x = self.conv_block4(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = x.mean(dim=3)
        x = x.amax(dim=2) + x.mean(dim=2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc_audioset(x)

    def forward(self, waveform):
        return self.tail(self.prefix(waveform))

def waveform32k(audio, sr):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError('Audio must contain finite samples.')
    if not 0.25 <= len(audio) / sr <= 10:
        raise ValueError('Use a WAV recording between 0.25 and 10 seconds.')
    if np.max(np.abs(audio)) < 1e-6:
        raise ValueError('This recording is silent. Choose an audible recording.')
    audio = librosa.resample(audio, orig_sr=sr, target_sr=32000)
    return librosa.util.fix_length(audio, size=160000)

def load_audioset():
    path = ROOT / 'data/Cnn6_mAP=0.343.pth'
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        with urlopen(PRETRAIN_URL, timeout=120) as response:
            content = response.read()
        if hashlib.md5(content).hexdigest() != PRETRAIN_MD5:
            raise ValueError('Downloaded PANNs checkpoint failed checksum verification.')
        path.write_bytes(content)
    if hashlib.md5(path.read_bytes()).hexdigest() != PRETRAIN_MD5:
        raise ValueError('PANNs checkpoint checksum mismatch.')
    model = Cnn6Transfer()
    state = torch.load(path, map_location='cpu', weights_only=True)['model']
    # Reinitialize only the original 527-class AudioSet output layer.
    state = {k: v for k, v in state.items() if not k.startswith('fc_audioset.')}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert set(missing) == {'fc_audioset.weight', 'fc_audioset.bias'} and not unexpected
    model.freeze_prefix()
    return model

"""The same audio-only feature extractor is used in training and serving."""
import librosa
import numpy as np

CLASSES = ['dog', 'rain', 'sea_waves', 'crackling_fire', 'clock_tick', 'sneezing']
LABELS = ['Dog bark', 'Rain', 'Sea waves', 'Crackling fire', 'Clock tick', 'Sneezing']
SAMPLE_RATE = 22050
FEATURE_VERSION = 'logmel64-mfcc20-mean-std-v1'
INPUT_DIM = 168


def extract_features(audio, sr):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError('Audio must contain finite samples.')
    if not 0.25 <= len(audio) / sr <= 10:
        raise ValueError('Use a WAV recording between 0.25 and 10 seconds.')
    if np.max(np.abs(audio)) < 1e-6:
        raise ValueError('This recording is silent. Choose an audible recording.')
    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    # Fixed five-second input: short files are padded, longer files use the first 5 s.
    audio = librosa.util.fix_length(audio, size=5 * SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_fft=1024,
                                        hop_length=256, n_mels=64)
    logmel = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=20)
    return np.concatenate([logmel.mean(axis=1), logmel.std(axis=1),
                           mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)

"""A small feed-forward network; outputs logits (not probabilities)."""
from torch import nn
from training.features import INPUT_DIM

class SoundMLP(nn.Module):
    def __init__(self, hidden=64, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(INPUT_DIM, hidden), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(hidden, 6))

    def forward(self, x):
        return self.layers(x)

"""Global MLP and controlled temporal-fusion variants (10 urban sound classes)."""
import torch
from torch import nn

class TemporalMLP(nn.Module):
    def __init__(self, variant, hidden=64, dropout=0.5):
        super().__init__()
        if variant not in ('baseline', 'concat', 'fixed', 'gated'):
            raise ValueError(variant)
        self.variant = variant
        self.global_layer = nn.Linear(840 if variant == 'concat' else 168, hidden)
        if variant in ('fixed', 'gated'):
            self.local_layer = nn.Linear(672, hidden)
        if variant == 'gated':
            self.gate = nn.Linear(hidden * 2, 1)
            # Start with the exact fixed-fusion mapping, not a preferred branch.
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden, 10)

    def forward(self, x, return_gate=False):
        g = torch.relu(self.global_layer(x if self.variant == 'concat' else x[:, :168]))
        a = torch.full((len(x), 1), 0.5, device=x.device)
        if self.variant in ('fixed', 'gated'):
            local = torch.relu(self.local_layer(x[:, 168:]))
            if self.variant == 'gated':
                a = torch.sigmoid(self.gate(torch.cat([g, local], dim=1)))
            g = (1 - a) * g + a * local
        logits = self.output(self.dropout(g))
        return (logits, a) if return_gate else logits

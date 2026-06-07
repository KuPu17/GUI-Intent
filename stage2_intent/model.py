import torch
import torch.nn as nn


class GRUIntentModel(nn.Module):
    def __init__(self, input_dim: int = 32, hidden_dim: int = 256, num_layers: int = 1, num_classes: int = 100):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, h = self.gru(x)
        # use last output
        last = out[:, -1, :]
        return self.head(last)

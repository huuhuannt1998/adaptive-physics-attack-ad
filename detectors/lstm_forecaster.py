"""
LSTM forecaster: a non-GNN ICS anomaly detector for the EPB
detector-agnostic generality study (paper Section V-E).

Same forecasting+residual paradigm as GDN (predict the next-step sensor
vector from a window; anomaly score = standardized forecasting residual),
but with an LSTM backbone instead of a learned sensor graph. This isolates
the detector architecture: if the OOD-clamp shortcut and its repair behave
the same on an LSTM as on GDN, the shortcut is a property of the
preprocessing pipeline, not of the GNN.

Interface matches the GDN victim used elsewhere: forward(X) where
X is (B, N, W) returns a (B, N) next-step forecast.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, n_sensors: int, hidden: int = 128, layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.n_sensors = n_sensors
        self.lstm = nn.LSTM(input_size=n_sensors, hidden_size=hidden,
                            num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_sensors)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, N, W) -> (B, W, N) for the LSTM (seq over time).
        if X.dim() == 2:
            X = X.unsqueeze(0)
        seq = X.transpose(1, 2).float()           # (B, W, N)
        out, _ = self.lstm(seq)                    # (B, W, H)
        last = out[:, -1, :]                       # (B, H)
        return self.head(last)                     # (B, N) next-step forecast

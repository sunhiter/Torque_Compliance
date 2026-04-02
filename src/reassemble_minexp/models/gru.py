"""GRU baseline models."""

from __future__ import annotations

import torch
from torch import nn


class GRUClassifier(nn.Module):
    """Sequence baseline using a GRU encoder."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional,
        )
        output_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(output_dim, num_classes),
        )
        self.bidirectional = bidirectional

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(inputs)
        if self.bidirectional:
            features = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            features = hidden[-1]
        return self.head(features)

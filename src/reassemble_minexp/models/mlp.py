"""MLP baseline models."""

from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """Simple MLP classifier over flattened history windows."""

    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        num_classes: int,
        hidden_dims: list[int] | tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims = [input_dim * sequence_length, *hidden_dims, num_classes]
        layers: list[nn.Module] = []
        for index in range(len(dims) - 2):
            layers.extend(
                [
                    nn.Linear(dims[index], dims[index + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs.reshape(inputs.shape[0], -1))

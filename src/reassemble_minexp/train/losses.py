"""Loss helpers for Milestone 7 training."""

from __future__ import annotations

import torch
from torch import nn


def build_classification_loss(class_weights: list[float] | None = None) -> nn.Module:
    """Build a cross-entropy loss, optionally with class weighting."""

    if class_weights:
        weights = torch.tensor(class_weights, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weights)
    return nn.CrossEntropyLoss()

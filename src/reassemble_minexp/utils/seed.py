"""Random seed helpers."""

from __future__ import annotations

import random


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when available."""

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

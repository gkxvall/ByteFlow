from __future__ import annotations

import torch


def batch_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of correct predictions in the batch (0..1)."""
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(targets.size(0), 1)


class AverageMeter:
    """Running mean for a scalar (e.g. loss) over weighted steps."""

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)

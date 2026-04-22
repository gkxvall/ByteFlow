from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import AverageMeter, batch_accuracy


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    t0 = time.perf_counter()

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        bs = inputs.size(0)
        acc = batch_accuracy(logits.detach(), targets)
        loss_meter.update(loss.item(), bs)
        acc_meter.update(acc, bs)

    elapsed = time.perf_counter() - t0
    return {
        "loss": loss_meter.avg,
        "accuracy": acc_meter.avg,
        "duration_s": elapsed,
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    t0 = time.perf_counter()

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)

        bs = inputs.size(0)
        acc = batch_accuracy(logits, targets)
        loss_meter.update(loss.item(), bs)
        acc_meter.update(acc, bs)

    elapsed = time.perf_counter() - t0
    return {
        "loss": loss_meter.avg,
        "accuracy": acc_meter.avg,
        "duration_s": elapsed,
    }

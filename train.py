from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import build_datasets
from engine import train_one_epoch, validate_one_epoch
from model import build_model
from utils.device import resolve_device
from utils.memory import cuda_mem_stats, format_cuda_mem_line
from utils.seed import set_seed


def main() -> None:
    set_seed(config.SEED)
    device = resolve_device(config.DEVICE)
    pin_memory = bool(config.PIN_MEMORY and device.type == "cuda")

    train_ds, val_ds, class_to_idx = build_datasets(
        config.DATASET_ROOT,
        config.IMAGE_SIZE,
        config.TRAIN_SPLIT,
        config.SEED,
    )
    num_classes = len(class_to_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
    )

    print("--- ByteFlow v1 ---")
    print(f"dataset root: {config.DATASET_ROOT}")
    print(f"classes ({num_classes}): {class_to_idx}")
    print(f"total samples: {len(train_ds) + len(val_ds)}")
    print(f"train samples: {len(train_ds)}")
    print(f"val samples:   {len(val_ds)}")
    print(f"batch size: {config.BATCH_SIZE}")
    print(f"num_workers: {config.NUM_WORKERS}")
    print(f"pin_memory: {pin_memory}")
    print(f"device: {device}")
    print(format_cuda_mem_line(cuda_mem_stats(device)))
    print("-------------------")

    model = build_model(config.MODEL_NAME, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0

    for epoch in range(1, config.EPOCHS + 1):
        t_epoch = time.perf_counter()
        train_m = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_m = validate_one_epoch(model, val_loader, criterion, device)
        epoch_s = time.perf_counter() - t_epoch

        print(
            f"Epoch {epoch}/{config.EPOCHS}  "
            f"train_loss={train_m['loss']:.4f}  train_acc={train_m['accuracy']:.4f}  "
            f"val_loss={val_m['loss']:.4f}  val_acc={val_m['accuracy']:.4f}  "
            f"time={epoch_s:.1f}s"
        )
        print(f"  (train_step={train_m['duration_s']:.1f}s, val_step={val_m['duration_s']:.1f}s)")
        print(format_cuda_mem_line(cuda_mem_stats(device)))

        if config.SAVE_BEST_MODEL and val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": best_val_acc,
                    "class_to_idx": class_to_idx,
                    "model_name": config.MODEL_NAME,
                },
                config.CHECKPOINT_PATH,
            )
            print(f"  saved best checkpoint -> {config.CHECKPOINT_PATH} (val_acc={best_val_acc:.4f})")

    print("Training finished.")


if __name__ == "__main__":
    main()

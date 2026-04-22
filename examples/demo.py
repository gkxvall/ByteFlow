import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import directly from the byteflow package
from byteflow import (
    build_datasets,
    train_one_epoch,
    validate_one_epoch,
    build_model,
)
from byteflow.utils.device import resolve_device
from byteflow.utils.memory import cuda_mem_stats, format_cuda_mem_line
from byteflow.utils.seed import set_seed


def main() -> None:
    # 1. Parse command-line arguments instead of hardcoding config
    parser = argparse.ArgumentParser(description="Train an image classifier using ByteFlow.")
    parser.add_argument("--dataset-root", type=str, default="examples/sample_dataset",
                        help="Path to the dataset directory (one subfolder per class)")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train/val split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--model-name", type=str, default="resnet18", help="Model backbone")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, or auto)")
    parser.add_argument("--checkpoint-path", type=str, default="best_model.pth",
                        help="Path to save the best model weights")
    args = parser.parse_args()

    # 2. Set up seed and device
    set_seed(args.seed)
    device = resolve_device(args.device)
    pin_memory = bool(device.type == "cuda")

    # 3. Build Datasets
    # build_datasets returns our ByteFlowImageFolder instances and class mapping
    try:
        train_ds, val_ds, class_to_idx = build_datasets(
            dataset_root=args.dataset_root,
            image_size=args.image_size,
            train_split=args.train_split,
            seed=args.seed,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    num_classes = len(class_to_idx)

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print("--- ByteFlow Package Demo ---")
    print(f"dataset root: {args.dataset_root}")
    print(f"classes ({num_classes}): {class_to_idx}")
    print(f"total samples: {len(train_ds) + len(val_ds)}")
    print(f"train samples: {len(train_ds)}")
    print(f"val samples:   {len(val_ds)}")
    print(f"batch size: {args.batch_size}")
    print(f"device: {device}")
    print(format_cuda_mem_line(cuda_mem_stats(device)))
    print("-------------------")

    # 5. Build Model, Optimizer, and Loss Criterion
    model = build_model(args.model_name, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Optional: Add a Learning Rate Scheduler!
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0

    # 6. Training Loop
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.perf_counter()
        
        # Train and Validate using the engine functions from byteflow
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m = validate_one_epoch(model, val_loader, criterion, device)
        
        # Step the scheduler
        scheduler.step()
        
        epoch_s = time.perf_counter() - t_epoch

        print(
            f"Epoch {epoch}/{args.epochs}  "
            f"train_loss={train_m['loss']:.4f}  train_acc={train_m['accuracy']:.4f}  "
            f"val_loss={val_m['loss']:.4f}  val_acc={val_m['accuracy']:.4f}  "
            f"time={epoch_s:.1f}s"
        )
        print(f"  (train_step={train_m['duration_s']:.1f}s, val_step={val_m['duration_s']:.1f}s)")
        print(format_cuda_mem_line(cuda_mem_stats(device)))

        # 7. Save Checkpoint
        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": best_val_acc,
                    "class_to_idx": class_to_idx,
                    "model_name": args.model_name,
                },
                args.checkpoint_path,
            )
            print(f"  saved best checkpoint -> {args.checkpoint_path} (val_acc={best_val_acc:.4f})")

    print("Training finished.")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import torch.amp as amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    ARTIST_LOSS_WEIGHT,
    BASE_LR,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DEVICE,
    EPOCHS,
    GRAD_ACCUMULATION_STEPS,
    HISTORY_FILE,
    IMAGE_SIZE_CNN,
    LABEL_SMOOTHING,
    MIXED_PRECISION,
    MODEL_CANDIDATES,
    MOVEMENT_LOSS_WEIGHT,
    VAL_BATCH_SIZE,
    WEIGHT_DECAY,
)
from data_loader import get_dataloaders
from model import MultiTaskClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-task WikiArt classifier.")
    parser.add_argument("--arch", type=str, default="convnext_tiny", choices=MODEL_CANDIDATES)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--val-batch-size", type=int, default=VAL_BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE_CNN)
    parser.add_argument("--lr", type=float, default=BASE_LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUMULATION_STEPS)
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained initialization.")
    parser.add_argument("--checkpoint-name", type=str, default=None)
    return parser.parse_args()


def create_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer, epochs: int):
    return CosineAnnealingLR(optimizer, T_max=epochs)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def run_epoch(
    model: MultiTaskClassifier,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer=None,
    scaler: amp.GradScaler = None,
    train: bool = True,
    grad_accum_steps: int = 1,
):
    if train:
        model.train()
    else:
        model.eval()

    if train:
        optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    movement_acc = 0.0
    artist_acc = 0.0
    total_samples = 0

    if scaler is None:
        scaler = amp.GradScaler('cuda', enabled=False) if DEVICE.type == 'cuda' else None

    device_type = 'cuda' if DEVICE.type == 'cuda' else 'cpu'
    use_amp = scaler is not None and scaler.is_enabled()

    for step, (images, movements, artists, _) in enumerate(dataloader):
        images = images.to(DEVICE, non_blocking=True)
        movements = movements.to(DEVICE)
        artists = artists.to(DEVICE)

        with amp.autocast(device_type=device_type, enabled=use_amp):
            outputs = model(images)
            movement_loss = criterion(outputs["movement_logits"], movements)
            artist_loss = criterion(outputs["artist_logits"], artists)
            loss = MOVEMENT_LOSS_WEIGHT * movement_loss + ARTIST_LOSS_WEIGHT * artist_loss

        if train:
            if scaler is not None:
                scaler.scale(loss / grad_accum_steps).backward()
                if (step + 1) % grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                (loss / grad_accum_steps).backward()
                if (step + 1) % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        movement_acc += accuracy(outputs["movement_logits"], movements) * batch_size
        artist_acc += accuracy(outputs["artist_logits"], artists) * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "movement_acc": movement_acc / total_samples,
        "artist_acc": artist_acc / total_samples,
    }


def collect_predictions(model, dataloader):
    model.eval()
    all_movements = []
    all_movement_targets = []
    all_artists = []
    all_artist_targets = []

    with torch.no_grad():
        for images, movement_targets, artist_targets, _ in dataloader:
            images = images.to(DEVICE, non_blocking=True)
            outputs = model(images)
            all_movements.append(outputs["movement_logits"].cpu())
            all_artists.append(outputs["artist_logits"].cpu())
            all_movement_targets.append(movement_targets)
            all_artist_targets.append(artist_targets)

    movement_logits = torch.cat(all_movements)
    artist_logits = torch.cat(all_artists)
    movement_preds = movement_logits.argmax(dim=1).numpy()
    artist_preds = artist_logits.argmax(dim=1).numpy()
    movement_targets = torch.cat(all_movement_targets).numpy()
    artist_targets = torch.cat(all_artist_targets).numpy()
    return (movement_preds, movement_targets), (artist_preds, artist_targets)


def plot_confusion(cm: np.ndarray, class_names: Dict[int, str], output_path: Path, title: str):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, [class_names[i] for i in tick_marks], rotation=90)
    plt.yticks(tick_marks, [class_names[i] for i in tick_marks])
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=6,
            )
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def export_embeddings(model, dataloader, output_path: Path):
    model.eval()
    features = []
    movement_targets = []
    artist_targets = []
    record_ids = []
    with torch.no_grad():
        for images, movements, artists, record_id in dataloader:
            images = images.to(DEVICE, non_blocking=True)
            outputs = model(images)
            features.append(outputs["features"].cpu().numpy())
            movement_targets.append(movements.numpy())
            artist_targets.append(artists.numpy())
            record_ids.extend(record_id)
    np.savez(
        output_path,
        features=np.concatenate(features),
        movement_targets=np.concatenate(movement_targets),
        artist_targets=np.concatenate(artist_targets),
        record_ids=np.array(record_ids),
    )


def save_history(entry: Dict):
    history = []
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            history = json.load(f)
    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def main():
    args = parse_args()
    dataloaders, datasets, dataset_info = get_dataloaders(
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
    )
    model = MultiTaskClassifier(
        arch=args.arch,
        num_movements=dataset_info["num_movements"],
        num_artists=dataset_info["num_artists"],
        dropout=args.dropout,
        pretrained=not args.no_pretrained,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = create_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, args.epochs)
    scaler = amp.GradScaler('cuda', enabled=MIXED_PRECISION) if DEVICE.type == "cuda" else None

    history = {"train": [], "val": []}
    best_val_acc = -1.0
    best_state = None

    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model,
            dataloaders["train"],
            criterion,
            optimizer,
            scaler,
            train=True,
            grad_accum_steps=args.grad_accum,
        )
        val_metrics = run_epoch(model, dataloaders["val"], criterion, optimizer=None, train=False)
        scheduler.step()

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        val_score = 0.5 * (val_metrics["movement_acc"] + val_metrics["artist_acc"])
        if val_score > best_val_acc:
            best_val_acc = val_score
            best_state = model.state_dict()

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss {train_metrics['loss']:.4f} | "
            f"Val Loss {val_metrics['loss']:.4f} | "
            f"Val Movement Acc {val_metrics['movement_acc']:.3f} | "
            f"Val Artist Acc {val_metrics['artist_acc']:.3f}"
        )

    checkpoint_name = args.checkpoint_name or f"{args.arch}_best.pt"
    checkpoint_path = CHECKPOINT_DIR / checkpoint_name
    torch.save({"model_state": best_state, "config": vars(args)}, checkpoint_path)
    print(f"Saved best model to {checkpoint_path}")
    model.load_state_dict(best_state)

    test_metrics = run_epoch(model, dataloaders["test"], criterion, train=False)
    print(
        f"Test Loss {test_metrics['loss']:.4f} | "
        f"Test Movement Acc {test_metrics['movement_acc']:.3f} | "
        f"Test Artist Acc {test_metrics['artist_acc']:.3f}"
    )

    (movement_preds, movement_targets), (artist_preds, artist_targets) = collect_predictions(
        model, dataloaders["test"]
    )
    movement_cm = confusion_matrix(movement_targets, movement_preds)
    artist_cm = confusion_matrix(artist_targets, artist_preds)
    movement_map = dataset_info["movement_map"]
    artist_map = dataset_info["artist_map"]
    from config import CONFUSION_DIR
    plot_confusion(movement_cm, movement_map, CONFUSION_DIR / f"{args.arch}_movement_cm.png", "Movement Confusion")
    plot_confusion(artist_cm, artist_map, CONFUSION_DIR / f"{args.arch}_artist_cm.png", "Artist Confusion")

    export_embeddings(model, dataloaders["test"], CHECKPOINT_DIR / f"{args.arch}_test_embeddings.npz")

    history_entry = {
        "arch": args.arch,
        "epochs": args.epochs,
        "train_history": history["train"],
        "val_history": history["val"],
        "test_metrics": test_metrics,
        "checkpoint": str(checkpoint_path),
    }
    save_history(history_entry)


if __name__ == "__main__":
    main()

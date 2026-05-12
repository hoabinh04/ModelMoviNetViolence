from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from buoc2_dataloader import CROPPED_DIR, ORIGINAL_DIR, get_dataloaders
from buoc3_model import create_tsm_model

LOGGER = logging.getLogger("train_tsm")


@dataclass
class TrainConfig:
    original_dir: Path
    cropped_dir: Path
    save_dir: Path
    num_classes: int = 2
    num_segments: int = 8
    image_size: int = 224
    batch_size: int = 8
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    min_frames: Optional[int] = None
    max_fallback_ratio: float = 1.0
    use_report_filter: bool = True
    pin_memory: bool = True
    pretrained: bool = True
    shift_div: int = 8
    shift_stride: int = 2
    dropout: float = 0.20
    freeze_backbone_epochs: int = 2
    label_smoothing: float = 0.03
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 7
    seed: int = 42
    amp: bool = True
    resume: Optional[Path] = None


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights_from_loader(train_loader) -> torch.Tensor:
    labels = [sample.label for sample in train_loader.dataset.samples]
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    counts = np.clip(counts, a_min=1.0, a_max=None)
    inv = 1.0 / counts
    weights = inv / inv.sum() * len(inv)
    return torch.tensor(weights, dtype=torch.float32)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predicted = logits.argmax(dim=1)
    correct = (predicted == labels).sum().item()
    return correct / max(1, labels.size(0))


def save_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    amp_enabled: bool,
    grad_clip_norm: float,
) -> Tuple[float, float]:
    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    progress = tqdm(loader, desc="train", leave=False, dynamic_ncols=True)
    for videos, labels in progress:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            logits = model(videos)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()

        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (logits.argmax(dim=1) == labels).sum().item()
        running_total += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.0 * running_correct / max(1, running_total):.2f}%",
        )

    epoch_loss = running_loss / max(1, len(loader.dataset))
    epoch_acc = running_correct / max(1, running_total)
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[float, float]:
    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    progress = tqdm(loader, desc="valid", leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for videos, labels in progress:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=amp_enabled):
                logits = model(videos)
                loss = criterion(logits, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_total += batch_size

    epoch_loss = running_loss / max(1, len(loader.dataset))
    epoch_acc = running_correct / max(1, running_total)
    return epoch_loss, epoch_acc


def maybe_load_checkpoint(
    resume_path: Optional[Path],
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
) -> Tuple[int, float]:
    if resume_path is None:
        return 0, 0.0
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])

    scaler_state = checkpoint.get("scaler_state")
    if scaler_state:
        scaler.load_state_dict(scaler_state)

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val_acc = float(checkpoint.get("best_val_acc", 0.0))
    LOGGER.info("Resumed from %s at epoch=%d", resume_path, start_epoch)
    return start_epoch, best_val_acc


def train(cfg: TrainConfig) -> int:
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg.amp and device.type == "cuda")

    LOGGER.info("Device: %s | AMP: %s", device.type.upper(), amp_enabled)

    train_loader, val_loader = get_dataloaders(
        orig_dir=cfg.original_dir,
        crop_dir=cfg.cropped_dir,
        batch_size=cfg.batch_size,
        num_segments=cfg.num_segments,
        image_size=cfg.image_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device.type == "cuda",
        min_frames=cfg.min_frames,
        max_fallback_ratio=cfg.max_fallback_ratio,
        use_report_filter=cfg.use_report_filter,
        seed=cfg.seed,
    )

    model = create_tsm_model(
        num_classes=cfg.num_classes,
        num_segments=cfg.num_segments,
        pretrained=cfg.pretrained,
        shift_div=cfg.shift_div,
        shift_stride=cfg.shift_stride,
        dropout=cfg.dropout,
    ).to(device)

    class_weights = compute_class_weights_from_loader(train_loader).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=cfg.learning_rate * 0.1)
    scaler = GradScaler(enabled=amp_enabled)

    start_epoch, best_val_acc = maybe_load_checkpoint(
        cfg.resume,
        model,
        optimizer,
        scheduler,
        scaler,
        device,
    )

    log_path = cfg.save_dir / "training_log.csv"
    config_path = cfg.save_dir / "train_config.json"
    best_state_path = cfg.save_dir / "best_tsm_model.pth"
    last_ckpt_path = cfg.save_dir / "last_tsm_checkpoint.pth"

    save_json(config_path, asdict(cfg))

    if start_epoch == 0:
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"])

    no_improve_count = 0
    begin = time.time()

    for epoch in range(start_epoch, cfg.num_epochs):
        if epoch < cfg.freeze_backbone_epochs:
            model.freeze_backbone()
        else:
            model.unfreeze_all()

        current_lr = optimizer.param_groups[0]["lr"]
        LOGGER.info("Epoch %d/%d | lr=%.7f", epoch + 1, cfg.num_epochs, current_lr)

        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            grad_clip_norm=cfg.grad_clip_norm,
        )

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
        )

        scheduler.step()

        LOGGER.info(
            "Epoch %d summary | train_loss=%.4f train_acc=%.2f%% | val_loss=%.4f val_acc=%.2f%%",
            epoch + 1,
            train_loss,
            train_acc * 100.0,
            val_loss,
            val_acc * 100.0,
        )

        with log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    epoch + 1,
                    f"{current_lr:.8f}",
                    f"{train_loss:.6f}",
                    f"{train_acc * 100.0:.4f}",
                    f"{val_loss:.6f}",
                    f"{val_acc * 100.0:.4f}",
                ]
            )

        checkpoint = {
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if amp_enabled else None,
            "config": asdict(cfg),
        }
        torch.save(checkpoint, last_ckpt_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0
            torch.save(model.state_dict(), best_state_path)
            LOGGER.info("New best model saved: %.2f%%", best_val_acc * 100.0)
        else:
            no_improve_count += 1

        if no_improve_count >= cfg.early_stop_patience:
            LOGGER.info("Early stopping triggered after %d stale epochs.", no_improve_count)
            break

    elapsed = time.time() - begin
    LOGGER.info(
        "Training completed in %.1f minutes. Best val accuracy: %.2f%%",
        elapsed / 60.0,
        best_val_acc * 100.0,
    )
    LOGGER.info("Best state dict: %s", best_state_path)
    LOGGER.info("Last checkpoint: %s", last_ckpt_path)
    LOGGER.info("Train log: %s", log_path)

    return 0


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train MobileNetV3 + TSM-lite for violence detection")

    parser.add_argument("--original-dir", type=Path, default=ORIGINAL_DIR)
    parser.add_argument("--cropped-dir", type=Path, default=CROPPED_DIR)
    parser.add_argument("--save-dir", type=Path, default=Path(__file__).resolve().parents[1] / "weights")

    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--num-segments", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--min-frames", type=int, default=None)
    parser.add_argument("--max-fallback-ratio", type=float, default=1.0)
    parser.add_argument("--disable-report-filter", action="store_true")

    parser.add_argument("--disable-pretrained", action="store_true")
    parser.add_argument("--shift-div", type=int, default=8)
    parser.add_argument("--shift-stride", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.20)

    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)

    args = parser.parse_args()

    return TrainConfig(
        original_dir=args.original_dir,
        cropped_dir=args.cropped_dir,
        save_dir=args.save_dir,
        num_classes=args.num_classes,
        num_segments=args.num_segments,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        min_frames=args.min_frames,
        max_fallback_ratio=args.max_fallback_ratio,
        use_report_filter=not args.disable_report_filter,
        pin_memory=True,
        pretrained=not args.disable_pretrained,
        shift_div=args.shift_div,
        shift_stride=args.shift_stride,
        dropout=args.dropout,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        label_smoothing=args.label_smoothing,
        grad_clip_norm=args.grad_clip_norm,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
        amp=not args.disable_amp,
        resume=args.resume,
    )


def main() -> None:
    setup_logging()
    cfg = parse_args()
    raise SystemExit(train(cfg))


if __name__ == "__main__":
    main()

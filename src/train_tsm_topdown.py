from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from buoc3_model import create_tsm_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIOLENCE_DIR = Path(r"C:\Users\Lenovo\Desktop\codeNCKH\codeDot2\Data_Violence_Detection\Violence")
DEFAULT_NONVIOLENCE_DIR = Path(r"C:\Users\Lenovo\Desktop\codeNCKH\codeDot2\Data_Violence_Detection\NonViolence")
DEFAULT_INIT_WEIGHTS = PROJECT_ROOT / "weights" / "best_tsm_model.pth"
DEFAULT_SAVE_DIR = PROJECT_ROOT / "weights"
DEFAULT_NUM_SEGMENTS = 8

LOGGER = logging.getLogger("train_tsm_topdown")


@dataclass
class FineTuneConfig:
    violence_dir: Path
    nonviolence_dir: Path
    init_weights: Path
    save_dir: Path
    output_name: str
    num_segments: int
    image_size: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    val_ratio: float
    max_videos_per_class: int
    max_frames_per_video: int
    hard_negative_list: Optional[Path]
    hard_negative_repeat: int
    freeze_backbone_epochs: int
    grad_clip_norm: float
    label_smoothing: float
    early_stop_patience: int
    seed: int
    amp: bool


class TopdownVideoDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[Path, int]],
        num_segments: int,
        image_size: int,
        is_train: bool,
        max_frames_per_video: int,
        seed: int,
    ) -> None:
        self.samples = list(samples)
        self.num_segments = num_segments
        self.image_size = image_size
        self.is_train = is_train
        self.max_frames_per_video = max_frames_per_video
        self.rng = random.Random(seed)

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        if is_train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_indices(self, frame_count: int) -> List[int]:
        if frame_count <= 0:
            return [0] * self.num_segments

        if self.max_frames_per_video > 0:
            frame_count = min(frame_count, self.max_frames_per_video)

        if frame_count < self.num_segments:
            return np.linspace(0, frame_count - 1, self.num_segments).astype(np.int32).tolist()

        edges = np.linspace(0, frame_count, self.num_segments + 1).astype(np.int32)
        indices: List[int] = []
        for i in range(self.num_segments):
            start = int(edges[i])
            end = int(edges[i + 1])
            if end <= start:
                end = min(start + 1, frame_count)

            if self.is_train:
                indices.append(self.rng.randint(start, end - 1))
            else:
                indices.append((start + end - 1) // 2)
        return indices

    def _blank_frame(self) -> Image.Image:
        return Image.fromarray(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8), mode="RGB")

    def _load_frames(self, video_path: Path) -> List[torch.Tensor]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            blank = self._blank_frame()
            return [self.transform(blank) for _ in range(self.num_segments)]

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = self._sample_indices(frame_count)

        frames: List[torch.Tensor] = []
        last_good_frame = None

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                last_good_frame = frame
            elif last_good_frame is not None:
                frame = last_good_frame
            else:
                frame = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            frames.append(self.transform(pil_img))

        cap.release()

        if len(frames) < self.num_segments:
            blank = self.transform(self._blank_frame())
            while len(frames) < self.num_segments:
                frames.append(blank)

        return frames

    def __getitem__(self, index: int):
        video_path, label = self.samples[index]
        frames = self._load_frames(video_path)
        return torch.stack(frames), int(label)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_videos(folder: Path, prefix: str) -> List[Path]:
    return sorted(p for p in folder.glob(f"{prefix}*.mp4") if p.is_file())


def split_samples(
    violence_videos: Sequence[Path],
    nonviolence_videos: Sequence[Path],
    val_ratio: float,
    max_videos_per_class: int,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    rng = random.Random(seed)

    def split_one(videos: Sequence[Path], label: int) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
        vids = list(videos)
        rng.shuffle(vids)
        if max_videos_per_class > 0:
            vids = vids[:max_videos_per_class]

        if len(vids) <= 1:
            train_vids = vids
            val_vids: List[Path] = []
        else:
            val_count = max(1, int(round(len(vids) * val_ratio)))
            val_count = min(val_count, len(vids) - 1)
            val_vids = vids[:val_count]
            train_vids = vids[val_count:]

        train_items = [(v, label) for v in train_vids]
        val_items = [(v, label) for v in val_vids]
        return train_items, val_items

    train_fight, val_fight = split_one(violence_videos, 1)
    train_non, val_non = split_one(nonviolence_videos, 0)

    train_samples = train_fight + train_non
    val_samples = val_fight + val_non
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    return train_samples, val_samples


def load_hard_negative_names(file_path: Optional[Path]) -> Set[str]:
    if file_path is None:
        return set()
    if not file_path.exists():
        raise FileNotFoundError(f"Hard-negative list not found: {file_path}")

    names: Set[str] = set()
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            names.add(Path(line).name.lower())

    return names


def compute_class_weights(train_samples: Sequence[Tuple[Path, int]]) -> torch.Tensor:
    labels = [label for _path, label in train_samples]
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    counts = np.clip(counts, a_min=1.0, a_max=None)
    inv = 1.0 / counts
    weights = inv / inv.sum() * len(inv)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
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
        running_loss += float(loss.item()) * batch_size
        running_correct += int((logits.argmax(dim=1) == labels).sum().item())
        running_total += int(batch_size)

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.0 * running_correct / max(1, running_total):.2f}%",
        )

    epoch_loss = running_loss / max(1, len(loader.dataset))
    epoch_acc = running_correct / max(1, running_total)
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
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
            running_loss += float(loss.item()) * batch_size
            running_correct += int((logits.argmax(dim=1) == labels).sum().item())
            running_total += int(batch_size)

    epoch_loss = running_loss / max(1, len(loader.dataset))
    epoch_acc = running_correct / max(1, running_total)
    return epoch_loss, epoch_acc


def resolve_num_segments(init_weights: Path, user_value: Optional[int]) -> int:
    if user_value is not None:
        if int(user_value) < 2:
            raise ValueError("--num-segments must be >= 2")
        return int(user_value)

    cfg_path = init_weights.parent / "train_config.json"
    if cfg_path.exists():
        try:
            payload = json.loads(cfg_path.read_text(encoding="utf-8"))
            value = int(payload.get("num_segments", DEFAULT_NUM_SEGMENTS))
            if value >= 2:
                return value
        except Exception:
            pass

    return DEFAULT_NUM_SEGMENTS


def parse_args() -> FineTuneConfig:
    parser = argparse.ArgumentParser(description="Fine-tune TSM model on top-down Violence/NonViolence videos")
    parser.add_argument("--violence-dir", type=Path, default=DEFAULT_VIOLENCE_DIR)
    parser.add_argument("--nonviolence-dir", type=Path, default=DEFAULT_NONVIOLENCE_DIR)
    parser.add_argument("--init-weights", type=Path, default=DEFAULT_INIT_WEIGHTS)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--output-name", type=str, default="best_tsm_topdown.pth")

    parser.add_argument("--num-segments", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-videos-per-class", type=int, default=0)
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    parser.add_argument("--hard-negative-list", type=Path, default=None)
    parser.add_argument("--hard-negative-repeat", type=int, default=0)

    parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-amp", action="store_true")

    args = parser.parse_args()

    if not (0.0 < args.val_ratio < 0.9):
        raise ValueError("--val-ratio must be between 0 and 0.9")
    if int(args.hard_negative_repeat) < 0:
        raise ValueError("--hard-negative-repeat must be >= 0")

    num_segments = resolve_num_segments(args.init_weights, args.num_segments)

    return FineTuneConfig(
        violence_dir=args.violence_dir,
        nonviolence_dir=args.nonviolence_dir,
        init_weights=args.init_weights,
        save_dir=args.save_dir,
        output_name=args.output_name,
        num_segments=num_segments,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        max_videos_per_class=args.max_videos_per_class,
        max_frames_per_video=args.max_frames_per_video,
        hard_negative_list=args.hard_negative_list,
        hard_negative_repeat=args.hard_negative_repeat,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        grad_clip_norm=args.grad_clip_norm,
        label_smoothing=args.label_smoothing,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
        amp=not args.disable_amp,
    )


def train(cfg: FineTuneConfig) -> int:
    if not cfg.init_weights.exists():
        raise FileNotFoundError(f"Init weights not found: {cfg.init_weights}")

    violence_videos = collect_videos(cfg.violence_dir, "f")
    nonviolence_videos = collect_videos(cfg.nonviolence_dir, "nf")
    if not violence_videos:
        raise FileNotFoundError(f"No violence videos found in {cfg.violence_dir}")
    if not nonviolence_videos:
        raise FileNotFoundError(f"No nonviolence videos found in {cfg.nonviolence_dir}")

    train_samples, val_samples = split_samples(
        violence_videos=violence_videos,
        nonviolence_videos=nonviolence_videos,
        val_ratio=cfg.val_ratio,
        max_videos_per_class=cfg.max_videos_per_class,
        seed=cfg.seed,
    )

    hard_negative_names = load_hard_negative_names(cfg.hard_negative_list)
    if hard_negative_names and cfg.hard_negative_repeat > 0:
        matched_hard_negatives = [
            item for item in train_samples if item[1] == 0 and item[0].name.lower() in hard_negative_names
        ]

        if matched_hard_negatives:
            added = len(matched_hard_negatives) * cfg.hard_negative_repeat
            train_samples.extend(matched_hard_negatives * cfg.hard_negative_repeat)
            random.Random(cfg.seed + 123).shuffle(train_samples)
            LOGGER.info(
                "Hard-negative oversampling | list=%s matched=%d repeat=%d added=%d",
                cfg.hard_negative_list,
                len(matched_hard_negatives),
                cfg.hard_negative_repeat,
                added,
            )
        else:
            LOGGER.warning(
                "Hard-negative list provided but no train samples matched: %s",
                cfg.hard_negative_list,
            )
    elif hard_negative_names:
        LOGGER.info(
            "Hard-negative list loaded (%d items) but repeat=0, oversampling skipped",
            len(hard_negative_names),
        )

    if len(train_samples) == 0 or len(val_samples) == 0:
        raise RuntimeError("Top-down split is empty. Increase data or adjust --val-ratio/--max-videos-per-class")

    train_dataset = TopdownVideoDataset(
        samples=train_samples,
        num_segments=cfg.num_segments,
        image_size=cfg.image_size,
        is_train=True,
        max_frames_per_video=cfg.max_frames_per_video,
        seed=cfg.seed,
    )
    val_dataset = TopdownVideoDataset(
        samples=val_samples,
        num_segments=cfg.num_segments,
        image_size=cfg.image_size,
        is_train=False,
        max_frames_per_video=cfg.max_frames_per_video,
        seed=cfg.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg.amp and device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )

    model = create_tsm_model(
        num_classes=2,
        num_segments=cfg.num_segments,
        pretrained=False,
        shift_div=8,
        shift_stride=2,
        dropout=0.20,
    ).to(device)
    model.load_state_dict(torch.load(cfg.init_weights, map_location=device))

    class_weights = compute_class_weights(train_samples).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.num_epochs,
        eta_min=cfg.learning_rate * 0.1,
    )
    scaler = GradScaler(enabled=amp_enabled)

    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(cfg.output_name).stem
    best_path = cfg.save_dir / cfg.output_name
    last_ckpt_path = cfg.save_dir / f"{stem}_last_checkpoint.pth"
    log_path = cfg.save_dir / f"{stem}_training_log.csv"
    config_path = cfg.save_dir / f"{stem}_config.json"

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2, default=str)

    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_val_acc = 0.0
    no_improve_count = 0
    begin = time.time()

    LOGGER.info(
        "Top-down fine-tune start | device=%s amp=%s train=%d val=%d segments=%d",
        device.type,
        amp_enabled,
        len(train_dataset),
        len(val_dataset),
        cfg.num_segments,
    )

    for epoch in range(cfg.num_epochs):
        if epoch < cfg.freeze_backbone_epochs:
            model.freeze_backbone()
        else:
            model.unfreeze_all()

        current_lr = float(optimizer.param_groups[0]["lr"])
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

        torch.save(
            {
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if amp_enabled else None,
                "config": asdict(cfg),
            },
            last_ckpt_path,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0
            torch.save(model.state_dict(), best_path)
            LOGGER.info("New best top-down model saved: %.2f%%", best_val_acc * 100.0)
        else:
            no_improve_count += 1

        if no_improve_count >= cfg.early_stop_patience:
            LOGGER.info("Early stopping after %d stale epochs", no_improve_count)
            break

    elapsed = (time.time() - begin) / 60.0
    LOGGER.info(
        "Top-down fine-tune done in %.1f minutes | best val_acc=%.2f%%",
        elapsed,
        best_val_acc * 100.0,
    )
    LOGGER.info("Best weights: %s", best_path)
    LOGGER.info("Last checkpoint: %s", last_ckpt_path)
    LOGGER.info("Train log: %s", log_path)
    return 0


def main() -> None:
    setup_logging()
    cfg = parse_args()
    set_seed(cfg.seed)
    raise SystemExit(train(cfg))


if __name__ == "__main__":
    main()

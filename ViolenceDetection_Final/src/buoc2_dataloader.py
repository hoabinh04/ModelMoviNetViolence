from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

LOGGER = logging.getLogger("rwf_dataloader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DIR = PROJECT_ROOT / "RWF-2000"
CROPPED_DIR = PROJECT_ROOT / "RWF-2000-Cropped"
DEFAULT_NUM_SEGMENTS = 8

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".mpeg"}
LABEL_MAP = {"NonFight": 0, "Fight": 1}


@dataclass(frozen=True)
class VideoSample:
    split: str
    label_name: str
    label: int
    video_name: str
    frame_paths: List[Path]


def _load_contract_image_size(cropped_dir: Path, fallback_size: int = 224) -> int:
    contract_path = cropped_dir / "crop_contract.json"
    if not contract_path.exists():
        LOGGER.warning("crop_contract.json not found. Fallback image_size=%d", fallback_size)
        return fallback_size

    try:
        with contract_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        saved_size = data.get("frame_policy", {}).get("saved_size", [fallback_size, fallback_size])
        if not isinstance(saved_size, list) or len(saved_size) != 2:
            return fallback_size
        if int(saved_size[0]) != int(saved_size[1]):
            LOGGER.warning("Non-square saved_size in contract, fallback image_size=%d", fallback_size)
            return fallback_size
        return int(saved_size[0])
    except Exception as ex:
        LOGGER.warning("Cannot parse crop contract (%s). Fallback image_size=%d", ex, fallback_size)
        return fallback_size


def _load_prep_report(cropped_dir: Path) -> Dict[str, Dict[str, str]]:
    report_path = cropped_dir / "prep_report.csv"
    if not report_path.exists():
        return {}

    rows: Dict[str, Dict[str, str]] = {}
    with report_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rel = row.get("video_rel_path", "")
            if rel:
                rows[rel] = row
    return rows


def _sorted_frames(folder: Path) -> List[Path]:
    return sorted(folder.glob("*.jpg"), key=lambda p: p.name)


class RWF2000TSMDataset(Dataset):
    def __init__(
        self,
        original_dir: Path,
        cropped_dir: Path,
        split: str,
        num_segments: int,
        transform,
        min_frames: int,
        max_fallback_ratio: float,
        use_report_filter: bool,
        seed: int,
    ):
        self.original_dir = Path(original_dir)
        self.cropped_dir = Path(cropped_dir)
        self.split = split
        self.num_segments = num_segments
        self.is_train = split == "train"
        self.transform = transform
        self.min_frames = min_frames
        self.max_fallback_ratio = max_fallback_ratio
        self.use_report_filter = use_report_filter
        self.rng = random.Random(seed)
        self.samples: List[VideoSample] = []

        self._build_index()

    def _build_index(self) -> None:
        split_root = self.original_dir / self.split
        if not split_root.exists():
            LOGGER.warning("Split folder missing: %s", split_root)
            return

        report_map = _load_prep_report(self.cropped_dir) if self.use_report_filter else {}

        skip_missing_crop = 0
        skip_short = 0
        skip_report = 0

        for label_name, label in LABEL_MAP.items():
            source_label_dir = split_root / label_name
            if not source_label_dir.exists():
                continue

            for video_path in sorted(source_label_dir.iterdir()):
                if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue

                crop_dir = self.cropped_dir / label_name / video_path.stem
                if not crop_dir.exists():
                    skip_missing_crop += 1
                    continue

                frame_paths = _sorted_frames(crop_dir)
                if len(frame_paths) < self.min_frames:
                    skip_short += 1
                    continue

                rel = str((Path(self.split) / label_name / video_path.name).as_posix())
                if report_map:
                    report_row = report_map.get(rel)
                    if report_row is not None:
                        status = report_row.get("status", "")
                        fallback_ratio = float(report_row.get("fallback_ratio", 1.0))
                        if status != "ok" or fallback_ratio > self.max_fallback_ratio:
                            skip_report += 1
                            continue

                self.samples.append(
                    VideoSample(
                        split=self.split,
                        label_name=label_name,
                        label=label,
                        video_name=video_path.name,
                        frame_paths=frame_paths,
                    )
                )

        LOGGER.info(
            "Loaded split=%s, samples=%d, skipped_missing_crop=%d, skipped_short=%d, skipped_report=%d",
            self.split,
            len(self.samples),
            skip_missing_crop,
            skip_short,
            skip_report,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_indices(self, num_frames: int) -> List[int]:
        if num_frames <= 0:
            return [0] * self.num_segments

        if num_frames < self.num_segments:
            return np.linspace(0, num_frames - 1, self.num_segments).astype(np.int32).tolist()

        edges = np.linspace(0, num_frames, self.num_segments + 1).astype(np.int32)
        indices: List[int] = []
        for i in range(self.num_segments):
            start = int(edges[i])
            end = int(edges[i + 1])
            if end <= start:
                end = min(start + 1, num_frames)

            if self.is_train:
                indices.append(self.rng.randint(start, end - 1))
            else:
                indices.append((start + end - 1) // 2)

        return indices

    def __getitem__(self, index: int):
        sample = self.samples[index]
        frame_count = len(sample.frame_paths)
        indices = self._sample_indices(frame_count)

        frames = []
        for idx in indices:
            img_path = sample.frame_paths[min(max(0, idx), frame_count - 1)]
            with Image.open(img_path) as img:
                img = img.convert("RGB")
            frames.append(self.transform(img))

        return torch.stack(frames), sample.label


def _build_transforms(image_size: int):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Keep geometric consistency with preprocessing contract to reduce train-serving skew.
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_transform, val_transform


def get_dataloaders(
    orig_dir: Path = ORIGINAL_DIR,
    crop_dir: Path = CROPPED_DIR,
    batch_size: int = 8,
    num_segments: int = DEFAULT_NUM_SEGMENTS,
    image_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    min_frames: Optional[int] = None,
    max_fallback_ratio: float = 1.0,
    use_report_filter: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    if num_segments < 2:
        raise ValueError("num_segments must be >= 2")

    orig_dir = Path(orig_dir)
    crop_dir = Path(crop_dir)

    if image_size is None:
        image_size = _load_contract_image_size(crop_dir, fallback_size=224)
    if min_frames is None:
        min_frames = num_segments

    train_transform, val_transform = _build_transforms(image_size)

    train_dataset = RWF2000TSMDataset(
        original_dir=orig_dir,
        cropped_dir=crop_dir,
        split="train",
        num_segments=num_segments,
        transform=train_transform,
        min_frames=min_frames,
        max_fallback_ratio=max_fallback_ratio,
        use_report_filter=use_report_filter,
        seed=seed,
    )

    val_dataset = RWF2000TSMDataset(
        original_dir=orig_dir,
        cropped_dir=crop_dir,
        split="val",
        num_segments=num_segments,
        transform=val_transform,
        min_frames=min_frames,
        max_fallback_ratio=max_fallback_ratio,
        use_report_filter=use_report_filter,
        seed=seed,
    )

    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty. Check preprocessing output and report filters.")
    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty. Check preprocessing output and report filters.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    LOGGER.info(
        "DataLoader ready: train=%d, val=%d, image_size=%d, segments=%d",
        len(train_dataset),
        len(val_dataset),
        image_size,
        num_segments,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(batch_size=4, num_segments=8)
    for videos, labels in train_loader:
        print("videos:", videos.shape)
        print("labels:", labels.shape)
        break
    print("val batches:", len(val_loader))

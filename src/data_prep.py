from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".mpeg"}
DATA_SPLITS = ("train", "val")
DATA_LABELS = ("Fight", "NonFight")


@dataclass(frozen=True)
class VideoJob:
    split: str
    label: str
    video_path: Path


@dataclass
class CropConfig:
    model_path: Path
    input_dir: Path
    output_dir: Path
    image_size: int = 224
    person_conf: float = 0.35
    person_iou: float = 0.60
    padding_ratio: float = 0.20
    grace_frames: int = 8
    ema_alpha: float = 0.35
    min_box_side: int = 10
    tracker_cfg: str = "bytetrack.yaml"
    overwrite: bool = False


@dataclass
class VideoStats:
    status: str
    frames_read: int = 0
    frames_written: int = 0
    detect_frames: int = 0
    grace_frames: int = 0
    full_frame_fallback: int = 0
    message: str = ""


LOGGER = logging.getLogger("data_prep")


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def scan_jobs(input_dir: Path) -> List[VideoJob]:
    jobs: List[VideoJob] = []

    for split in DATA_SPLITS:
        for label in DATA_LABELS:
            folder = input_dir / split / label
            if not folder.exists():
                LOGGER.warning("Missing folder: %s", folder)
                continue

            for item in sorted(folder.iterdir()):
                if not item.is_file():
                    continue
                if item.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue
                jobs.append(VideoJob(split=split, label=label, video_path=item))

    jobs.sort(key=lambda x: (x.split, x.label, x.video_path.name.lower()))
    return jobs


def reset_tracker_state(model: YOLO) -> None:
    predictor = getattr(model, "predictor", None)
    if predictor is None:
        return

    trackers = getattr(predictor, "trackers", None)
    if trackers is None:
        return

    for tracker in trackers:
        reset_fn = getattr(tracker, "reset", None)
        if callable(reset_fn):
            reset_fn()


def get_person_union_box(result, frame_shape: Tuple[int, int, int]) -> Optional[Tuple[int, int, int, int]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return None

    coords = boxes.xyxy.detach().cpu().numpy()

    # Keep only person class even if class filtering was not applied upstream.
    if boxes.cls is not None:
        classes = boxes.cls.detach().cpu().numpy().astype(np.int32)
        coords = coords[classes == 0]

    if coords.shape[0] == 0:
        return None

    x1 = int(np.floor(np.min(coords[:, 0])))
    y1 = int(np.floor(np.min(coords[:, 1])))
    x2 = int(np.ceil(np.max(coords[:, 2])))
    y2 = int(np.ceil(np.max(coords[:, 3])))

    h_img, w_img = frame_shape[:2]
    x1 = max(0, min(x1, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    x2 = max(1, min(x2, w_img))
    y2 = max(1, min(y2, h_img))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def expand_and_clip(
    box: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int, int],
    padding_ratio: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    h_img, w_img = frame_shape[:2]

    w_box = max(1, x2 - x1)
    h_box = max(1, y2 - y1)
    pad_x = int(round(w_box * padding_ratio))
    pad_y = int(round(h_box * padding_ratio))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w_img, x2 + pad_x)
    y2 = min(h_img, y2 + pad_y)

    return x1, y1, x2, y2


def smooth_box(
    previous_box: Optional[Tuple[int, int, int, int]],
    current_box: Tuple[int, int, int, int],
    alpha: float,
    frame_shape: Tuple[int, int, int],
) -> Tuple[int, int, int, int]:
    if previous_box is None:
        return current_box

    alpha = float(np.clip(alpha, 0.0, 1.0))
    smoothed = [
        int(round(alpha * current_box[i] + (1.0 - alpha) * previous_box[i])) for i in range(4)
    ]

    h_img, w_img = frame_shape[:2]
    x1 = max(0, min(smoothed[0], w_img - 1))
    y1 = max(0, min(smoothed[1], h_img - 1))
    x2 = max(1, min(smoothed[2], w_img))
    y2 = max(1, min(smoothed[3], h_img))

    if x2 <= x1 or y2 <= y1:
        return current_box

    return x1, y1, x2, y2


def has_existing_frames(folder: Path) -> bool:
    return folder.exists() and any(folder.glob("*.jpg"))


def clear_existing_frames(folder: Path) -> None:
    for jpg_path in folder.glob("*.jpg"):
        jpg_path.unlink(missing_ok=True)


def validate_box(
    box: Tuple[int, int, int, int],
    min_box_side: int,
) -> bool:
    x1, y1, x2, y2 = box
    return (x2 - x1) >= min_box_side and (y2 - y1) >= min_box_side


def process_video(job: VideoJob, model: YOLO, cfg: CropConfig, save_folder: Path) -> VideoStats:
    cap = cv2.VideoCapture(str(job.video_path))
    if not cap.isOpened():
        return VideoStats(status="error", message="cannot_open_video")

    stats = VideoStats(status="ok")
    last_box: Optional[Tuple[int, int, int, int]] = None
    miss_count = 0

    reset_tracker_state(model)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            stats.frames_read += 1
            frame_h, frame_w = frame.shape[:2]

            try:
                results = model.track(
                    frame,
                    persist=True,
                    tracker=cfg.tracker_cfg,
                    conf=cfg.person_conf,
                    iou=cfg.person_iou,
                    classes=[0],
                    verbose=False,
                )
            except Exception as ex:
                return VideoStats(
                    status="error",
                    frames_read=stats.frames_read,
                    frames_written=stats.frames_written,
                    detect_frames=stats.detect_frames,
                    grace_frames=stats.grace_frames,
                    full_frame_fallback=stats.full_frame_fallback,
                    message=f"track_error:{ex}",
                )

            current_box = get_person_union_box(results[0], frame.shape)

            if current_box is not None:
                expanded = expand_and_clip(current_box, frame.shape, cfg.padding_ratio)
                if validate_box(expanded, cfg.min_box_side):
                    current_box = smooth_box(last_box, expanded, cfg.ema_alpha, frame.shape)
                    if validate_box(current_box, cfg.min_box_side):
                        last_box = current_box
                        miss_count = 0
                        stats.detect_frames += 1
                    else:
                        current_box = None
                else:
                    current_box = None

            if current_box is None:
                miss_count += 1
                if last_box is not None and miss_count <= cfg.grace_frames:
                    current_box = last_box
                    stats.grace_frames += 1
                else:
                    current_box = (0, 0, frame_w, frame_h)
                    stats.full_frame_fallback += 1
                    if miss_count > cfg.grace_frames:
                        last_box = None

            x1, y1, x2, y2 = current_box
            if x2 <= x1 or y2 <= y1:
                # Last safety fallback to avoid writing invalid crops.
                x1, y1, x2, y2 = 0, 0, frame_w, frame_h
                stats.full_frame_fallback += 1

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            resized = cv2.resize(crop, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_AREA)
            out_name = f"frame_{stats.frames_read:05d}.jpg"
            out_path = save_folder / out_name
            if cv2.imwrite(str(out_path), resized):
                stats.frames_written += 1

        if stats.frames_read == 0:
            stats.status = "error"
            stats.message = "empty_video"
    except Exception as ex:
        stats.status = "error"
        stats.message = f"runtime_error:{ex}"
    finally:
        cap.release()

    return stats


def write_report(report_path: Path, rows: Iterable[Dict[str, str]]) -> None:
    rows = list(rows)
    if not rows:
        return

    fieldnames = [
        "split",
        "label",
        "video_name",
        "video_rel_path",
        "status",
        "frames_read",
        "frames_written",
        "detect_frames",
        "grace_frames",
        "full_frame_fallback",
        "detect_ratio",
        "grace_ratio",
        "fallback_ratio",
        "message",
    ]

    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_crop_contract(contract_path: Path, cfg: CropConfig) -> None:
    contract = {
        "schema_version": "1.0",
        "purpose": "keep training and inference crop policy identical",
        "detector": {
            "framework": "ultralytics-yolo",
            "model_path": str(cfg.model_path),
            "class_filter": [0],
            "conf_threshold": cfg.person_conf,
            "iou_threshold": cfg.person_iou,
            "tracker": cfg.tracker_cfg,
        },
        "crop_policy": {
            "type": "person_union_box",
            "padding_ratio": cfg.padding_ratio,
            "ema_alpha": cfg.ema_alpha,
            "grace_frames": cfg.grace_frames,
            "min_box_side": cfg.min_box_side,
            "fallback": "full_frame",
        },
        "frame_policy": {
            "saved_size": [cfg.image_size, cfg.image_size],
            "file_pattern": "frame_%05d.jpg",
            "color_space_on_disk": "BGR",
        },
        "notes": [
            "Inference ROI crop must follow this contract to avoid training-serving skew.",
            "Temporal model sampler should use the same number of frames as training.",
        ],
    }

    with contract_path.open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2)


def run(cfg: CropConfig) -> int:
    if not cfg.model_path.exists():
        LOGGER.error("Model not found: %s", cfg.model_path)
        return 1

    if not cfg.input_dir.exists():
        LOGGER.error("Input dataset folder not found: %s", cfg.input_dir)
        return 1

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    jobs = scan_jobs(cfg.input_dir)
    if not jobs:
        LOGGER.error("No input videos were found under: %s", cfg.input_dir)
        return 1

    LOGGER.info("Found %d videos to process.", len(jobs))
    LOGGER.info("Loading model: %s", cfg.model_path)
    model = YOLO(str(cfg.model_path))

    report_rows: List[Dict[str, str]] = []
    success_count = 0

    for idx, job in enumerate(jobs, start=1):
        save_folder = cfg.output_dir / job.label / job.video_path.stem
        save_folder.mkdir(parents=True, exist_ok=True)

        if cfg.overwrite and save_folder.exists():
            clear_existing_frames(save_folder)

        if not cfg.overwrite and has_existing_frames(save_folder):
            LOGGER.info("[%d/%d] Skip existing %s", idx, len(jobs), job.video_path.name)
            report_rows.append(
                {
                    "split": job.split,
                    "label": job.label,
                    "video_name": job.video_path.name,
                    "video_rel_path": str(job.video_path.relative_to(cfg.input_dir)),
                    "status": "skipped_existing",
                    "frames_read": "0",
                    "frames_written": "0",
                    "detect_frames": "0",
                    "grace_frames": "0",
                    "full_frame_fallback": "0",
                    "detect_ratio": "0.0000",
                    "grace_ratio": "0.0000",
                    "fallback_ratio": "0.0000",
                    "message": "existing_frames",
                }
            )
            continue

        LOGGER.info("[%d/%d] Processing %s", idx, len(jobs), job.video_path.name)
        stats = process_video(job, model, cfg, save_folder)

        if stats.status == "ok" and stats.frames_written > 0:
            success_count += 1

        denom = max(stats.frames_read, 1)
        report_rows.append(
            {
                "split": job.split,
                "label": job.label,
                "video_name": job.video_path.name,
                "video_rel_path": str(job.video_path.relative_to(cfg.input_dir)),
                "status": stats.status,
                "frames_read": str(stats.frames_read),
                "frames_written": str(stats.frames_written),
                "detect_frames": str(stats.detect_frames),
                "grace_frames": str(stats.grace_frames),
                "full_frame_fallback": str(stats.full_frame_fallback),
                "detect_ratio": f"{stats.detect_frames / denom:.4f}",
                "grace_ratio": f"{stats.grace_frames / denom:.4f}",
                "fallback_ratio": f"{stats.full_frame_fallback / denom:.4f}",
                "message": stats.message,
            }
        )

    report_path = cfg.output_dir / "prep_report.csv"
    write_report(report_path, report_rows)

    contract_path = cfg.output_dir / "crop_contract.json"
    write_crop_contract(contract_path, cfg)

    LOGGER.info("Processing done: %d/%d successful videos", success_count, len(jobs))
    LOGGER.info("Report saved to: %s", report_path)
    LOGGER.info("Crop contract saved to: %s", contract_path)
    return 0


def parse_args() -> CropConfig:
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="RWF-2000 data preparation and person-centric crop pipeline")
    parser.add_argument("--model-path", type=Path, default=project_root / "src" / "yolo26n.pt")
    parser.add_argument("--input-dir", type=Path, default=project_root / "RWF-2000")
    parser.add_argument("--output-dir", type=Path, default=project_root / "RWF-2000-Cropped")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--person-conf", type=float, default=0.35)
    parser.add_argument("--person-iou", type=float, default=0.60)
    parser.add_argument("--padding-ratio", type=float, default=0.20)
    parser.add_argument("--grace-frames", type=int, default=8)
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--min-box-side", type=int, default=10)
    parser.add_argument("--tracker-cfg", type=str, default="bytetrack.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)

    return CropConfig(
        model_path=args.model_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        person_conf=args.person_conf,
        person_iou=args.person_iou,
        padding_ratio=args.padding_ratio,
        grace_frames=args.grace_frames,
        ema_alpha=args.ema_alpha,
        min_box_side=args.min_box_side,
        tracker_cfg=args.tracker_cfg,
        overwrite=args.overwrite,
    )


def main() -> None:
    cfg = parse_args()
    raise SystemExit(run(cfg))


if __name__ == "__main__":
    main()

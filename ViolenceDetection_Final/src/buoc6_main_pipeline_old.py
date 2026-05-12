from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from buoc3_model import MobileNetV3_TSM
from buoc5_kinematics import KinematicsGate

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TSM_WEIGHTS = PROJECT_ROOT / "weights" / "best_tsm_topdown_hn1.pth"
DEFAULT_YOLO_WEIGHTS = PROJECT_ROOT / "src" / "yolo26n.pt"
DEFAULT_TEST_VIDEO = Path(r"F:\NCKH\codeDot1\datasetthamkhao\SCVD_converted\Test\Violence\t_v006_converted.avi")
DEFAULT_OUTPUT_VIDEO = PROJECT_ROOT / "demo_lockon_widefix.mp4"
DEFAULT_NUM_SEGMENTS = 8

# ===== Pair/interaction evidence =====
INTERACTION_IOU_FOR_TRACK = 0.08
PAIR_IOU_MIN = 0.10
PAIR_IOU_HARD = 0.38
MIN_INTERACTION_FRAMES = 7

PROXIMITY_DIST_RATIO = 1.75
MIN_PROXIMITY_FRAMES = 7
APPROACH_SPEED_MIN = 1.8
PAIR_MIN_NET_MARGIN = 72.0
SCENE_MARGIN_REQUIRED = 55.0
MIN_IOU_FOR_CONTACT = 0.015
DIST_RATIO_STRICT_CONTACT = 0.95
PROXIMITY_REL_SPEED_MIN = 3.0
PROXIMITY_APPROACH_MIN = 3.2
NO_CONTACT_PENALTY = 145.0
NO_CONTACT_LOCK_BLOCK_DIST = 1.15
NO_CONTACT_LOCK_BLOCK_IOU = 0.02

# ===== Wide/merged target evidence =====
WIDE_BOX_RATIO = 1.12
WIDE_BOX_MIN_HEIGHT_RATIO = 0.07
WIDE_BOX_MIN_AREA_RATIO = 0.012
WIDE_BOX_STABLE_FRAMES = 10

MERGED_BOX_RATIO = 1.03
MERGED_BOX_MIN_AREA_RATIO = 0.018
MERGED_BOX_MIN_STABLE = 3

# ===== Lock-on control =====
MIN_ACTIVATION_SCORE = 340
LOCK_SWITCH_MARGIN = 70
MAX_TUBE_LIFE = 130
MAX_GRACE_PERIOD = 28
SCORE_DECAY_TRACKED = 2
SCORE_DECAY_GRACE = 7

REACQUIRE_SCORE_MIN = 300
REACQUIRE_DIST_FACTOR = 1.35
SINGLE_LOCK_MARGIN_BONUS = 40.0
SINGLE_LOCK_SCENE_BONUS = 26.0
SINGLE_LOCK_MIN_SPEED = 2.2
SINGLE_LOCK_BLOCK_CROWD = 3

# ===== Fallback ID assignment when tracker returns boxes without IDs =====
FALLBACK_TRACK_START_ID = 1_000_000
FALLBACK_TRACK_MAX_GAP = 14
FALLBACK_TRACK_PRUNE_GAP = 24
FALLBACK_MATCH_MIN_DIST = 60.0
FALLBACK_MATCH_DIST_FACTOR = 0.60

# ===== Human-shape filter for noisy class-compatible detections =====
HUMAN_MIN_HEIGHT_RATIO = 0.045
HUMAN_MIN_ASPECT = 0.22
HUMAN_MAX_ASPECT = 4.8
HUMAN_MIN_AREA_RATIO = 0.0007
HUMAN_MAX_AREA_RATIO = 0.95
LOW_CONF_BOX_THRESHOLD = 0.22
LOW_CONF_MIN_HEIGHT_RATIO = 0.055
ALT_CLASS_MIN_CONF = 0.20
ALT_CLASS_MIN_HEIGHT_RATIO = 0.055
ALT_CLASS_MIN_ASPECT = 0.22

# ===== Benign-motion suppression (crowded scenes, hugging, standing, sweeping) =====
STATIONARY_SPEED_PX = 3.6
IDLE_STATIONARY_FRAMES = 10
SYNC_MOTION_COS = 0.72
LOW_REL_SPEED_PX = 2.4
SEATED_HEIGHT_RATIO = 0.58
BENIGN_PENALTY_SCALE = 0.18
GATE_V_THRESHOLD = 11.5
GATE_A_THRESHOLD = 5.8
ALLOW_PROXIMITY_CONTACT = False

# ===== ROI/action tube =====
BASE_TUBE_SIZE = 380
MIN_TUBE_SIZE = 180
MAX_TUBE_SIZE = 1080
ROI_EXPAND_RATIO = 2.0
ROI_SIZE_SMOOTH = 0.4
SEARCH_EXPAND_MULT = 1.25
CONFIRM_SHRINK_MULT = 0.90
SEARCH_SIZE_SMOOTH = 0.4
CONFIRM_SIZE_SMOOTH = 0.6
SEARCH_CENTER_ALPHA = 0.65
CONFIRM_CENTER_ALPHA = 0.85
CONFIRM_ENTER_STREAK = 2
CONFIRM_EXIT_STREAK = 4
NO_CONTACT_RELEASE_FRAMES = 5
NO_CONTACT_SCORE_DECAY = 14
NO_CONTACT_LIFE_DECAY = 7
SINGLE_NO_GATE_RELEASE_FRAMES = 6
SINGLE_EXTRA_SCORE_DECAY = 7
SINGLE_EXTRA_LIFE_DECAY = 2
SINGLE_ALERT_GATE_FRAMES = 3
SINGLE_LOCK_MAX_FRAMES = 20
SINGLE_BURST_GATE_FRAMES = 2
SINGLE_BURST_MIN_SCORE = 580
SINGLE_BURST_MIN_MARGIN = 450
SINGLE_BURST_RAW_MIN = 0.20
SINGLE_BURST_RAW_MAX = 0.95
SELECTIVE_BURST_GATE_FRAMES = 6
SELECTIVE_BURST_MIN_SCORE = 740
SELECTIVE_BURST_MIN_MARGIN = 600
SELECTIVE_BURST_RAW_MIN = 0.22
SELECTIVE_BURST_RAW_MAX = 0.45
SELECTIVE_BURST_FUSED_MIN = 0.30
SELECTIVE_BURST_FUSED_MAX = 0.45
SELECTIVE_BURST_SMOOTH_MIN = 0.22
SELECTIVE_BURST_SMOOTH_MAX = 0.33
SELECTIVE_BURST_BENIGN_MIN = 120.0
SELECTIVE_BURST_BENIGN_MAX = 240.0
SELECTIVE_BURST_MIN_SINGLE_LOCK_RATIO = 0.70
SELECTIVE_BURST_MIN_LOCK_FRAMES = 24
SELECTIVE_BURST_MAX_TARGETS = 2
SELECTIVE_BURST_MAX_NO_CONTACT = 6
SELECTIVE_BURST_EXTRA_LOW_CONF = 3
SELECTIVE_COOLDOWN_FRAMES = 8
SWITCH_STALE_SCORE_GAP = 90
SWITCH_STALE_DISTANCE_FACTOR = 0.45
LOW_CONF_RELEASE_THRESHOLD = 0.30
LOW_CONF_RELEASE_FRAMES = 5

# ===== Fight decision/hysteresis =====
FIGHT_ON_THRESHOLD = 0.61
FIGHT_OFF_THRESHOLD = 0.41
ADAPTIVE_ON_MAX_REDUCTION = 0.10

MIN_VIOLENCE_STREAK = 1
MIN_NORMAL_STREAK = 7
STRONG_FIGHT_RAW_THRESHOLD = 0.86
STRONG_FIGHT_SCORE_THRESHOLD = 370

INFER_STRIDE = 2
EDGE_MARGIN = 20

# ===== Guarded full-frame rescue for weak single-lock crops =====
FULLFRAME_RESCUE_ENABLED = True
FULLFRAME_RESCUE_MIN_ACTIVE_SCORE = 0
FULLFRAME_RESCUE_MIN_SINGLE_GATE_STREAK = 2
FULLFRAME_RESCUE_MIN_LOCK_FRAMES = 0
FULLFRAME_RESCUE_MAX_BENIGN = 999.0
FULLFRAME_RESCUE_BLEND = 0.90

# ===== Auto crowd-adaptive tuning =====
CROWD_AUTO_ENABLED = True
CROWD_EMA_ALPHA = 0.15
CROWD_LOW = 3.0
CROWD_HIGH = 10.0
CROWD_ACTIVATION_BOOST = 85.0
CROWD_PAIR_MARGIN_BOOST = 45.0
CROWD_SCENE_MARGIN_BOOST = 40.0
CROWD_FIGHT_ON_BOOST = 0.06
CROWD_BENIGN_SCALE_BOOST = 0.55
CROWD_DISABLE_PROX_CONTACT_AT = 0.62

PROFILE_PRESETS = {
    "balanced": {},
    "school_park": {
        "MIN_ACTIVATION_SCORE": 430,
        "PAIR_MIN_NET_MARGIN": 120.0,
        "SCENE_MARGIN_REQUIRED": 110.0,
        "MIN_INTERACTION_FRAMES": 12,
        "MIN_PROXIMITY_FRAMES": 11,
        "APPROACH_SPEED_MIN": 2.8,
        "REACQUIRE_SCORE_MIN": 380,
        "FIGHT_ON_THRESHOLD": 0.67,
        "FIGHT_OFF_THRESHOLD": 0.44,
        "MIN_VIOLENCE_STREAK": 3,
        "MIN_NORMAL_STREAK": 6,
        "BENIGN_PENALTY_SCALE": 0.26,
        "SYNC_MOTION_COS": 0.60,
        "LOW_REL_SPEED_PX": 2.8,
        "SEARCH_EXPAND_MULT": 1.10,
        "CONFIRM_SHRINK_MULT": 0.84,
        "GATE_V_THRESHOLD": 12.5,
        "GATE_A_THRESHOLD": 6.2,
        "ALLOW_PROXIMITY_CONTACT": False,
    },
    "high_risk": {
        "MIN_ACTIVATION_SCORE": 340,
        "PAIR_MIN_NET_MARGIN": 65.0,
        "SCENE_MARGIN_REQUIRED": 55.0,
        "MIN_INTERACTION_FRAMES": 7,
        "MIN_PROXIMITY_FRAMES": 7,
        "APPROACH_SPEED_MIN": 1.6,
        "REACQUIRE_SCORE_MIN": 300,
        "FIGHT_ON_THRESHOLD": 0.56,
        "FIGHT_OFF_THRESHOLD": 0.38,
        "MIN_VIOLENCE_STREAK": 1,
        "MIN_NORMAL_STREAK": 4,
        "BENIGN_PENALTY_SCALE": 0.12,
        "SYNC_MOTION_COS": 0.80,
        "LOW_REL_SPEED_PX": 2.1,
        "SEARCH_EXPAND_MULT": 1.18,
        "CONFIRM_SHRINK_MULT": 0.90,
        "GATE_V_THRESHOLD": 10.5,
        "GATE_A_THRESHOLD": 5.0,
        "ALLOW_PROXIMITY_CONTACT": True,
    },
}

PROFILE_LABELS = {
    "balanced": "Balanced",
    "school_park": "SchoolPark",
    "high_risk": "HighRisk",
}


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def center_of_box(box: Sequence[int]) -> Tuple[float, float]:
    return ((box[1] + box[3]) / 2.0, (box[2] + box[4]) / 2.0)


def get_iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    x_a = max(box_a[1], box_b[1])
    y_a = max(box_a[2], box_b[2])
    x_b = min(box_a[3], box_b[3])
    y_b = min(box_a[4], box_b[4])

    inter_w = max(0, x_b - x_a)
    inter_h = max(0, y_b - y_a)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(1, (box_a[3] - box_a[1]) * (box_a[4] - box_a[2]))
    area_b = max(1, (box_b[3] - box_b[1]) * (box_b[4] - box_b[2]))
    return float(inter_area / max(1.0, area_a + area_b - inter_area))


def pair_distance_ratio(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    cax, cay = center_of_box(box_a)
    cbx, cby = center_of_box(box_b)
    distance = math.hypot(cax - cbx, cay - cby)

    h_a = max(1.0, float(box_a[4] - box_a[2]))
    h_b = max(1.0, float(box_b[4] - box_b[2]))
    ref = max(1.0, (h_a + h_b) / 2.0)
    return distance / ref


def union_box(boxes: Sequence[Sequence[int]]) -> Tuple[int, int, int, int]:
    x1 = min(int(box[1]) for box in boxes)
    y1 = min(int(box[2]) for box in boxes)
    x2 = max(int(box[3]) for box in boxes)
    y2 = max(int(box[4]) for box in boxes)
    return x1, y1, x2, y2


def make_pair_key(id_a: int, id_b: int) -> Tuple[int, int]:
    return (id_a, id_b) if id_a <= id_b else (id_b, id_a)


def roi_size_from_box(box: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    box_w = max(1.0, float(x2 - x1))
    box_h = max(1.0, float(y2 - y1))
    size = max(box_w, box_h) * ROI_EXPAND_RATIO
    return clamp(size, MIN_TUBE_SIZE, MAX_TUBE_SIZE)


def phase_scaled_roi_size(base_size: float, roi_mode: str) -> float:
    mult = SEARCH_EXPAND_MULT if roi_mode == "search" else CONFIRM_SHRINK_MULT
    return clamp(base_size * mult, MIN_TUBE_SIZE, MAX_TUBE_SIZE)


def focused_pair_roi_size(base_box: Tuple[int, int, int, int], roi_mode: str) -> float:
    x1, y1, x2, y2 = base_box
    span = max(1.0, float(max(x2 - x1, y2 - y1)))
    focus_mult = 1.55 if roi_mode == "search" else 1.35
    return clamp(span * focus_mult, MIN_TUBE_SIZE, MAX_TUBE_SIZE)


def resolve_track_ids(
    boxes_xyxy: Sequence[Sequence[float]],
    raw_ids: Optional[Sequence[float]],
    frame_idx: int,
    fallback_tracks: Dict[int, Tuple[float, float, int, int, int, int, int]],
    next_fallback_id: int,
) -> Tuple[List[int], int]:
    if raw_ids is not None and len(raw_ids) == len(boxes_xyxy):
        resolved = [int(v) for v in raw_ids]
        for tid, box in zip(resolved, boxes_xyxy):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            fallback_tracks[tid] = (cx, cy, x1, y1, x2, y2, frame_idx)

        for tid in list(fallback_tracks.keys()):
            if frame_idx - fallback_tracks[tid][6] > FALLBACK_TRACK_PRUNE_GAP:
                del fallback_tracks[tid]
        return resolved, next_fallback_id

    active_prev = {
        tid: value
        for tid, value in fallback_tracks.items()
        if frame_idx - value[6] <= FALLBACK_TRACK_MAX_GAP
    }

    used_prev: Set[int] = set()
    resolved: List[int] = [-1] * len(boxes_xyxy)

    for idx, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        box_h = max(1.0, float(y2 - y1))

        best_tid: Optional[int] = None
        best_cost = 1e9
        for tid, (pcx, pcy, px1, py1, px2, py2, _last_seen) in active_prev.items():
            if tid in used_prev:
                continue

            dist = math.hypot(cx - pcx, cy - pcy)
            if dist > max(FALLBACK_MATCH_MIN_DIST, FALLBACK_MATCH_DIST_FACTOR * box_h):
                continue

            prev_box = [tid, px1, py1, px2, py2]
            curr_box = [0, x1, y1, x2, y2]
            iou = get_iou(curr_box, prev_box)
            cost = dist - 30.0 * iou
            if cost < best_cost:
                best_cost = cost
                best_tid = tid

        if best_tid is None:
            resolved[idx] = next_fallback_id
            next_fallback_id += 1
        else:
            resolved[idx] = best_tid
            used_prev.add(best_tid)

    for tid, box in zip(resolved, boxes_xyxy):
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        fallback_tracks[tid] = (cx, cy, x1, y1, x2, y2, frame_idx)

    for tid in list(fallback_tracks.keys()):
        if frame_idx - fallback_tracks[tid][6] > FALLBACK_TRACK_PRUNE_GAP:
            del fallback_tracks[tid]

    return resolved, next_fallback_id


def pair_approach_speed(
    history_a: Optional[Deque[Tuple[float, float]]],
    history_b: Optional[Deque[Tuple[float, float]]],
) -> float:
    if history_a is None or history_b is None:
        return 0.0
    if len(history_a) < 2 or len(history_b) < 2:
        return 0.0

    prev_a = history_a[-2]
    prev_b = history_b[-2]
    curr_a = history_a[-1]
    curr_b = history_b[-1]

    prev_dist = math.hypot(prev_a[0] - prev_b[0], prev_a[1] - prev_b[1])
    curr_dist = math.hypot(curr_a[0] - curr_b[0], curr_a[1] - curr_b[1])
    return prev_dist - curr_dist


def velocity_from_history(history: Optional[Deque[Tuple[float, float]]]) -> Tuple[float, float, float]:
    if history is None or len(history) < 2:
        return 0.0, 0.0, 0.0

    vx = history[-1][0] - history[-2][0]
    vy = history[-1][1] - history[-2][1]
    speed = math.hypot(vx, vy)
    return vx, vy, speed


def cosine_similarity(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return clamp((v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2), -1.0, 1.0)


def apply_profile(profile_name: str) -> str:
    preset = PROFILE_PRESETS.get(profile_name)
    if preset is None:
        raise ValueError(f"Unsupported profile: {profile_name}")

    for key, value in preset.items():
        globals()[key] = value

    return PROFILE_LABELS.get(profile_name, profile_name)


def compute_crowd_factor(crowd_ema: float) -> float:
    if CROWD_HIGH <= CROWD_LOW:
        return 0.0
    return clamp01((crowd_ema - CROWD_LOW) / (CROWD_HIGH - CROWD_LOW))


def compute_dynamic_tuning(crowd_factor: float) -> Tuple[float, float, float, float, float, bool]:
    dynamic_min_activation = MIN_ACTIVATION_SCORE + CROWD_ACTIVATION_BOOST * crowd_factor
    dynamic_pair_margin = PAIR_MIN_NET_MARGIN + CROWD_PAIR_MARGIN_BOOST * crowd_factor
    dynamic_scene_margin = SCENE_MARGIN_REQUIRED + CROWD_SCENE_MARGIN_BOOST * crowd_factor
    dynamic_fight_on_base = FIGHT_ON_THRESHOLD + CROWD_FIGHT_ON_BOOST * crowd_factor
    dynamic_benign_scale = BENIGN_PENALTY_SCALE * (1.0 + CROWD_BENIGN_SCALE_BOOST * crowd_factor)
    dynamic_allow_prox_contact = ALLOW_PROXIMITY_CONTACT and crowd_factor < CROWD_DISABLE_PROX_CONTACT_AT

    return (
        dynamic_min_activation,
        dynamic_pair_margin,
        dynamic_scene_margin,
        dynamic_fight_on_base,
        dynamic_benign_scale,
        dynamic_allow_prox_contact,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Violence inference pipeline with robust lock-on for wide-angle fights")
    parser.add_argument("--tsm-weights", type=Path, default=DEFAULT_TSM_WEIGHTS)
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--video-path", type=Path, default=DEFAULT_TEST_VIDEO)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_VIDEO)
    parser.add_argument("--profile", choices=sorted(PROFILE_PRESETS.keys()), default="balanced")
    parser.add_argument("--disable-auto-crowd", action="store_true")
    parser.add_argument("--num-segments", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--det-conf", type=float, default=0.18)
    parser.add_argument("--person-classes", type=str, default="0,18,72")
    parser.add_argument("--detector-mode", choices=["predict", "track"], default="predict")
    parser.add_argument("--decision-mode", choices=["hybrid", "tsm_only"], default="tsm_only")
    parser.add_argument("--tsm-only-on-threshold", type=float, default=0.56)
    parser.add_argument("--tsm-only-off-threshold", type=float, default=0.40)
    parser.add_argument("--tsm-only-use-raw-boost", dest="tsm_only_use_raw_boost", action="store_true")
    parser.add_argument("--no-tsm-only-use-raw-boost", dest="tsm_only_use_raw_boost", action="store_false")
    parser.set_defaults(tsm_only_use_raw_boost=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--stats-json", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--disable-fullframe-rescue", action="store_true")
    return parser.parse_args()


def resolve_num_segments(args: argparse.Namespace) -> int:
    if args.num_segments is not None:
        if int(args.num_segments) < 2:
            raise ValueError("--num-segments must be >= 2")
        return int(args.num_segments)

    cfg_path = args.tsm_weights.parent / "train_config.json"
    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            value = int(payload.get("num_segments", DEFAULT_NUM_SEGMENTS))
            if value >= 2:
                return value
        except Exception:
            pass

    return DEFAULT_NUM_SEGMENTS


def main() -> None:
    args = parse_args()
    profile_label = apply_profile(args.profile)
    auto_crowd_enabled = CROWD_AUTO_ENABLED and (not args.disable_auto_crowd)
    fullframe_rescue_enabled = FULLFRAME_RESCUE_ENABLED and (not args.disable_fullframe_rescue)
    person_class_ids = {int(x.strip()) for x in args.person_classes.split(",") if x.strip()}
    if not person_class_ids:
        person_class_ids = {0}

    if not args.tsm_weights.exists():
        raise FileNotFoundError(f"TSM weights not found: {args.tsm_weights}")
    if not args.yolo_weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {args.yolo_weights}")
    if not args.video_path.exists():
        raise FileNotFoundError(f"Input video not found: {args.video_path}")

    if not (0.0 <= args.tsm_only_off_threshold < args.tsm_only_on_threshold <= 1.0):
        raise ValueError("Require 0 <= --tsm-only-off-threshold < --tsm-only-on-threshold <= 1")

    args.num_segments = resolve_num_segments(args)

    print(
        f"🚀 START PIPELINE: lock-on + wide-angle fix mode | "
        f"profile={profile_label} | auto_crowd={auto_crowd_enabled} | "
        f"detector={args.detector_mode} | segments={args.num_segments} | "
        f"decision={args.decision_mode} | fullframe_rescue={fullframe_rescue_enabled}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo_model = YOLO(str(args.yolo_weights))
    tsm_model = MobileNetV3_TSM(num_classes=2, num_segments=args.num_segments, pretrained=False)
    tsm_model.load_state_dict(torch.load(args.tsm_weights, map_location=device))
    tsm_model.to(device)
    tsm_model.eval()

    gate = KinematicsGate(v_threshold=GATE_V_THRESHOLD, a_threshold=GATE_A_THRESHOLD)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_path}")

    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out = None
    if not args.no_save:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(
            str(args.output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w_frame, h_frame),
        )

    frame_idx = 0

    frame_buffer: Deque[torch.Tensor] = deque(maxlen=args.num_segments)
    full_frame_buffer: Deque[torch.Tensor] = deque(maxlen=args.num_segments)
    interaction_tracker: Dict[Tuple[int, int], int] = defaultdict(int)
    proximity_tracker: Dict[Tuple[int, int], int] = defaultdict(int)
    wide_box_tracker: Dict[int, int] = defaultdict(int)
    motion_history: Dict[int, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=4))
    stationary_counter: Dict[int, int] = defaultdict(int)
    fallback_tracks: Dict[int, Tuple[float, float, int, int, int, int, int]] = {}
    next_fallback_id = FALLBACK_TRACK_START_ID

    active_tube_center: Optional[Tuple[float, float]] = None
    active_target_ids: Set[int] = set()
    active_tube_size = float(BASE_TUBE_SIZE)
    tube_life = 0
    grace_period_counter = 0
    current_active_score = 0
    current_benign_score = 0.0
    active_no_contact_streak = 0
    active_low_conf_streak = 0
    active_single_gate_streak = 0
    active_single_no_gate_streak = 0
    active_single_lock_age = 0
    selective_cooldown = 0

    raw_fight_prob = 0.0
    fused_prob = 0.0
    smooth_prob = 0.0
    on_threshold_now = FIGHT_ON_THRESHOLD
    roi_mode = "search"
    roi_confirm_streak = 0
    roi_release_streak = 0
    tsm_prob_history: Deque[float] = deque(maxlen=5)

    crowd_ema = 0.0
    crowd_factor = 0.0
    (
        dynamic_min_activation,
        dynamic_pair_margin,
        dynamic_scene_margin,
        dynamic_fight_on_base,
        dynamic_benign_scale,
        dynamic_allow_prox_contact,
    ) = compute_dynamic_tuning(crowd_factor)

    alarm_on = False
    violence_streak = 0
    normal_streak = 0
    total_frames = 0
    alert_frames = 0
    lock_frames = 0
    single_lock_frames = 0
    max_lock_duration = 0
    current_lock_duration = 0
    det_count_sum = 0
    frames_with_track_data = 0
    incident_candidate_frames = 0
    single_candidate_frames = 0
    single_candidate_reject_frames = 0
    candidate_gate_pass_frames = 0
    lock_switch_events = 0
    alarm_on_events = 0
    prev_alarm_on = False
    max_active_score_observed = 0
    max_single_gate_streak_observed = 0
    max_raw_fight_prob_observed = 0.0
    max_fullframe_prob_observed = 0.0
    max_fused_prob_observed = 0.0
    max_smooth_prob_observed = 0.0
    max_active_margin_observed = 0.0
    high_conf_fight_frames = 0
    strong_kinematic_override_frames = 0
    single_target_burst_override_frames = 0
    single_target_selective_override_frames = 0
    selective_core_frames = 0
    selective_core_sg_frames = 0
    selective_core_lockshape_frames = 0
    selective_core_context_frames = 0
    selective_core_sg_target_frames = 0
    selective_core_sg_target_contact_frames = 0
    selective_core_first_lock_frames = -1
    selective_core_first_single_lock_ratio = 0.0
    selective_core_max_single_lock_ratio = 0.0
    score_gate_frames = 0
    tsm_only_trigger_frames = 0
    fullframe_rescue_frames = 0
    fullframe_rescue_eval_frames = 0
    max_single_gate_streak_at_infer = 0
    max_active_score_at_infer = 0
    infer_frames_count = 0
    rescue_gate_streak_ok_frames = 0
    rescue_active_score_ok_frames = 0
    rescue_benign_ok_frames = 0
    rescue_all_ok_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        total_frames += 1
        if args.max_frames > 0 and total_frames > args.max_frames:
            break
        raw_frame = frame
        vis_frame = frame.copy()

        best_incident_center: Optional[Tuple[float, float]] = None
        best_incident_ids: Set[int] = set()
        best_incident_box: Optional[Tuple[int, int, int, int]] = None
        best_incident_contact = False
        highest_frame_score = 0.0
        highest_incident_margin = -1e9
        best_incident_benign = 0.0
        best_incident_has_gate = False
        best_incident_wide_hint = False
        best_incident_speed = 0.0
        frame_best_benign = 0.0
        display_scene_margin = 0.0

        if args.detector_mode == "track":
            results = yolo_model.track(
                raw_frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                conf=args.det_conf,
                classes=sorted(person_class_ids),
                imgsz=args.imgsz,
            )
        else:
            results = yolo_model.predict(
                raw_frame,
                verbose=False,
                conf=args.det_conf,
                classes=sorted(person_class_ids),
                imgsz=args.imgsz,
            )

        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []
        raw_ids = None
        if args.detector_mode == "track":
            raw_ids = results[0].boxes.id.cpu().numpy() if (results[0].boxes and results[0].boxes.id is not None) else None
        confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes else []
        cls_ids = results[0].boxes.cls.cpu().numpy() if results[0].boxes else []

        track_data: List[List[int]] = []
        if len(boxes) == 0:
            for tid in list(fallback_tracks.keys()):
                if frame_idx - fallback_tracks[tid][6] > FALLBACK_TRACK_PRUNE_GAP:
                    del fallback_tracks[tid]

        if len(boxes) > 0:
            resolved_ids, next_fallback_id = resolve_track_ids(
                boxes,
                raw_ids,
                frame_idx,
                fallback_tracks,
                next_fallback_id,
            )

            current_ids: Set[int] = set()
            track_velocity: Dict[int, Tuple[float, float]] = {}
            track_speed: Dict[int, float] = {}

            for box, track_id, conf, cls_id in zip(boxes, resolved_ids, confs, cls_ids):
                x1, y1, x2, y2 = map(int, box)
                box_w = max(1.0, float(x2 - x1))
                box_h = max(1.0, float(y2 - y1))
                aspect = box_h / max(1.0, box_w)
                area_ratio = (box_w * box_h) / max(1.0, float(w_frame * h_frame))

                if box_h < h_frame * HUMAN_MIN_HEIGHT_RATIO:
                    continue
                if area_ratio < HUMAN_MIN_AREA_RATIO or area_ratio > HUMAN_MAX_AREA_RATIO:
                    continue
                if aspect < HUMAN_MIN_ASPECT or aspect > HUMAN_MAX_ASPECT:
                    continue
                if float(conf) < LOW_CONF_BOX_THRESHOLD and box_h < h_frame * LOW_CONF_MIN_HEIGHT_RATIO:
                    continue

                # Class 0 is true person; compatible proxy classes are kept only with tighter shape/conf constraints.
                if int(cls_id) != 0:
                    if float(conf) < ALT_CLASS_MIN_CONF:
                        continue
                    if box_h < h_frame * ALT_CLASS_MIN_HEIGHT_RATIO:
                        continue
                    if aspect < ALT_CLASS_MIN_ASPECT:
                        continue

                tid = int(track_id)
                current_ids.add(tid)
                track_data.append([tid, x1, y1, x2, y2])

                cx, cy = center_of_box([tid, x1, y1, x2, y2])
                motion_history[tid].append((cx, cy))

                vx, vy, raw_speed = velocity_from_history(motion_history[tid])
                
                # Normalize velocity based on person height (baseline 100px)
                height_scale = max(1.0, box_h / 100.0)
                vx_norm = vx / height_scale
                vy_norm = vy / height_scale
                speed_norm = raw_speed / height_scale

                track_velocity[tid] = (vx_norm, vy_norm)
                track_speed[tid] = speed_norm
                if speed_norm < STATIONARY_SPEED_PX:
                    stationary_counter[tid] += 1
                else:
                    stationary_counter[tid] = max(0, stationary_counter[tid] - 1)

                is_fighting = prev_alarm_on and (tid in active_target_ids)
                box_color = (0, 0, 255) if is_fighting else (255, 0, 0)
                text_color = (0, 0, 255) if is_fighting else (0, 255, 0)

                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), box_color, 2)
                # Vẽ thêm nhãn và ID để nhận diện người
                label_text = f"ID: {tid} ({conf*100:.0f}%)"
                cv2.putText(vis_frame, label_text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

            for tid in list(motion_history.keys()):
                if tid not in current_ids:
                    del motion_history[tid]
            for tid in list(stationary_counter.keys()):
                if tid not in current_ids:
                    del stationary_counter[tid]

            current_iou_pairs = set()
            current_close_pairs = set()

            for i in range(len(track_data)):
                for j in range(i + 1, len(track_data)):
                    box_a = track_data[i]
                    box_b = track_data[j]
                    pair_id = make_pair_key(box_a[0], box_b[0])

                    iou = get_iou(box_a, box_b)
                    if iou > INTERACTION_IOU_FOR_TRACK:
                        interaction_tracker[pair_id] += 1
                        current_iou_pairs.add(pair_id)

                    dist_ratio = pair_distance_ratio(box_a, box_b)
                    if dist_ratio < PROXIMITY_DIST_RATIO:
                        proximity_tracker[pair_id] += 1
                        current_close_pairs.add(pair_id)

            for pair in list(interaction_tracker.keys()):
                if pair not in current_iou_pairs:
                    interaction_tracker[pair] = max(0, interaction_tracker[pair] - 2)
                    if interaction_tracker[pair] == 0:
                        del interaction_tracker[pair]

            for pair in list(proximity_tracker.keys()):
                if pair not in current_close_pairs:
                    proximity_tracker[pair] = max(0, proximity_tracker[pair] - 2)
                    if proximity_tracker[pair] == 0:
                        del proximity_tracker[pair]

            if auto_crowd_enabled:
                crowd_raw = float(len(current_ids) + 0.35 * len(current_close_pairs) + 0.55 * len(current_iou_pairs))
                crowd_ema = (1.0 - CROWD_EMA_ALPHA) * crowd_ema + CROWD_EMA_ALPHA * crowd_raw
                crowd_factor = compute_crowd_factor(crowd_ema)
            else:
                crowd_ema = max(0.0, crowd_ema * 0.92)
                crowd_factor = 0.0

            (
                dynamic_min_activation,
                dynamic_pair_margin,
                dynamic_scene_margin,
                dynamic_fight_on_base,
                dynamic_benign_scale,
                dynamic_allow_prox_contact,
            ) = compute_dynamic_tuning(crowd_factor)

            suspicious_ids = gate.update_and_check(track_data)

            current_wide_ids = set()
            for box in track_data:
                tid, x1, y1, x2, y2 = box
                box_w = max(1, x2 - x1)
                box_h = max(1, y2 - y1)
                area_ratio = (box_w * box_h) / max(1.0, float(w_frame * h_frame))

                is_at_edge = (
                    x1 <= EDGE_MARGIN
                    or y1 <= EDGE_MARGIN
                    or x2 >= w_frame - EDGE_MARGIN
                    or y2 >= h_frame - EDGE_MARGIN
                )

                if (
                    box_w > box_h * WIDE_BOX_RATIO
                    and box_h > h_frame * WIDE_BOX_MIN_HEIGHT_RATIO
                    and area_ratio > WIDE_BOX_MIN_AREA_RATIO
                    and not is_at_edge
                ):
                    wide_box_tracker[tid] += 1
                    current_wide_ids.add(tid)

            for tid in list(wide_box_tracker.keys()):
                if tid not in current_wide_ids:
                    wide_box_tracker[tid] = max(0, wide_box_tracker[tid] - 2)
                    if wide_box_tracker[tid] == 0:
                        del wide_box_tracker[tid]

            for box in track_data:
                tid, x1, y1, x2, y2 = box
                box_w = max(1, x2 - x1)
                box_h = max(1, y2 - y1)
                area_ratio = (box_w * box_h) / max(1.0, float(w_frame * h_frame))
                has_wide_gate = tid in suspicious_ids
                has_very_strong_wide = wide_box_tracker.get(tid, 0) >= (WIDE_BOX_STABLE_FRAMES + 6)
                has_large_merged = (
                    wide_box_tracker.get(tid, 0) >= max(6, WIDE_BOX_STABLE_FRAMES - 2)
                    and area_ratio > 0.026
                    and box_w > box_h * 1.08
                )
                if wide_box_tracker.get(tid, 0) >= WIDE_BOX_STABLE_FRAMES and (has_wide_gate or has_very_strong_wide or has_large_merged):
                    aggressive_score = 760.0
                    benign_score = 0.0
                    if stationary_counter.get(tid, 0) >= IDLE_STATIONARY_FRAMES:
                        benign_score += 220.0
                    if not has_wide_gate:
                        benign_score += 80.0 if has_large_merged else 100.0
                    margin = aggressive_score - benign_score

                    if margin >= dynamic_pair_margin and margin > highest_incident_margin:
                        highest_incident_margin = margin
                        highest_frame_score = aggressive_score
                        best_incident_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        best_incident_ids = {tid}
                        best_incident_box = (x1, y1, x2, y2)
                        best_incident_benign = benign_score
                        best_incident_contact = True
                        best_incident_has_gate = has_wide_gate
                        best_incident_wide_hint = True
                        best_incident_speed = track_speed.get(tid, 0.0)
                        
                    frame_best_benign = max(frame_best_benign, benign_score)

            # Sparse-detector fallback: one merged person-like box can still represent a multi-person scuffle.
            if len(track_data) <= 1:
                for box in track_data:
                    tid, x1, y1, x2, y2 = box
                    box_w = max(1, x2 - x1)
                    box_h = max(1, y2 - y1)
                    area_ratio = (box_w * box_h) / max(1.0, float(w_frame * h_frame))
                    speed = track_speed.get(tid, 0.0)
                    gate_hit = tid in suspicious_ids
                    wide_hit = wide_box_tracker.get(tid, 0) >= max(4, WIDE_BOX_STABLE_FRAMES - 4)

                    if box_h < h_frame * 0.065 or area_ratio < 0.008:
                        continue

                    aggressive_score = 320.0
                    aggressive_score += min(140.0, max(0.0, speed - 1.0) * 32.0)
                    if gate_hit:
                        aggressive_score += 75.0
                    if wide_hit:
                        aggressive_score += 60.0

                    benign_score = 0.0
                    if stationary_counter.get(tid, 0) >= IDLE_STATIONARY_FRAMES:
                        benign_score += 170.0
                    if (not gate_hit) and speed < 1.25:
                        benign_score += 90.0

                    margin = aggressive_score - benign_score
                    
                    if (
                        margin >= max(40.0, dynamic_pair_margin - 18.0)
                        and aggressive_score >= dynamic_min_activation - 35.0
                        and margin > highest_incident_margin
                    ):
                        highest_incident_margin = margin
                        highest_frame_score = aggressive_score
                        best_incident_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        best_incident_ids = {tid}
                        best_incident_box = (x1, y1, x2, y2)
                        best_incident_benign = benign_score
                        best_incident_contact = True
                        best_incident_has_gate = gate_hit
                        best_incident_wide_hint = wide_hit
                        best_incident_speed = speed
                        
                    frame_best_benign = max(frame_best_benign, benign_score)

            # Pair scoring works even when people do not overlap in wide-angle camera views.
            for i in range(len(track_data)):
                for j in range(i + 1, len(track_data)):
                    box_a = track_data[i]
                    box_b = track_data[j]
                    id_a, id_b = box_a[0], box_b[0]

                    pair_id = make_pair_key(id_a, id_b)
                    iou = get_iou(box_a, box_b)
                    dist_ratio = pair_distance_ratio(box_a, box_b)
                    pair_iou_frames = interaction_tracker.get(pair_id, 0)
                    pair_close_frames = proximity_tracker.get(pair_id, 0)
                    approach_speed = pair_approach_speed(motion_history.get(id_a), motion_history.get(id_b))
                    vel_a = track_velocity.get(id_a, (0.0, 0.0))
                    vel_b = track_velocity.get(id_b, (0.0, 0.0))
                    speed_a = track_speed.get(id_a, 0.0)
                    speed_b = track_speed.get(id_b, 0.0)
                    rel_speed = math.hypot(vel_a[0] - vel_b[0], vel_a[1] - vel_b[1])
                    dir_cos = cosine_similarity(vel_a, vel_b)
                    stationary_a = stationary_counter.get(id_a, 0)
                    stationary_b = stationary_counter.get(id_b, 0)

                    has_gate_evidence = id_a in suspicious_ids or id_b in suspicious_ids
                    iou_evidence = (iou > PAIR_IOU_MIN and pair_iou_frames >= MIN_INTERACTION_FRAMES) or iou > PAIR_IOU_HARD
                    close_evidence = pair_close_frames >= MIN_PROXIMITY_FRAMES and dist_ratio < PROXIMITY_DIST_RATIO
                    motion_evidence = approach_speed > APPROACH_SPEED_MIN

                    contact_by_iou = iou >= MIN_IOU_FOR_CONTACT
                    contact_by_proximity = (
                        dynamic_allow_prox_contact
                        and
                        close_evidence
                        and has_gate_evidence
                        and dist_ratio <= DIST_RATIO_STRICT_CONTACT
                        and rel_speed >= PROXIMITY_REL_SPEED_MIN
                        and approach_speed >= PROXIMITY_APPROACH_MIN
                        and pair_close_frames >= MIN_PROXIMITY_FRAMES + 2
                    )
                    contact_evidence = contact_by_iou or contact_by_proximity

                    benign_score = 0.0
                    if pair_close_frames >= MIN_PROXIMITY_FRAMES and rel_speed < LOW_REL_SPEED_PX and not has_gate_evidence:
                        benign_score += 120.0
                    if dir_cos > SYNC_MOTION_COS and rel_speed < LOW_REL_SPEED_PX:
                        benign_score += 110.0
                    if stationary_a >= IDLE_STATIONARY_FRAMES and stationary_b >= IDLE_STATIONARY_FRAMES:
                        benign_score += 150.0
                    elif stationary_a >= IDLE_STATIONARY_FRAMES or stationary_b >= IDLE_STATIONARY_FRAMES:
                        benign_score += 80.0

                    h_a = max(1.0, float(box_a[4] - box_a[2]))
                    h_b = max(1.0, float(box_b[4] - box_b[2]))
                    min_h = min(h_a, h_b)
                    max_h = max(h_a, h_b)
                    if min_h < SEATED_HEIGHT_RATIO * max_h and (stationary_a >= IDLE_STATIONARY_FRAMES or stationary_b >= IDLE_STATIONARY_FRAMES):
                        benign_score += 120.0
                    
                    if stationary_a >= IDLE_STATIONARY_FRAMES and stationary_b >= IDLE_STATIONARY_FRAMES:
                        benign_score += 450.0  # Massive penalty if BOTH are completely stationary

                    if max(speed_a, speed_b) < 4.5 and iou > 0.15:
                        benign_score += 350.0  # Massive penalty for people sitting/standing together quietly

                    if approach_speed < 1.0 and not has_gate_evidence:
                        benign_score += 60.0
                    if abs(speed_a - speed_b) < 1.2 and dir_cos > 0.65:
                        benign_score += 45.0
                    if pair_iou_frames < 2 and pair_close_frames >= MIN_PROXIMITY_FRAMES and not has_gate_evidence:
                        benign_score += 45.0
                    if pair_close_frames >= MIN_PROXIMITY_FRAMES and dist_ratio > NO_CONTACT_LOCK_BLOCK_DIST and iou < NO_CONTACT_LOCK_BLOCK_IOU:
                        benign_score += 90.0
                    if not contact_evidence:
                        benign_score += NO_CONTACT_PENALTY

                    frame_best_benign = max(frame_best_benign, benign_score)

                    if not (iou_evidence or (close_evidence and (has_gate_evidence or motion_evidence))):
                        continue

                    if not contact_evidence:
                        continue

                    aggressive_score = 360.0
                    aggressive_score += min(110.0, pair_iou_frames * 6.0)
                    aggressive_score += min(90.0, pair_close_frames * 5.0)
                    aggressive_score += min(130.0, max(0.0, approach_speed) * 15.0)
                    aggressive_score += min(90.0, max(0.0, rel_speed - 1.5) * 10.0)
                    if has_gate_evidence:
                        aggressive_score += 60.0
                    if dist_ratio < 1.20:
                        aggressive_score += 25.0
                    if contact_by_iou:
                        aggressive_score += 80.0
                    elif contact_by_proximity:
                        aggressive_score += 35.0

                    margin = aggressive_score - benign_score

                    if margin >= dynamic_pair_margin and margin > highest_incident_margin:
                        highest_incident_margin = margin
                        highest_frame_score = aggressive_score
                        ca = center_of_box(box_a)
                        cb = center_of_box(box_b)
                        best_incident_center = ((ca[0] + cb[0]) / 2.0, (ca[1] + cb[1]) / 2.0)
                        best_incident_ids = {id_a, id_b}
                        best_incident_box = union_box([box_a, box_b])
                        best_incident_benign = benign_score
                        best_incident_contact = contact_evidence
                        best_incident_has_gate = has_gate_evidence
                        best_incident_wide_hint = False
                        best_incident_speed = max(speed_a, speed_b)

            # Debug at end of frame
            if frame_idx % 10 == 0:
                pass

            for sid in suspicious_ids:
                box_s = next((item for item in track_data if item[0] == sid), None)
                if box_s is None:
                    continue

                box_w = max(1, box_s[3] - box_s[1])
                box_h = max(1, box_s[4] - box_s[2])
                area_ratio = (box_w * box_h) / max(1.0, float(w_frame * h_frame))

                if (
                    box_w > box_h * MERGED_BOX_RATIO
                    and area_ratio > MERGED_BOX_MIN_AREA_RATIO
                    and wide_box_tracker.get(sid, 0) >= MERGED_BOX_MIN_STABLE
                ):
                    aggressive_score = 650.0
                    benign_score = 0.0
                    if stationary_counter.get(sid, 0) >= IDLE_STATIONARY_FRAMES:
                        benign_score += 120.0
                    margin = aggressive_score - benign_score
                    frame_best_benign = max(frame_best_benign, benign_score)
                    if margin > highest_incident_margin:
                        highest_incident_margin = margin

                    if margin >= dynamic_pair_margin and aggressive_score > highest_frame_score:
                        highest_frame_score = aggressive_score
                        best_incident_center = ((box_s[1] + box_s[3]) / 2.0, (box_s[2] + box_s[4]) / 2.0)
                        best_incident_ids = {sid}
                        best_incident_box = (box_s[1], box_s[2], box_s[3], box_s[4])
                        best_incident_benign = benign_score
                        best_incident_contact = True
                        best_incident_has_gate = True
                        best_incident_wide_hint = True
                        best_incident_speed = track_speed.get(sid, 0.0)

            scene_margin = highest_frame_score - frame_best_benign
            display_scene_margin = scene_margin
            single_target_candidate = len(best_incident_ids) == 1
            if best_incident_center is not None and highest_frame_score > 0:
                incident_candidate_frames += 1
            if best_incident_center is not None and single_target_candidate:
                single_candidate_frames += 1
            single_lock_allowed = True
            if single_target_candidate:
                single_lock_allowed = (
                    highest_incident_margin >= dynamic_pair_margin + SINGLE_LOCK_MARGIN_BONUS
                    and scene_margin >= dynamic_scene_margin + SINGLE_LOCK_SCENE_BONUS
                    and best_incident_speed >= SINGLE_LOCK_MIN_SPEED
                    and (best_incident_has_gate or best_incident_wide_hint)
                )

                if (
                    len(track_data) >= SINGLE_LOCK_BLOCK_CROWD
                    and (not best_incident_has_gate)
                    and best_incident_speed < (SINGLE_LOCK_MIN_SPEED + 0.4)
                ):
                    single_lock_allowed = False
                if not single_lock_allowed:
                    single_candidate_reject_frames += 1

            if (
                best_incident_center
                and highest_frame_score >= dynamic_min_activation
                and highest_incident_margin >= dynamic_pair_margin
                and scene_margin >= dynamic_scene_margin
                and (len(best_incident_ids) < 2 or best_incident_contact)
                and single_lock_allowed
            ):
                candidate_gate_pass_frames += 1
                stale_lock = (
                    active_low_conf_streak >= 2
                    or active_no_contact_streak >= 2
                    or current_benign_score >= 230.0
                )
                candidate_far = False
                if active_tube_center is not None:
                    cand_dist = math.hypot(
                        best_incident_center[0] - active_tube_center[0],
                        best_incident_center[1] - active_tube_center[1],
                    )
                    candidate_far = cand_dist >= max(80.0, active_tube_size * SWITCH_STALE_DISTANCE_FACTOR)

                should_switch = (
                    active_tube_center is None
                    or highest_frame_score >= current_active_score + LOCK_SWITCH_MARGIN
                    or (
                        stale_lock
                        and candidate_far
                        and highest_frame_score >= current_active_score - SWITCH_STALE_SCORE_GAP
                    )
                )

                if should_switch:
                    active_tube_center = best_incident_center
                    active_target_ids = set(best_incident_ids)
                    tube_life = MAX_TUBE_LIFE
                    grace_period_counter = MAX_GRACE_PERIOD
                    current_active_score = int(highest_frame_score)
                    current_benign_score = best_incident_benign
                    roi_mode = "search"
                    roi_confirm_streak = 0
                    roi_release_streak = 0
                    active_low_conf_streak = 0
                    active_single_gate_streak = 0
                    active_single_no_gate_streak = 0
                    active_single_lock_age = 0
                    tsm_prob_history.clear()

                    if best_incident_box is not None:
                        desired_size = phase_scaled_roi_size(roi_size_from_box(best_incident_box), roi_mode)
                        if len(best_incident_ids) >= 2:
                            desired_size = min(desired_size, focused_pair_roi_size(best_incident_box, roi_mode))
                        active_tube_size = 0.40 * desired_size + 0.60 * active_tube_size
                    lock_switch_events += 1

            if active_tube_center is not None and tube_life > 0:
                target_boxes = [box for box in track_data if box[0] in active_target_ids]

                if target_boxes:
                    target_union = union_box(target_boxes)
                    target_center = (
                        (target_union[0] + target_union[2]) / 2.0,
                        (target_union[1] + target_union[3]) / 2.0,
                    )

                    center_alpha = SEARCH_CENTER_ALPHA if roi_mode == "search" else CONFIRM_CENTER_ALPHA

                    active_tube_center = (
                        center_alpha * target_center[0] + (1.0 - center_alpha) * active_tube_center[0],
                        center_alpha * target_center[1] + (1.0 - center_alpha) * active_tube_center[1],
                    )

                    desired_size = phase_scaled_roi_size(roi_size_from_box(target_union), roi_mode)
                    if len(active_target_ids) >= 2:
                        desired_size = min(desired_size, focused_pair_roi_size(target_union, roi_mode))
                    size_smooth = SEARCH_SIZE_SMOOTH if roi_mode == "search" else CONFIRM_SIZE_SMOOTH
                    active_tube_size = size_smooth * desired_size + (1.0 - size_smooth) * active_tube_size

                    tube_life -= 1
                    grace_period_counter = MAX_GRACE_PERIOD
                    current_active_score = max(0, current_active_score - SCORE_DECAY_TRACKED)
                    current_benign_score = max(0.0, current_benign_score - 3.0)

                    if len(active_target_ids) >= 2 and len(target_boxes) >= 2:
                        max_target_iou = 0.0
                        for i in range(len(target_boxes)):
                            for j in range(i + 1, len(target_boxes)):
                                max_target_iou = max(max_target_iou, get_iou(target_boxes[i], target_boxes[j]))

                        if max_target_iou < MIN_IOU_FOR_CONTACT:
                            active_no_contact_streak += 1
                            current_active_score = max(0, current_active_score - NO_CONTACT_SCORE_DECAY)
                            tube_life = max(0, tube_life - NO_CONTACT_LIFE_DECAY)
                        else:
                            active_no_contact_streak = 0
                    else:
                        active_no_contact_streak = 0

                    if len(active_target_ids) == 1 and len(target_boxes) == 1:
                        active_single_lock_age += 1
                        single_tid = target_boxes[0][0]
                        if single_tid in suspicious_ids:
                            active_single_gate_streak = min(active_single_gate_streak + 1, 30)
                            active_single_no_gate_streak = 0
                        else:
                            active_single_no_gate_streak = min(active_single_no_gate_streak + 1, SINGLE_NO_GATE_RELEASE_FRAMES + 3)
                            active_single_gate_streak = max(0, active_single_gate_streak - 1)
                            current_active_score = max(0, current_active_score - SINGLE_EXTRA_SCORE_DECAY)
                            tube_life = max(0, tube_life - SINGLE_EXTRA_LIFE_DECAY)

                        if active_single_no_gate_streak >= SINGLE_NO_GATE_RELEASE_FRAMES:
                            tube_life = 0
                            active_tube_center = None
                            active_target_ids = set()
                            current_active_score = 0
                            current_benign_score = 0.0
                            active_no_contact_streak = 0
                            active_low_conf_streak = 0
                            active_single_gate_streak = 0
                            active_single_no_gate_streak = 0
                            active_single_lock_age = 0
                            frame_buffer.clear()
                            full_frame_buffer.clear()

                        if active_single_lock_age >= SINGLE_LOCK_MAX_FRAMES:
                            tube_life = 0
                            active_tube_center = None
                            active_target_ids = set()
                            current_active_score = 0
                            current_benign_score = 0.0
                            active_no_contact_streak = 0
                            active_low_conf_streak = 0
                            active_single_gate_streak = 0
                            active_single_no_gate_streak = 0
                            active_single_lock_age = 0
                            frame_buffer.clear()
                            full_frame_buffer.clear()
                    else:
                        active_single_gate_streak = 0
                        active_single_no_gate_streak = 0
                        active_single_lock_age = 0

                    active_low_conf_streak = max(0, active_low_conf_streak - 1)

                    if active_no_contact_streak >= NO_CONTACT_RELEASE_FRAMES:
                        tube_life = 0
                        active_tube_center = None
                        active_target_ids = set()
                        current_active_score = 0
                        current_benign_score = 0.0
                        active_no_contact_streak = 0
                        active_low_conf_streak = 0
                        active_single_gate_streak = 0
                        active_single_no_gate_streak = 0
                        active_single_lock_age = 0
                        frame_buffer.clear()
                        full_frame_buffer.clear()
                else:
                    reacquired = False
                    if (
                        best_incident_center is not None
                        and best_incident_ids
                        and highest_frame_score >= REACQUIRE_SCORE_MIN
                    ):
                        dist = math.hypot(
                            best_incident_center[0] - active_tube_center[0],
                            best_incident_center[1] - active_tube_center[1],
                        )

                        if dist <= max(140.0, active_tube_size * REACQUIRE_DIST_FACTOR):
                            reacquire_alpha = 0.45 if roi_mode == "search" else 0.58
                            active_tube_center = (
                                reacquire_alpha * best_incident_center[0] + (1.0 - reacquire_alpha) * active_tube_center[0],
                                reacquire_alpha * best_incident_center[1] + (1.0 - reacquire_alpha) * active_tube_center[1],
                            )
                            active_target_ids = set(best_incident_ids)
                            current_active_score = max(current_active_score, int(highest_frame_score))
                            current_benign_score = max(0.0, 0.7 * current_benign_score + 0.3 * best_incident_benign)
                            active_no_contact_streak = 0
                            active_low_conf_streak = max(0, active_low_conf_streak - 1)
                            active_single_gate_streak = 0
                            active_single_no_gate_streak = 0
                            active_single_lock_age = 0
                            tube_life = max(tube_life - 1, MAX_TUBE_LIFE // 3)
                            grace_period_counter = MAX_GRACE_PERIOD
                            if best_incident_box is not None:
                                desired_size = phase_scaled_roi_size(roi_size_from_box(best_incident_box), roi_mode)
                                if len(active_target_ids) >= 2:
                                    desired_size = min(desired_size, focused_pair_roi_size(best_incident_box, roi_mode))
                                size_smooth = SEARCH_SIZE_SMOOTH if roi_mode == "search" else CONFIRM_SIZE_SMOOTH
                                active_tube_size = size_smooth * desired_size + (1.0 - size_smooth) * active_tube_size
                            reacquired = True

                    if not reacquired:
                        if grace_period_counter > 0:
                            grace_period_counter -= 1
                            tube_life -= 1
                            current_active_score = max(0, current_active_score - SCORE_DECAY_GRACE)
                            current_benign_score = max(0.0, current_benign_score - 1.0)
                            active_no_contact_streak = min(active_no_contact_streak + 1, NO_CONTACT_RELEASE_FRAMES + 2)
                            active_low_conf_streak = min(active_low_conf_streak + 1, LOW_CONF_RELEASE_FRAMES + 2)
                            active_single_gate_streak = max(0, active_single_gate_streak - 1)
                            active_single_no_gate_streak = min(active_single_no_gate_streak + 1, SINGLE_NO_GATE_RELEASE_FRAMES + 3)
                        else:
                            tube_life = 0
                            active_tube_center = None
                            active_target_ids = set()
                            current_active_score = 0
                            current_benign_score = 0.0
                            active_no_contact_streak = 0
                            active_low_conf_streak = 0
                            active_single_gate_streak = 0
                            active_single_no_gate_streak = 0
                            active_single_lock_age = 0
                            frame_buffer.clear()
                            full_frame_buffer.clear()
            else:
                tube_life = 0
                active_tube_center = None
                active_target_ids = set()
                current_active_score = 0
                current_benign_score = 0.0
                active_no_contact_streak = 0
                active_low_conf_streak = 0
                active_single_gate_streak = 0
                active_single_no_gate_streak = 0
                active_single_lock_age = 0
                frame_buffer.clear()
                full_frame_buffer.clear()

            if active_tube_center is not None and tube_life > 0:
                roi_size = int(clamp(active_tube_size, MIN_TUBE_SIZE, MAX_TUBE_SIZE))
                half = roi_size // 2

                cx = int(active_tube_center[0])
                cy = int(active_tube_center[1])

                x1_u = max(0, cx - half)
                y1_u = max(0, cy - half)
                x2_u = min(w_frame, cx + half)
                y2_u = min(h_frame, cy + half)

                crop_img = raw_frame[y1_u:y2_u, x1_u:x2_u]
                if crop_img.size != 0:
                    if frame_idx % 10 == 0:
                        import os
                        debug_dir = os.path.join(os.getcwd(), "debug_roi")
                        os.makedirs(debug_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(debug_dir, f"roi_{frame_idx:04d}.jpg"), crop_img)
                        
                    pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                    frame_buffer.append(transform(pil_img))
                    full_img = Image.fromarray(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB))
                    full_frame_buffer.append(transform(full_img))

                tube_color = (0, 0, 255) if prev_alarm_on else (0, 255, 255)
                cv2.rectangle(vis_frame, (x1_u, y1_u), (x2_u, y2_u), tube_color, 2)
                cv2.putText(
                    vis_frame,
                    "Target Lock-on",
                    (x1_u, max(20, y1_u - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    tube_color,
                    2,
                )

                if (
                    len(frame_buffer) == args.num_segments
                    and len(full_frame_buffer) == args.num_segments
                    and frame_idx % INFER_STRIDE == 0
                ):
                    infer_frames_count += 1
                    max_single_gate_streak_at_infer = max(max_single_gate_streak_at_infer, int(active_single_gate_streak))
                    max_active_score_at_infer = max(max_active_score_at_infer, int(current_active_score))
                    gate_streak_ok = active_single_gate_streak >= FULLFRAME_RESCUE_MIN_SINGLE_GATE_STREAK
                    active_score_ok = current_active_score >= FULLFRAME_RESCUE_MIN_ACTIVE_SCORE
                    benign_ok = current_benign_score <= FULLFRAME_RESCUE_MAX_BENIGN
                    if gate_streak_ok:
                        rescue_gate_streak_ok_frames += 1
                    if active_score_ok:
                        rescue_active_score_ok_frames += 1
                    if benign_ok:
                        rescue_benign_ok_frames += 1
                    video_tensor = torch.stack(list(frame_buffer), dim=0).unsqueeze(0).to(device)
                    use_fullframe_rescue = (
                        fullframe_rescue_enabled
                        and gate_streak_ok
                        and active_score_ok
                        and lock_frames >= FULLFRAME_RESCUE_MIN_LOCK_FRAMES
                        and benign_ok
                    )
                    if use_fullframe_rescue:
                        rescue_all_ok_frames += 1
                        fullframe_rescue_eval_frames += 1
                    raw_lock_prob = 0.0
                    raw_fullframe_prob = 0.0
                    with torch.no_grad():
                        logits = tsm_model(video_tensor)
                        probs = torch.softmax(logits, dim=1)
                        raw_lock_prob = float(probs[0, 1].item())

                        if use_fullframe_rescue:
                            full_tensor = torch.stack(list(full_frame_buffer), dim=0).unsqueeze(0).to(device)
                            full_logits = tsm_model(full_tensor)
                            full_probs = torch.softmax(full_logits, dim=1)
                            raw_fullframe_prob = float(full_probs[0, 1].item())

                    raw_fight_prob = raw_lock_prob
                    if use_fullframe_rescue:
                        max_fullframe_prob_observed = max(max_fullframe_prob_observed, raw_fullframe_prob)
                        rescued_prob = max(raw_lock_prob, raw_fullframe_prob * FULLFRAME_RESCUE_BLEND)
                        if rescued_prob > raw_lock_prob + 1e-6:
                            raw_fight_prob = rescued_prob
                            fullframe_rescue_frames += 1

                    tsm_prob_history.append(raw_fight_prob)
                    sorted_probs = sorted(tsm_prob_history)
                    median_prob = sorted_probs[len(sorted_probs) // 2]
                    stable_raw_prob = 0.70 * raw_fight_prob + 0.30 * median_prob

                    kinematic_prob = clamp01((current_active_score - 300.0) / 620.0)
                    scene_prob = clamp01((current_active_score - 330.0) / 520.0)
                    evidence_bonus = clamp01((current_active_score - 460.0) / 520.0) * 0.08
                    if len(active_target_ids) >= 2:
                        evidence_bonus += 0.03

                    if active_tube_size > 470 and current_active_score > 430:
                        evidence_bonus += 0.02

                    benign_penalty = clamp01(current_benign_score / 380.0) * dynamic_benign_scale

                    fused_prob = clamp01(
                        0.78 * stable_raw_prob
                        + 0.14 * kinematic_prob
                        + 0.08 * scene_prob
                        + evidence_bonus
                        - benign_penalty
                    )
                    smooth_prob = 0.55 * smooth_prob + 0.45 * fused_prob

                    single_target_lock = len(active_target_ids) <= 1
                    if single_target_lock:
                        low_conf_threshold_now = LOW_CONF_RELEASE_THRESHOLD
                        low_conf_release_frames_now = LOW_CONF_RELEASE_FRAMES
                    else:
                        low_conf_threshold_now = max(0.14, LOW_CONF_RELEASE_THRESHOLD - 0.08)
                        low_conf_release_frames_now = LOW_CONF_RELEASE_FRAMES + 4

                    if smooth_prob < low_conf_threshold_now:
                        active_low_conf_streak = min(active_low_conf_streak + 1, low_conf_release_frames_now + 3)
                    else:
                        active_low_conf_streak = max(0, active_low_conf_streak - 1)

                    on_threshold_now = dynamic_fight_on_base - min(
                        ADAPTIVE_ON_MAX_REDUCTION,
                        max(0.0, current_active_score - 500.0) / 1800.0,
                    )
                    on_threshold_now += min(0.07, current_benign_score / 1500.0)
                    on_threshold_now = max(FIGHT_OFF_THRESHOLD + 0.09, on_threshold_now)
                    active_margin_now = float(current_active_score - current_benign_score)
                    max_active_margin_observed = max(max_active_margin_observed, active_margin_now)

                    single_alert_supported = (
                        len(active_target_ids) >= 2
                        or active_single_gate_streak >= SINGLE_ALERT_GATE_FRAMES
                    )

                    score_gate = (
                        current_active_score >= 360
                        and (current_active_score - current_benign_score) >= 130
                        and active_no_contact_streak <= 2
                        and single_alert_supported
                    )

                    high_conf_fight = (
                        stable_raw_prob >= STRONG_FIGHT_RAW_THRESHOLD
                        and fused_prob >= 0.34
                        and current_active_score >= STRONG_FIGHT_SCORE_THRESHOLD
                        and (current_active_score - current_benign_score) >= 220
                        and current_benign_score < 120.0
                        and active_no_contact_streak <= 1
                        and active_low_conf_streak <= max(2, low_conf_release_frames_now - 2)
                        and single_alert_supported
                    )
                    strong_kinematic_override = (
                        current_active_score >= 500
                        and (current_active_score - current_benign_score) >= 320
                        and current_benign_score <= 90.0
                        and fused_prob >= 0.18
                        and smooth_prob >= 0.21
                        and (stable_raw_prob <= 0.30 or score_gate)
                        and (current_active_score < 900 or score_gate)
                        and active_no_contact_streak <= 4
                        and active_low_conf_streak <= low_conf_release_frames_now
                        and single_alert_supported
                    )
                    single_target_burst_override = (
                        single_target_lock
                        and active_single_gate_streak >= SINGLE_BURST_GATE_FRAMES
                        and current_active_score >= SINGLE_BURST_MIN_SCORE
                        and (current_active_score - current_benign_score) >= SINGLE_BURST_MIN_MARGIN
                        and SINGLE_BURST_RAW_MIN <= stable_raw_prob <= SINGLE_BURST_RAW_MAX
                        and current_benign_score <= 320.0
                        and active_no_contact_streak <= 4
                        and active_low_conf_streak <= (low_conf_release_frames_now + 2)
                    )
                    selective_core_pass = (
                        current_active_score >= SELECTIVE_BURST_MIN_SCORE
                        and (current_active_score - current_benign_score) >= SELECTIVE_BURST_MIN_MARGIN
                        and SELECTIVE_BURST_RAW_MIN <= stable_raw_prob <= SELECTIVE_BURST_RAW_MAX
                        and SELECTIVE_BURST_FUSED_MIN <= fused_prob <= SELECTIVE_BURST_FUSED_MAX
                        and SELECTIVE_BURST_SMOOTH_MIN <= smooth_prob <= SELECTIVE_BURST_SMOOTH_MAX
                        and SELECTIVE_BURST_BENIGN_MIN <= current_benign_score <= SELECTIVE_BURST_BENIGN_MAX
                    )
                    selective_sg_pass = max(active_single_gate_streak, max_single_gate_streak_observed) >= SELECTIVE_BURST_GATE_FRAMES
                    selective_single_lock_ratio_now = float(single_lock_frames / max(1, lock_frames))
                    selective_lock_shape_pass = (
                        lock_frames >= SELECTIVE_BURST_MIN_LOCK_FRAMES
                        and selective_single_lock_ratio_now >= SELECTIVE_BURST_MIN_SINGLE_LOCK_RATIO
                    )
                    selective_context_pass = selective_sg_pass or selective_lock_shape_pass
                    selective_target_pass = len(active_target_ids) <= SELECTIVE_BURST_MAX_TARGETS
                    selective_contact_pass = active_no_contact_streak <= SELECTIVE_BURST_MAX_NO_CONTACT
                    selective_low_conf_pass = active_low_conf_streak <= (low_conf_release_frames_now + SELECTIVE_BURST_EXTRA_LOW_CONF)
                    if selective_core_pass:
                        selective_core_frames += 1
                        if selective_core_first_lock_frames < 0:
                            selective_core_first_lock_frames = int(lock_frames)
                            selective_core_first_single_lock_ratio = float(selective_single_lock_ratio_now)
                        selective_core_max_single_lock_ratio = max(
                            selective_core_max_single_lock_ratio,
                            float(selective_single_lock_ratio_now),
                        )
                        if selective_sg_pass:
                            selective_core_sg_frames += 1
                        if selective_lock_shape_pass:
                            selective_core_lockshape_frames += 1
                        if selective_context_pass:
                            selective_core_context_frames += 1
                            if selective_target_pass:
                                selective_core_sg_target_frames += 1
                                if selective_contact_pass:
                                    selective_core_sg_target_contact_frames += 1

                    single_target_selective_override = (
                        selective_core_pass
                        and selective_context_pass
                        and selective_target_pass
                        and selective_contact_pass
                        and selective_low_conf_pass
                    )
                    if args.decision_mode == "tsm_only":
                        score_gate = False
                        high_conf_fight = False
                        strong_kinematic_override = False
                        single_target_burst_override = False
                        single_target_selective_override = False

                    if high_conf_fight:
                        high_conf_fight_frames += 1
                    if strong_kinematic_override:
                        strong_kinematic_override_frames += 1
                    if single_target_burst_override:
                        single_target_burst_override_frames += 1
                    if single_target_selective_override:
                        single_target_selective_override_frames += 1
                        selective_cooldown = max(selective_cooldown, SELECTIVE_COOLDOWN_FRAMES)
                    if score_gate and smooth_prob >= on_threshold_now:
                        score_gate_frames += 1

                    tsm_only_trigger = False
                    if args.decision_mode == "tsm_only":
                        if args.tsm_only_use_raw_boost:
                            tsm_decision_prob = max(smooth_prob, stable_raw_prob)
                        else:
                            tsm_decision_prob = smooth_prob
                        tsm_only_trigger = tsm_decision_prob >= args.tsm_only_on_threshold
                        if tsm_only_trigger:
                            tsm_only_trigger_frames += 1
                            violence_streak += 1
                            normal_streak = 0
                            roi_confirm_streak += 1
                            roi_release_streak = 0
                        elif tsm_decision_prob <= args.tsm_only_off_threshold:
                            normal_streak += 1
                            violence_streak = max(0, violence_streak - 1)
                            roi_release_streak += 1
                            roi_confirm_streak = max(0, roi_confirm_streak - 1)
                        else:
                            violence_streak = max(0, violence_streak - 1)
                            normal_streak = max(0, normal_streak - 1)
                            roi_confirm_streak = max(0, roi_confirm_streak - 1)
                            roi_release_streak = max(0, roi_release_streak - 1)

                        if roi_mode == "search" and (roi_confirm_streak >= CONFIRM_ENTER_STREAK or tsm_only_trigger):
                            roi_mode = "confirm"
                        if roi_mode == "confirm" and roi_release_streak >= CONFIRM_EXIT_STREAK:
                            roi_mode = "search"

                        if not alarm_on and violence_streak >= MIN_VIOLENCE_STREAK:
                            alarm_on = True
                        if alarm_on and normal_streak >= MIN_NORMAL_STREAK:
                            alarm_on = False
                    else:
                        if (smooth_prob >= on_threshold_now and score_gate) or high_conf_fight or strong_kinematic_override or single_target_burst_override or single_target_selective_override:
                            violence_streak += 2 if (high_conf_fight or strong_kinematic_override or single_target_burst_override or single_target_selective_override) else 1
                            normal_streak = 0
                            roi_confirm_streak += 1
                            roi_release_streak = 0
                        elif smooth_prob <= FIGHT_OFF_THRESHOLD and current_active_score < 270:
                            normal_streak += 1
                            violence_streak = max(0, violence_streak - 1)
                            roi_release_streak += 1
                            roi_confirm_streak = max(0, roi_confirm_streak - 1)
                        else:
                            violence_streak = max(0, violence_streak - 1)
                            normal_streak = max(0, normal_streak - 1)
                            roi_confirm_streak = max(0, roi_confirm_streak - 1)
                            roi_release_streak = max(0, roi_release_streak - 1)

                        if roi_mode == "search" and (roi_confirm_streak >= CONFIRM_ENTER_STREAK or high_conf_fight or single_target_burst_override or single_target_selective_override):
                            roi_mode = "confirm"
                        if roi_mode == "confirm" and roi_release_streak >= CONFIRM_EXIT_STREAK:
                            roi_mode = "search"

                        if not alarm_on and violence_streak >= MIN_VIOLENCE_STREAK:
                            alarm_on = True
                        if alarm_on and (
                            normal_streak >= MIN_NORMAL_STREAK
                            or ((smooth_prob < 0.28 and current_active_score < 170) and selective_cooldown == 0)
                        ):
                            alarm_on = False

                    if (
                        active_low_conf_streak >= low_conf_release_frames_now
                        and not single_target_selective_override
                        and selective_cooldown == 0
                    ):
                        tube_life = 0
                        active_tube_center = None
                        active_target_ids = set()
                        current_active_score = 0
                        current_benign_score = 0.0
                        active_no_contact_streak = 0
                        active_low_conf_streak = 0
                        active_single_gate_streak = 0
                        active_single_no_gate_streak = 0
                        active_single_lock_age = 0
                        alarm_on = False
                        frame_buffer.clear()
                        full_frame_buffer.clear()
            else:
                frame_buffer.clear()
                full_frame_buffer.clear()
                tsm_prob_history.clear()
                smooth_prob *= 0.93
                fused_prob = smooth_prob
                raw_fight_prob = smooth_prob
                on_threshold_now = dynamic_fight_on_base
                current_benign_score = 0.0
                active_no_contact_streak = 0
                active_low_conf_streak = 0
                active_single_gate_streak = 0
                active_single_no_gate_streak = 0
                active_single_lock_age = 0
                roi_mode = "search"
                roi_confirm_streak = 0
                roi_release_streak = 0
                normal_streak += 1
                violence_streak = max(0, violence_streak - 1)
                if normal_streak >= MIN_NORMAL_STREAK:
                    alarm_on = False
        else:
            interaction_tracker.clear()
            proximity_tracker.clear()
            wide_box_tracker.clear()
            motion_history.clear()

            if auto_crowd_enabled:
                crowd_ema = max(0.0, crowd_ema * 0.90)
                crowd_factor = compute_crowd_factor(crowd_ema)
            else:
                crowd_ema = max(0.0, crowd_ema * 0.90)
                crowd_factor = 0.0

            (
                dynamic_min_activation,
                dynamic_pair_margin,
                dynamic_scene_margin,
                dynamic_fight_on_base,
                dynamic_benign_scale,
                dynamic_allow_prox_contact,
            ) = compute_dynamic_tuning(crowd_factor)

            tube_life = 0
            active_tube_center = None
            active_target_ids = set()
            current_active_score = 0
            current_benign_score = 0.0
            active_no_contact_streak = 0
            active_low_conf_streak = 0
            active_single_gate_streak = 0
            active_single_no_gate_streak = 0
            active_single_lock_age = 0
            active_tube_size = float(BASE_TUBE_SIZE)
            frame_buffer.clear()
            full_frame_buffer.clear()
            tsm_prob_history.clear()

            smooth_prob *= 0.90
            fused_prob = smooth_prob
            raw_fight_prob = smooth_prob
            on_threshold_now = dynamic_fight_on_base
            roi_mode = "search"
            roi_confirm_streak = 0
            roi_release_streak = 0
            normal_streak += 1
            violence_streak = max(0, violence_streak - 1)
            if normal_streak >= MIN_NORMAL_STREAK:
                alarm_on = False

        status_text = f"ALERT: VIOLENCE ({smooth_prob * 100:.0f}%)" if alarm_on else f"Normal ({smooth_prob * 100:.0f}%)"
        status_color = (0, 0, 255) if alarm_on else (0, 255, 0)

        cv2.putText(
            vis_frame,
            f"TUBE:{tube_life:03d} SCORE:{current_active_score:03d} BENIGN:{int(current_benign_score):03d} NC:{active_no_contact_streak:02d} LC:{active_low_conf_streak:02d} ROI:{int(active_tube_size):03d} MODE:{roi_mode} CROWD:{crowd_factor:.2f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 180, 255) if tube_life > 0 else (0, 255, 0),
            2,
        )

        cv2.putText(
            vis_frame,
                f"raw={raw_fight_prob:.2f} fused={fused_prob:.2f} smooth={smooth_prob:.2f} on={on_threshold_now:.2f} margin={display_scene_margin:.0f} act={dynamic_min_activation:.0f} det={len(track_data):02d}",
            (10, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            vis_frame,
            status_text,
            (10, h_frame - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            status_color,
            3,
        )

        if tube_life > 0 and active_tube_center is not None:
            lock_frames += 1
            current_lock_duration += 1
            if len(active_target_ids) <= 1:
                single_lock_frames += 1
        else:
            if current_lock_duration > max_lock_duration:
                max_lock_duration = current_lock_duration
            current_lock_duration = 0

        if alarm_on:
            alert_frames += 1
        if alarm_on and not prev_alarm_on:
            alarm_on_events += 1
        prev_alarm_on = alarm_on

        det_count_sum += len(track_data)
        if len(track_data) > 0:
            frames_with_track_data += 1

        if out is not None:
            out.write(vis_frame)
        if args.show:
            cv2.imshow("Violence System", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if selective_cooldown > 0:
            selective_cooldown -= 1

        max_active_score_observed = max(max_active_score_observed, int(current_active_score))
        max_single_gate_streak_observed = max(max_single_gate_streak_observed, int(active_single_gate_streak))
        max_raw_fight_prob_observed = max(max_raw_fight_prob_observed, float(raw_fight_prob))
        max_fused_prob_observed = max(max_fused_prob_observed, float(fused_prob))
        max_smooth_prob_observed = max(max_smooth_prob_observed, float(smooth_prob))

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    if current_lock_duration > max_lock_duration:
        max_lock_duration = current_lock_duration

    stats = {
        "video_path": str(args.video_path),
        "profile": args.profile,
        "decision_mode": args.decision_mode,
        "fullframe_rescue_enabled": bool(fullframe_rescue_enabled),
        "tsm_only_on_threshold": args.tsm_only_on_threshold,
        "tsm_only_off_threshold": args.tsm_only_off_threshold,
        "det_conf": args.det_conf,
        "person_classes": sorted(person_class_ids),
        "total_frames": total_frames,
        "alert_frames": alert_frames,
        "alert_ratio": float(alert_frames / max(1, total_frames)),
        "lock_frames": lock_frames,
        "single_lock_frames": single_lock_frames,
        "single_lock_ratio": float(single_lock_frames / max(1, lock_frames)),
        "max_lock_duration": max_lock_duration,
        "mean_detections_per_frame": float(det_count_sum / max(1, total_frames)),
        "frames_with_track_data": frames_with_track_data,
        "incident_candidate_frames": incident_candidate_frames,
        "single_candidate_frames": single_candidate_frames,
        "single_candidate_reject_frames": single_candidate_reject_frames,
        "candidate_gate_pass_frames": candidate_gate_pass_frames,
        "lock_switch_events": lock_switch_events,
        "alarm_on_events": alarm_on_events,
        "max_active_score_observed": max_active_score_observed,
        "max_single_gate_streak_observed": max_single_gate_streak_observed,
        "max_raw_fight_prob_observed": max_raw_fight_prob_observed,
        "max_fullframe_prob_observed": max_fullframe_prob_observed,
        "max_fused_prob_observed": max_fused_prob_observed,
        "max_smooth_prob_observed": max_smooth_prob_observed,
        "max_active_margin_observed": max_active_margin_observed,
        "high_conf_fight_frames": high_conf_fight_frames,
        "strong_kinematic_override_frames": strong_kinematic_override_frames,
        "single_target_burst_override_frames": single_target_burst_override_frames,
        "single_target_selective_override_frames": single_target_selective_override_frames,
        "selective_core_frames": selective_core_frames,
        "selective_core_sg_frames": selective_core_sg_frames,
        "selective_core_lockshape_frames": selective_core_lockshape_frames,
        "selective_core_context_frames": selective_core_context_frames,
        "selective_core_sg_target_frames": selective_core_sg_target_frames,
        "selective_core_sg_target_contact_frames": selective_core_sg_target_contact_frames,
        "selective_core_first_lock_frames": selective_core_first_lock_frames,
        "selective_core_first_single_lock_ratio": selective_core_first_single_lock_ratio,
        "selective_core_max_single_lock_ratio": selective_core_max_single_lock_ratio,
        "score_gate_frames": score_gate_frames,
        "tsm_only_trigger_frames": tsm_only_trigger_frames,
        "fullframe_rescue_frames": fullframe_rescue_frames,
        "fullframe_rescue_eval_frames": fullframe_rescue_eval_frames,
        "max_single_gate_streak_at_infer": max_single_gate_streak_at_infer,
        "max_active_score_at_infer": max_active_score_at_infer,
        "infer_frames_count": infer_frames_count,
        "rescue_gate_streak_ok_frames": rescue_gate_streak_ok_frames,
        "rescue_active_score_ok_frames": rescue_active_score_ok_frames,
        "rescue_benign_ok_frames": rescue_benign_ok_frames,
        "rescue_all_ok_frames": rescue_all_ok_frames,
    }

    if args.stats_json is not None:
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.stats_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    print("✅ Done.")
    if out is not None:
        print(f"Output video: {args.output_path}")
    print(
        f"Stats: frames={total_frames} alert={alert_frames} lock={lock_frames} "
        f"single_lock={single_lock_frames} single_ratio={stats['single_lock_ratio']:.3f}"
    )


if __name__ == "__main__":
    main()

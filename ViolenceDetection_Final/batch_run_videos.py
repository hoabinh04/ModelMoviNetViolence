from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_INPUT_DIR = Path(r"F:\NCKH\codeDot1\datasetthamkhao\SCVD_converted\Test\Violence")
DEFAULT_OUTPUT_DIR = Path("outputs/scvd_violence")
DEFAULT_STATS_DIR = Path("outputs/scvd_violence_stats")
DEFAULT_PIPELINE = Path("src/buoc6_main_pipeline_stable.py")
DEFAULT_TSM_WEIGHTS = Path("weights/best_tsm_topdown.pth")
DEFAULT_YOLO_WEIGHTS = Path("src/yolo26n.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run violence pipeline and collect stats")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--pipeline-script", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument("--tsm-weights", type=Path, default=DEFAULT_TSM_WEIGHTS)
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--num-segments", type=int, default=12)
    parser.add_argument("--tsm-variant", type=str, default="large", choices=["small", "large"])
    parser.add_argument("--profile", type=str, default="balanced")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--det-conf", type=float, default=0.18)
    parser.add_argument("--detector-mode", type=str, default="track", choices=["predict", "track"])
    parser.add_argument("--decision-mode", type=str, default="tsm_only", choices=["hybrid", "tsm_only"])
    parser.add_argument("--tsm-only-on-threshold", type=float, default=0.50)
    parser.add_argument("--tsm-only-off-threshold", type=float, default=0.36)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--alert-frames-thresh", type=int, default=8)
    parser.add_argument("--exts", type=str, default=".mp4,.avi,.mkv")
    return parser.parse_args()


def collect_videos(folder: Path, exts: List[str]) -> List[Path]:
    exts_lower = {e.lower() for e in exts}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts_lower])


def run_video(video_path: Path, args: argparse.Namespace, root_dir: Path) -> Dict:
    output_path = args.output_dir / f"{video_path.stem}_out.mp4"
    stats_path = args.stats_dir / f"{video_path.stem}.json"

    cmd = [
        sys.executable,
        str(args.pipeline_script),
        "--video-path",
        str(video_path),
        "--output-path",
        str(output_path),
        "--stats-json",
        str(stats_path),
        "--num-segments",
        str(args.num_segments),
        "--tsm-variant",
        args.tsm_variant,
        "--profile",
        args.profile,
        "--imgsz",
        str(args.imgsz),
        "--det-conf",
        str(args.det_conf),
        "--detector-mode",
        args.detector_mode,
        "--decision-mode",
        args.decision_mode,
        "--tsm-only-on-threshold",
        str(args.tsm_only_on_threshold),
        "--tsm-only-off-threshold",
        str(args.tsm_only_off_threshold),
    ]

    if args.max_frames and args.max_frames > 0:
        cmd.extend(["--max-frames", str(args.max_frames)])
    if args.tsm_weights is not None:
        cmd.extend(["--tsm-weights", str(args.tsm_weights)])
    if args.yolo_weights is not None:
        cmd.extend(["--yolo-weights", str(args.yolo_weights)])

    proc = subprocess.run(
        cmd,
        cwd=str(root_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    result: Dict = {
        "video": str(video_path),
        "output": str(output_path),
        "returncode": proc.returncode,
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-8:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-8:]),
        "stats": None,
    }

    if proc.returncode == 0 and stats_path.exists():
        result["stats"] = json.loads(stats_path.read_text(encoding="utf-8"))

    return result


def summarize(results: List[Dict], alert_frames_thresh: int) -> Dict:
    ok_rows = [r for r in results if r.get("stats")]
    total = len(ok_rows)

    positives = 0
    alert_ratios: List[float] = []
    single_lock_ratios: List[float] = []

    for row in ok_rows:
        stats = row["stats"]
        if int(stats.get("alert_frames", 0)) >= alert_frames_thresh:
            positives += 1
        alert_ratios.append(float(stats.get("alert_ratio", 0.0)))
        single_lock_ratios.append(float(stats.get("single_lock_ratio", 0.0)))

    failed = [r for r in results if not r.get("stats")]

    return {
        "total_ok": total,
        "total_failed": len(failed),
        "positive_videos": positives,
        "positive_ratio": float(positives / max(1, total)),
        "mean_alert_ratio": float(sum(alert_ratios) / max(1, len(alert_ratios))),
        "mean_single_lock_ratio": float(sum(single_lock_ratios) / max(1, len(single_lock_ratios))),
        "failed": failed,
    }


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent

    if not args.pipeline_script.is_absolute():
        args.pipeline_script = (root_dir / args.pipeline_script).resolve()
    if not args.tsm_weights.is_absolute():
        args.tsm_weights = (root_dir / args.tsm_weights).resolve()
    if not args.yolo_weights.is_absolute():
        args.yolo_weights = (root_dir / args.yolo_weights).resolve()
    if not args.output_dir.is_absolute():
        args.output_dir = (root_dir / args.output_dir).resolve()
    if not args.stats_dir.is_absolute():
        args.stats_dir = (root_dir / args.stats_dir).resolve()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.stats_dir.mkdir(parents=True, exist_ok=True)

    exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    videos = collect_videos(args.input_dir, exts)
    if not videos:
        raise FileNotFoundError(f"No videos found in {args.input_dir}")

    print(f"[Batch] videos: {len(videos)}")

    results: List[Dict] = []
    for idx, video in enumerate(videos, start=1):
        print(f"[{idx}/{len(videos)}] {video.name}")
        results.append(run_video(video, args, root_dir))

    report = {
        "config": {
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
            "stats_dir": str(args.stats_dir),
            "pipeline_script": str(args.pipeline_script),
            "tsm_weights": str(args.tsm_weights),
            "yolo_weights": str(args.yolo_weights),
            "num_segments": args.num_segments,
            "tsm_variant": args.tsm_variant,
            "profile": args.profile,
            "imgsz": args.imgsz,
            "det_conf": args.det_conf,
            "detector_mode": args.detector_mode,
            "decision_mode": args.decision_mode,
            "tsm_only_on_threshold": args.tsm_only_on_threshold,
            "tsm_only_off_threshold": args.tsm_only_off_threshold,
            "max_frames": args.max_frames,
            "alert_frames_thresh": args.alert_frames_thresh,
        },
        "summary": summarize(results, args.alert_frames_thresh),
        "results": results,
    }

    report_path = args.stats_dir / "batch_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== BATCH SUMMARY ===")
    print(f"OK: {report['summary']['total_ok']} | Failed: {report['summary']['total_failed']}")
    print(
        "Positive ratio: "
        f"{report['summary']['positive_ratio']:.3f} "
        f"({report['summary']['positive_videos']}/{report['summary']['total_ok']})"
    )
    print(f"Mean alert ratio: {report['summary']['mean_alert_ratio']:.3f}")
    print(f"Mean single-lock ratio: {report['summary']['mean_single_lock_ratio']:.3f}")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()

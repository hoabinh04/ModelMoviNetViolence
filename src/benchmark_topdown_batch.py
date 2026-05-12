from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DEFAULT_VIOLENCE_DIR = Path(r"C:\Users\Lenovo\Desktop\codeNCKH\codeDot2\Data_Violence_Detection\Violence")
DEFAULT_NONVIOLENCE_DIR = Path(r"C:\Users\Lenovo\Desktop\codeNCKH\codeDot2\Data_Violence_Detection\NonViolence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch benchmark for top-down CCTV violence pipeline")
    parser.add_argument("--violence-dir", type=Path, default=DEFAULT_VIOLENCE_DIR)
    parser.add_argument("--nonviolence-dir", type=Path, default=DEFAULT_NONVIOLENCE_DIR)
    parser.add_argument("--violence-sample", type=int, default=24)
    parser.add_argument("--nonviolence-sample", type=int, default=24)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--num-segments", type=int, default=None)
    parser.add_argument("--profile", type=str, default="balanced")
    parser.add_argument("--decision-mode", type=str, default="hybrid", choices=["hybrid", "tsm_only"])
    parser.add_argument("--tsm-only-on-threshold", type=float, default=0.56)
    parser.add_argument("--tsm-only-off-threshold", type=float, default=0.40)
    parser.add_argument("--tsm-only-use-raw-boost", action="store_true")
    parser.add_argument("--det-conf", type=float, default=0.18)
    parser.add_argument("--person-classes", type=str, default="0,18,72")
    parser.add_argument("--detector-mode", type=str, default="predict", choices=["predict", "track"])
    parser.add_argument("--alert-frames-thresh", type=int, default=8)
    parser.add_argument("--stats-dir", type=Path, default=Path("tmp_batch_stats"))
    parser.add_argument("--output-json", type=Path, default=Path("benchmark_topdown_report.json"))
    parser.add_argument("--pipeline-script", type=Path, default=Path("buoc6_main_pipeline.py"))
    parser.add_argument("--tsm-weights", type=Path, default=None)
    parser.add_argument("--yolo-weights", type=Path, default=None)
    return parser.parse_args()


def pick_evenly(paths: List[Path], count: int) -> List[Path]:
    if count <= 0 or not paths:
        return []
    if count >= len(paths):
        return paths
    if count == 1:
        return [paths[len(paths) // 2]]

    selected: List[Path] = []
    for i in range(count):
        idx = round(i * (len(paths) - 1) / (count - 1))
        selected.append(paths[idx])

    unique: List[Path] = []
    seen = set()
    for p in selected:
        if p not in seen:
            unique.append(p)
            seen.add(p)

    if len(unique) < count:
        for p in paths:
            if p not in seen:
                unique.append(p)
                seen.add(p)
            if len(unique) >= count:
                break

    return unique[:count]


def collect_videos(folder: Path, prefix: str) -> List[Path]:
    return sorted(p for p in folder.glob(f"{prefix}*.mp4") if p.is_file())


def run_single_video(
    video_path: Path,
    label: str,
    args: argparse.Namespace,
    script_dir: Path,
) -> Dict:
    stats_path = args.stats_dir / f"{label}_{video_path.stem}.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(args.pipeline_script),
        "--video-path",
        str(video_path),
        "--profile",
        args.profile,
        "--decision-mode",
        args.decision_mode,
        "--det-conf",
        str(args.det_conf),
        "--person-classes",
        args.person_classes,
        "--detector-mode",
        args.detector_mode,
        "--max-frames",
        str(args.max_frames),
        "--no-save",
        "--stats-json",
        str(stats_path),
    ]

    if args.num_segments is not None:
        cmd.extend(["--num-segments", str(args.num_segments)])
    if args.decision_mode == "tsm_only":
        cmd.extend(
            [
                "--tsm-only-on-threshold",
                str(args.tsm_only_on_threshold),
                "--tsm-only-off-threshold",
                str(args.tsm_only_off_threshold),
            ]
        )
        if args.tsm_only_use_raw_boost:
            cmd.append("--tsm-only-use-raw-boost")
    if args.tsm_weights is not None:
        cmd.extend(["--tsm-weights", str(args.tsm_weights)])
    if args.yolo_weights is not None:
        cmd.extend(["--yolo-weights", str(args.yolo_weights)])

    proc = subprocess.run(
        cmd,
        cwd=str(script_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    result: Dict = {
        "video": str(video_path),
        "label": label,
        "returncode": proc.returncode,
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-8:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-8:]),
    }

    if proc.returncode == 0 and stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            result["stats"] = json.load(f)
    else:
        result["stats"] = None

    return result


def summarize(results: List[Dict], alert_frames_thresh: int) -> Dict:
    ok_rows = [r for r in results if r.get("stats")]

    total = len(ok_rows)
    positives = 0
    single_lock_ratios: List[float] = []
    alert_ratios: List[float] = []

    for row in ok_rows:
        stats = row["stats"]
        if int(stats.get("alert_frames", 0)) >= alert_frames_thresh:
            positives += 1
        single_lock_ratios.append(float(stats.get("single_lock_ratio", 0.0)))
        alert_ratios.append(float(stats.get("alert_ratio", 0.0)))

    failed = [r for r in results if not r.get("stats")]

    return {
        "total_ok": total,
        "total_failed": len(failed),
        "positive_videos": positives,
        "positive_ratio": float(positives / max(1, total)),
        "mean_single_lock_ratio": float(sum(single_lock_ratios) / max(1, len(single_lock_ratios))),
        "mean_alert_ratio": float(sum(alert_ratios) / max(1, len(alert_ratios))),
        "failed": failed,
    }


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent

    if not args.stats_dir.is_absolute():
        args.stats_dir = (workspace_root / args.stats_dir).resolve()
    if not args.output_json.is_absolute():
        args.output_json = (workspace_root / args.output_json).resolve()
    if not args.pipeline_script.is_absolute():
        args.pipeline_script = (script_dir / args.pipeline_script).resolve()
    if args.tsm_weights is not None and not args.tsm_weights.is_absolute():
        args.tsm_weights = (workspace_root / args.tsm_weights).resolve()
    if args.yolo_weights is not None and not args.yolo_weights.is_absolute():
        args.yolo_weights = (workspace_root / args.yolo_weights).resolve()

    violence_all = collect_videos(args.violence_dir, "f")
    nonviolence_all = collect_videos(args.nonviolence_dir, "nf")

    if not violence_all:
        raise FileNotFoundError(f"No violence videos found in {args.violence_dir}")
    if not nonviolence_all:
        raise FileNotFoundError(f"No nonviolence videos found in {args.nonviolence_dir}")

    violence_set = pick_evenly(violence_all, args.violence_sample)
    nonviolence_set = pick_evenly(nonviolence_all, args.nonviolence_sample)

    print(f"[Batch] violence sample: {len(violence_set)} / {len(violence_all)}")
    print(f"[Batch] nonviolence sample: {len(nonviolence_set)} / {len(nonviolence_all)}")

    violence_results: List[Dict] = []
    nonviolence_results: List[Dict] = []

    for idx, video in enumerate(violence_set, start=1):
        print(f"[Violence {idx}/{len(violence_set)}] {video.name}")
        violence_results.append(run_single_video(video, "violence", args, script_dir))

    for idx, video in enumerate(nonviolence_set, start=1):
        print(f"[NonViolence {idx}/{len(nonviolence_set)}] {video.name}")
        nonviolence_results.append(run_single_video(video, "nonviolence", args, script_dir))

    violence_summary = summarize(violence_results, args.alert_frames_thresh)
    nonviolence_summary = summarize(nonviolence_results, args.alert_frames_thresh)

    report = {
        "config": {
            "violence_sample": args.violence_sample,
            "nonviolence_sample": args.nonviolence_sample,
            "max_frames": args.max_frames,
            "num_segments": args.num_segments,
            "profile": args.profile,
            "decision_mode": args.decision_mode,
            "tsm_only_on_threshold": args.tsm_only_on_threshold,
            "tsm_only_off_threshold": args.tsm_only_off_threshold,
            "tsm_only_use_raw_boost": args.tsm_only_use_raw_boost,
            "det_conf": args.det_conf,
            "person_classes": args.person_classes,
            "detector_mode": args.detector_mode,
            "alert_frames_thresh": args.alert_frames_thresh,
            "tsm_weights": str(args.tsm_weights) if args.tsm_weights else None,
            "yolo_weights": str(args.yolo_weights) if args.yolo_weights else None,
        },
        "violence_summary": violence_summary,
        "nonviolence_summary": nonviolence_summary,
        "violence_results": violence_results,
        "nonviolence_results": nonviolence_results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== BENCHMARK SUMMARY ===")
    print(f"Violence recall proxy: {violence_summary['positive_ratio']:.3f} ({violence_summary['positive_videos']}/{violence_summary['total_ok']})")
    print(f"NonViolence false positive proxy: {nonviolence_summary['positive_ratio']:.3f} ({nonviolence_summary['positive_videos']}/{nonviolence_summary['total_ok']})")
    print(f"Mean single-lock ratio (Violence): {violence_summary['mean_single_lock_ratio']:.3f}")
    print(f"Mean single-lock ratio (NonViolence): {nonviolence_summary['mean_single_lock_ratio']:.3f}")
    print(f"Report saved: {args.output_json}")


if __name__ == "__main__":
    main()

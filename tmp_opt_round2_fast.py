import os
import json
import subprocess

def run_benchmark(v_sample, nv_sample, alert_thresh):
    out_file = f"tmp_batch_stats/rush_tune_round2_fast/temp_{v_sample}_{nv_sample}_at{alert_thresh}.json"
    cmd = [
        "rtk", "python", "src/benchmark_topdown_batch.py",
        "--violence-sample", str(v_sample),
        "--nonviolence-sample", str(nv_sample),
        "--max-frames", "240",
        "--detector-mode", "predict",
        "--decision-mode", "tsm_only",
        "--alert-frames-thresh", str(alert_thresh),
        "--tsm-only-on-threshold", "0.56",
        "--tsm-only-off-threshold", "0.40",
        "--tsm-only-use-raw-boost",
        "--profile", "balanced",
        "--output-json", out_file
    ]
    subprocess.run(cmd, shell=True)
    
    if not os.path.exists(out_file):
        return None
        
    with open(out_file, "r") as f:
        data = json.load(f)
    
    v_results = data.get("violence_results", [])
    nv_results = data.get("nonviolence_results", [])
    
    tp = sum(1 for x in v_results if x["stats"]["alert_frames"] > 0)
    p = len(v_results)
    fp = sum(1 for x in nv_results if x["stats"]["alert_frames"] > 0)
    n = len(nv_results)
    tn = n - fp
    fn = p - tp
    
    tpr = tp / f_p if (f_p := float(p)) > 0 else 0
    fpr = fp / f_n if (f_n := float(n)) > 0 else 0
    tnr = 1 - fpr
    
    acc = (tp + tn) / (p + n) if (p + n) > 0 else 0
    bal_acc = (tpr + tnr) / 2
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    score = bal_acc + 0.25 * tpr - 0.15 * fpr
    
    return {
        "alert_thresh": alert_thresh,
        "tp": tp, "p": p, "fp": fp, "n": n,
        "tpr": tpr, "fpr": fpr, "acc": acc, "bal_acc": bal_acc, "f1": f1,
        "score": score
    }

print("Starting Round 2 Fast Optimization (Corrected Args)...")
configs = [8, 6, 4]
results_20 = []
for at in configs:
    print(f"Testing alert_frames_thresh = {at}...")
    res = run_benchmark(20, 20, at)
    if res:
        results_20.append(res)

if results_20:
    results_20.sort(key=lambda x: x["score"], reverse=True)
    with open("tmp_batch_stats/rush_tune_round2_fast/rank_20_20.json", "w") as f:
        json.dump(results_20, f, indent=2)

    best = results_20[0]
    with open("tmp_batch_stats/rush_tune_round2_fast/best_20_20.json", "w") as f:
        json.dump(best, f, indent=2)

    print(f"Best Config: AL={best['alert_thresh']}, Score={best['score']:.4f}. Confirming on 30+40...")
    confirm_res = run_benchmark(30, 40, best["alert_thresh"])

    if confirm_res:
        with open("tmp_batch_stats/rush_tune_round2_fast/confirm_best_30_40.json", "w") as f:
            json.dump(confirm_res, f, indent=2)
        print("Confirmation complete.")
else:
    print("No results generated.")

import json
import subprocess
import sys
import os
from pathlib import Path

out_dir = Path('tmp_batch_stats/rush_tune_round4')
out_dir.mkdir(parents=True, exist_ok=True)

# Important: Use the actual python path to avoid subprocess issues if possible, 
# but "rtk python" was requested. We will use it with shell=True.

configs = [
    {'name':'tsm_bal_raw', 'mode':'tsm_only', 'profile':'balanced', 'on':0.56, 'off':0.40, 'raw':True},
    {'name':'tsm_bal_no_raw', 'mode':'tsm_only', 'profile':'balanced', 'on':0.56, 'off':0.40, 'raw':False},
    {'name':'tsm_hr_raw', 'mode':'tsm_only', 'profile':'high_risk', 'on':0.56, 'off':0.40, 'raw':True},
    {'name':'tsm_hr_loose_raw', 'mode':'tsm_only', 'profile':'high_risk', 'on':0.50, 'off':0.34, 'raw':True},
    {'name':'tsm_sp_raw', 'mode':'tsm_only', 'profile':'school_park', 'on':0.56, 'off':0.40, 'raw':True},
    {'name':'hyb_bal', 'mode':'hybrid', 'profile':'balanced'},
    {'name':'hyb_hr', 'mode':'hybrid', 'profile':'high_risk'},
]

def run_cfg(cfg, vs, nvs, suffix):
    out_json = out_dir / f"{cfg['name']}_{suffix}.json"
    # Ensure any old file is gone so we don't read stale data if it fails
    if out_json.exists():
        out_json.unlink()
        
    cmd = [
        'rtk', 'python', 'src/benchmark_topdown_batch.py',
        '--violence-sample', str(vs),
        '--nonviolence-sample', str(nvs),
        '--max-frames', '240',
        '--detector-mode', 'predict',
        '--num-segments', '12',
        '--profile', cfg['profile'],
        '--decision-mode', cfg['mode'],
        '--alert-frames-thresh', '8',
        '--tsm-weights', 'weights/best_tsm_topdown_hn1.pth',
        '--output-json', str(out_json),
    ]
    if cfg['mode'] == 'tsm_only':
        cmd += ['--tsm-only-on-threshold', str(cfg['on']), '--tsm-only-off-threshold', str(cfg['off'])]
        if cfg.get('raw', False):
            cmd.append('--tsm-only-use-raw-boost')

    print(f"\n>>> RUNNING: {cfg['name']} ({vs}v, {nvs}nv)")
    # Using a string command for shell=True to be safer with spaces/args
    cmd_str = " ".join(cmd)
    result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR running {cfg['name']}: {result.stderr}")
        raise Exception(f"Exit code {result.returncode}")

    if not out_json.exists():
         raise Exception(f"Output file {out_json} was not created.")

    with open(out_json, 'r', encoding='utf-8') as f:
        d = json.load(f)
        
    v = d.get('violence_summary', {})
    nv = d.get('nonviolence_summary', {})
    tp = int(v.get('positive_videos', 0)); p = int(v.get('total_ok', 0))
    fp = int(nv.get('positive_videos', 0)); n = int(nv.get('total_ok', 0))
    tpr = tp / p if p else 0.0
    fpr = fp / n if n else 0.0
    tnr = 1.0 - fpr
    acc = (tp + (n-fp)) / (p + n) if (p + n) else 0.0
    bal_acc = (tpr + tnr) / 2.0
    score = 0.70 * tpr + 0.30 * tnr
    
    row = {**cfg, 'tp':tp, 'p':p, 'fp':fp, 'n':n, 'tpr':tpr, 'fpr':fpr, 'tnr':tnr, 'acc':acc, 'bal_acc':bal_acc, 'score':score, 'json':str(out_json)}
    print(f"RESULT: {cfg['name']} -> TPR={tpr:.3f}, FPR={fpr:.3f}, Score={score:.3f}")
    return row

# Stage-1: quick compare on 20+20 (using small samples first as requested by logic, though prompt said 20/20)
# Actually, the user's provided code uses vs=20, nvs=20.
stage1 = []
print("=== STAGE 1: 20 vs 20 ===")
for cfg in configs:
    try:
        stage1.append(run_cfg(cfg, 20, 20, '20_20'))
    except Exception as e:
        print(f"Config {cfg['name']} failed: {e}")
        stage1.append({**cfg, 'error': str(e)})

stage1_ok = [r for r in stage1 if 'error' not in r]
stage1_sorted = sorted(stage1_ok, key=lambda r: (r['score'], r['bal_acc'], r['tpr'], -r['fpr']), reverse=True)
Path(out_dir / 'stage1_ranked_20_20.json').write_text(json.dumps(stage1_sorted, indent=2), encoding='utf-8')

# Stage-2: confirm top2 on 30+40
print("\n=== STAGE 2: 30 vs 40 ===")
confirm = []
for c in stage1_sorted[:2]:
    try:
        confirm.append(run_cfg(c, 30, 40, '30_40'))
    except Exception as e:
        print(f"Confirm {c['name']} failed: {e}")

if confirm:
    confirm_sorted = sorted(confirm, key=lambda r: (r['score'], r['bal_acc'], r['tpr'], -r['fpr']), reverse=True)
    Path(out_dir / 'confirm_ranked_30_40.json').write_text(json.dumps(confirm_sorted, indent=2), encoding='utf-8')
    Path(out_dir / 'best_config_30_40.json').write_text(json.dumps(confirm_sorted[0], indent=2), encoding='utf-8')

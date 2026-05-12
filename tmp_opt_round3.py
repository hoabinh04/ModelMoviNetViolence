import json
import subprocess
import sys
from pathlib import Path

out_dir = Path('tmp_batch_stats/rush_tune_round3')
out_dir.mkdir(parents=True, exist_ok=True)

configs = [
    {'name':'hyb_bal', 'mode':'hybrid', 'profile':'balanced'},
    {'name':'hyb_agg', 'mode':'hybrid', 'profile':'aggressive'},
    {'name':'tsm_bal_056_040_raw', 'mode':'tsm_only', 'profile':'balanced', 'on':0.56, 'off':0.40, 'raw':True},
    {'name':'tsm_agg_056_040_raw', 'mode':'tsm_only', 'profile':'aggressive', 'on':0.56, 'off':0.40, 'raw':True},
    {'name':'tsm_bal_050_034_raw', 'mode':'tsm_only', 'profile':'balanced', 'on':0.50, 'off':0.34, 'raw':True},
    {'name':'tsm_agg_050_034_raw', 'mode':'tsm_only', 'profile':'aggressive', 'on':0.50, 'off':0.34, 'raw':True},
    {'name':'tsm_bal_045_030_raw', 'mode':'tsm_only', 'profile':'balanced', 'on':0.45, 'off':0.30, 'raw':True},
    {'name':'tsm_agg_045_030_raw', 'mode':'tsm_only', 'profile':'aggressive', 'on':0.45, 'off':0.30, 'raw':True},
]

def metrics(report):
    v = report['violence_summary']; nv = report['nonviolence_summary']
    tp = int(v['positive_videos']); p = int(v['total_ok'])
    fp = int(nv['positive_videos']); n = int(nv['total_ok'])
    fn = p - tp; tn = n - fp
    tpr = tp / p if p else 0.0
    fpr = fp / n if n else 0.0
    tnr = tn / n if n else 0.0
    acc = (tp + tn) / (p + n) if (p + n) else 0.0
    bal = (tpr + tnr) / 2.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    score = 0.75 * tpr + 0.25 * tnr
    return {
        'tp': tp, 'p': p, 'fp': fp, 'n': n,
        'tpr': tpr, 'fpr': fpr, 'tnr': tnr,
        'acc': acc, 'bal_acc': bal, 'f1': f1, 'score': score
    }

def run_cfg(cfg, vs, nvs, suffix):
    out_json = out_dir / f"{cfg['name']}_{suffix}.json"
    cmd = [
        sys.executable,
        'src/benchmark_topdown_batch.py',
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
        cmd += [
            '--tsm-only-on-threshold', str(cfg['on']),
            '--tsm-only-off-threshold', str(cfg['off'])
        ]
        if cfg.get('raw', False):
            cmd.append('--tsm-only-use-raw-boost')

    print('RUN', cfg['name'], suffix)
    # Using 'rtk' is typically a prefix or an alias in the user's environment.
    # We will assume calling subprocess with sys.executable is correct or use 'rtk' if it's an executable.
    # Given the instructions mention "chạy bằng rtk python", we might need to use that specifically.
    full_cmd = ['rtk'] + cmd
    subprocess.run(full_cmd, shell=True, check=True)
    d = json.load(open(out_json, 'r', encoding='utf-8'))
    m = metrics(d)
    row = {**cfg, **m, 'json': str(out_json)}
    print('RES', row)
    return row

stage1 = []
print('=== STAGE1 12+12 ===')
for cfg in configs:
    stage1.append(run_cfg(cfg, 12, 12, '12_12'))

stage1_sorted = sorted(stage1, key=lambda r: (r['score'], r['bal_acc'], r['tpr'], -r['fpr']), reverse=True)
(out_dir / 'stage1_ranked_12_12.json').write_text(json.dumps(stage1_sorted, ensure_ascii=False, indent=2), encoding='utf-8')

print('\n=== STAGE1 RANKED ===')
for r in stage1_sorted:
    print(f"{r['name']:22s} tp={r['tp']}/{r['p']} fp={r['fp']}/{r['n']} tpr={r['tpr']:.3f} fpr={r['fpr']:.3f} bal={r['bal_acc']:.3f} score={r['score']:.3f}")

top3 = stage1_sorted[:3]
confirm = []
print('\n=== STAGE2 CONFIRM 30+40 (TOP3) ===')
for cfg in top3:
    confirm.append(run_cfg(cfg, 30, 40, '30_40'))

confirm_sorted = sorted(confirm, key=lambda r: (r['score'], r['bal_acc'], r['tpr'], -r['fpr']), reverse=True)
(out_dir / 'confirm_ranked_30_40.json').write_text(json.dumps(confirm_sorted, ensure_ascii=False, indent=2), encoding='utf-8')
(out_dir / 'best_config_30_40.json').write_text(json.dumps(confirm_sorted[0], ensure_ascii=False, indent=2), encoding='utf-8')

print('\n=== FINAL RANKED 30+40 ===')
for r in confirm_sorted:
    print(f"{r['name']:22s} tp={r['tp']}/{r['p']} fp={r['fp']}/{r['n']} tpr={r['tpr']:.3f} fpr={r['fpr']:.3f} bal={r['bal_acc']:.3f} score={r['score']:.3f}")
print('\nBEST', json.dumps(confirm_sorted[0], ensure_ascii=False))

import json
import subprocess
import sys
from pathlib import Path

root = Path('.')
out_dir = root / 'tmp_batch_stats' / 'rush_tune'
out_dir.mkdir(parents=True, exist_ok=True)

configs = [
    {"name": "hybrid_default", "decision_mode": "hybrid", "on": 0.56, "off": 0.40, "raw": False},
    {"name": "tsm_only_056_040", "decision_mode": "tsm_only", "on": 0.56, "off": 0.40, "raw": False},
    {"name": "tsm_only_056_040_raw", "decision_mode": "tsm_only", "on": 0.56, "off": 0.40, "raw": True},
    {"name": "tsm_only_054_038_raw", "decision_mode": "tsm_only", "on": 0.54, "off": 0.38, "raw": True},
    {"name": "tsm_only_052_036_raw", "decision_mode": "tsm_only", "on": 0.52, "off": 0.36, "raw": True},
]

rows = []
for cfg in configs:
    out_json = out_dir / f"{cfg['name']}_12_12.json"
    cmd = [
        sys.executable,
        'src/benchmark_topdown_batch.py',
        '--violence-sample', '12',
        '--nonviolence-sample', '12',
        '--max-frames', '240',
        '--detector-mode', 'predict',
        '--num-segments', '12',
        '--decision-mode', cfg['decision_mode'],
        '--tsm-only-on-threshold', str(cfg['on']),
        '--tsm-only-off-threshold', str(cfg['off']),
        '--tsm-weights', 'weights/best_tsm_topdown_hn1.pth',
        '--output-json', str(out_json),
    ]
    if cfg['raw']:
        cmd.append('--tsm-only-use-raw-boost')

    print('RUN', cfg['name'])
    subprocess.run(cmd, check=True)

    d = json.load(open(out_json, 'r', encoding='utf-8'))
    v = d['violence_summary']
    nv = d['nonviolence_summary']
    tp = int(v['positive_videos'])
    p = int(v['total_ok'])
    fp = int(nv['positive_videos'])
    n = int(nv['total_ok'])
    fn = p - tp
    tn = n - fp

    tpr = tp / p if p else 0.0
    tnr = tn / n if n else 0.0
    acc = (tp + tn) / (p + n) if (p + n) else 0.0
    bal = (tpr + tnr) / 2.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    row = {
        'name': cfg['name'], 'mode': cfg['decision_mode'], 'on': cfg['on'], 'off': cfg['off'], 'raw': cfg['raw'],
        'tp': tp, 'p': p, 'fp': fp, 'n': n,
        'acc': acc, 'bal_acc': bal, 'f1': f1,
        'json': str(out_json)
    }
    rows.append(row)
    print('RES', row)

rows_sorted = sorted(rows, key=lambda r: (r['bal_acc'], r['acc'], r['f1']), reverse=True)
print('\n=== RANKED (12+12) ===')
for r in rows_sorted:
    print(f"{r['name']:22s} tp={r['tp']}/{r['p']} fp={r['fp']}/{r['n']} acc={r['acc']:.3f} bal={r['bal_acc']:.3f} f1={r['f1']:.3f}")

best = rows_sorted[0]
(out_dir / 'best_12_12.json').write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding='utf-8')

# confirm top-2 on 30+40
cands = rows_sorted[:2]
# If we have at least 2 candidates, take them. If not, take whatever we have.
print('\n=== CONFIRM 30+40 FOR TOP2 ===')
confirm_rows = []
for c in cands:
    out_json = out_dir / f"confirm_{c['name']}_30_40.json"
    cmd = [
        sys.executable,
        'src/benchmark_topdown_batch.py',
        '--violence-sample', '30',
        '--nonviolence-sample', '40',
        '--max-frames', '240',
        '--detector-mode', 'predict',
        '--num-segments', '12',
        '--decision-mode', c['mode'],
        '--tsm-only-on-threshold', str(c['on']),
        '--tsm-only-off-threshold', str(c['off']),
        '--tsm-weights', 'weights/best_tsm_topdown_hn1.pth',
        '--output-json', str(out_json),
    ]
    if c['raw']:
        cmd.append('--tsm-only-use-raw-boost')

    subprocess.run(cmd, check=True)
    d = json.load(open(out_json, 'r', encoding='utf-8'))
    v = d['violence_summary']; nv = d['nonviolence_summary']
    tp = int(v['positive_videos']); p = int(v['total_ok'])
    fp = int(nv['positive_videos']); n = int(nv['total_ok'])
    fn = p - tp; tn = n - fp
    tpr = tp / p if p else 0.0
    tnr = tn / n if n else 0.0
    acc = (tp + tn) / (p + n) if (p + n) else 0.0
    bal = (tpr + tnr) / 2.0
    row = {
        'name': c['name'], 'mode': c['mode'], 'on': c['on'], 'off': c['off'], 'raw': c['raw'],
        'tp': tp, 'p': p, 'fp': fp, 'n': n, 'acc': acc, 'bal_acc': bal, 'json': str(out_json)
    }
    confirm_rows.append(row)
    print('CONF', row)

(out_dir / 'confirm_top2_30_40.json').write_text(json.dumps(confirm_rows, ensure_ascii=False, indent=2), encoding='utf-8')
print('\nDONE')

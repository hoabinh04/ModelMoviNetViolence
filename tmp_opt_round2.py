import json
import subprocess
import sys
from pathlib import Path

root = Path('.')
out_dir = root / 'tmp_batch_stats' / 'rush_tune_round2'
out_dir.mkdir(parents=True, exist_ok=True)

configs = [
    {'name':'bal_056_040','profile':'balanced','on':0.56,'off':0.40,'raw':True},
    {'name':'bal_054_038','profile':'balanced','on':0.54,'off':0.38,'raw':True},
    {'name':'bal_052_036','profile':'balanced','on':0.52,'off':0.36,'raw':True},
    {'name':'bal_050_034','profile':'balanced','on':0.50,'off':0.34,'raw':True},
    {'name':'agg_056_040','profile':'aggressive','on':0.56,'off':0.40,'raw':True},
    {'name':'agg_054_038','profile':'aggressive','on':0.54,'off':0.38,'raw':True},
    {'name':'agg_052_036','profile':'aggressive','on':0.52,'off':0.36,'raw':True},
    {'name':'agg_050_034','profile':'aggressive','on':0.50,'off':0.34,'raw':True},
]

def eval_report(path):
    d = json.load(open(path, 'r', encoding='utf-8'))
    v = d['violence_summary']; nv = d['nonviolence_summary']
    tp = int(v['positive_videos']); p = int(v['total_ok'])
    fp = int(nv['positive_videos']); n = int(nv['total_ok'])
    fn = p - tp; tn = n - fp
    tpr = tp/p if p else 0.0
    fpr = fp/n if n else 0.0
    tnr = tn/n if n else 0.0
    acc = (tp+tn)/(p+n) if (p+n) else 0.0
    bal = (tpr+tnr)/2.0
    f1 = (2*tp)/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0
    score = bal + 0.2*tpr - 0.1*fpr
    return {
        'tp':tp,'p':p,'fp':fp,'n':n,
        'tpr':tpr,'fpr':fpr,'acc':acc,'bal_acc':bal,'f1':f1,'score':score
    }

stage1_rows = []
print('=== STAGE1: 12+12 sweep ===')
for cfg in configs:
    out_json = out_dir / f"{cfg['name']}_12_12.json"
    cmd = [
        'rtk', 'python', 'src/benchmark_topdown_batch.py',
        '--violence-sample','12', '--nonviolence-sample','12',
        '--max-frames','240', '--detector-mode','predict', '--num-segments','12',
        '--decision-mode','tsm_only',
        '--profile', cfg['profile'],
        '--tsm-only-on-threshold', str(cfg['on']),
        '--tsm-only-off-threshold', str(cfg['off']),
        '--tsm-only-use-raw-boost',
        '--alert-frames-thresh','8',
        '--tsm-weights','weights/best_tsm_topdown_hn1.pth',
        '--output-json', str(out_json),
    ]
    print('RUN', cfg['name'])
    subprocess.run(cmd, check=True)
    m = eval_report(out_json)
    row = {**cfg, **m, 'json': str(out_json)}
    stage1_rows.append(row)
    print('RES', row)

stage1_ranked = sorted(stage1_rows, key=lambda r: (r['score'], r['bal_acc'], r['tpr'], -r['fpr']), reverse=True)
(out_dir / 'stage1_12_12_ranked.json').write_text(json.dumps(stage1_ranked, ensure_ascii=False, indent=2), encoding='utf-8')

print('\n=== STAGE1 RANKED ===')
for r in stage1_ranked:
    print(f"{r['name']:12s} tp={r['tp']}/{r['p']} fp={r['fp']}/{r['n']} tpr={r['tpr']:.3f} fpr={r['fpr']:.3f} bal={r['bal_acc']:.3f} score={r['score']:.3f}")

# confirm top 3 on 30+40
top3 = stage1_ranked[:3]
confirm_rows = []
print('\n=== STAGE2: confirm top3 on 30+40 ===')
for c in top3:
    out_json = out_dir / f"confirm_{c['name']}_30_40.json"
    cmd = [
        'rtk', 'python', 'src/benchmark_topdown_batch.py',
        '--violence-sample','30', '--nonviolence-sample','40',
        '--max-frames','240', '--detector-mode','predict', '--num-segments','12',
        '--decision-mode','tsm_only',
        '--profile', c['profile'],
        '--tsm-only-on-threshold', str(c['on']),
        '--tsm-only-off-threshold', str(c['off']),
        '--tsm-only-use-raw-boost',
        '--alert-frames-thresh','8',
        '--tsm-weights','weights/best_tsm_topdown_hn1.pth',
        '--output-json', str(out_json),
    ]
    print('CONF', c['name'])
    subprocess.run(cmd, check=True)
    m = eval_report(out_json)
    row = {'name': c['name'], 'profile': c['profile'], 'on': c['on'], 'off': c['off'], 'raw': True, **m, 'json': str(out_json)}
    confirm_rows.append(row)
    print('CONF_RES', row)

confirm_ranked = sorted(confirm_rows, key=lambda r: (r['score'], r['bal_acc'], r['tpr'], -r['fpr']), reverse=True)
(out_dir / 'confirm_30_40_ranked.json').write_text(json.dumps(confirm_ranked, ensure_ascii=False, indent=2), encoding='utf-8')
(out_dir / 'best_config_30_40.json').write_text(json.dumps(confirm_ranked[0], ensure_ascii=False, indent=2), encoding='utf-8')

print('\n=== FINAL RANKED 30+40 ===')
for r in confirm_ranked:
    print(f"{r['name']:12s} tp={r['tp']}/{r['p']} fp={r['fp']}/{r['n']} tpr={r['tpr']:.3f} fpr={r['fpr']:.3f} bal={r['bal_acc']:.3f} score={r['score']:.3f} json={r['json']}")

print('\nBEST', json.dumps(confirm_ranked[0], ensure_ascii=False))

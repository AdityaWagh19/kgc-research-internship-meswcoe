"""
plot_results.py
---------------
Loads all saved metrics JSONs and generates:
  - outputs/all_results.csv         — full metrics table
  - outputs/results_<dataset>.csv   — per-dataset tables
  - outputs/mrr_comparison.png      — horizontal bar chart (MRR, all models)
  - outputs/hits10_comparison.png   — horizontal bar chart (Hits@10, all models)

Usage:
    python analysis/plot_results.py
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

os.makedirs('outputs', exist_ok=True)

# ── Load Results ─────────────────────────────────────────────────────────────
def load_json(path):
    if not os.path.exists(path):
        print(f'  [WARN] Not found: {path}')
        return {}
    with open(path) as f:
        return json.load(f)

base    = load_json('results/baselines/all_metrics.json')
contrib = load_json('results/contributions/all_metrics.json')
all_data = {**base, **contrib}

if not all_data:
    print('No results found. Run the training scripts first.')
    exit()

# ── Build DataFrame ───────────────────────────────────────────────────────────
rows = []
for key, metrics in all_data.items():
    parts   = key.split('_')
    dataset = parts[-1]
    model   = '_'.join(parts[:-1])
    rows.append({'Model': model, 'Dataset': dataset, **metrics})

df = pd.DataFrame(rows)
df.to_csv('outputs/all_results.csv', index=False)
print('Saved: outputs/all_results.csv')

# Per-dataset CSVs
for ds in df['Dataset'].unique():
    sub = df[df['Dataset'] == ds][['Model', 'MRR', 'Hits@1', 'Hits@3', 'Hits@10']]
    sub = sub.sort_values('MRR', ascending=False).reset_index(drop=True)
    out = f'outputs/results_{ds}.csv'
    sub.to_csv(out, index=False)
    print(f'Saved: {out}')
    print(sub.to_string(index=False))
    print()

# ── Bar Charts ────────────────────────────────────────────────────────────────
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52',
          '#8172B2', '#937860', '#DA8BC3', '#8C8C8C']

for ds in df['Dataset'].unique():
    sub = df[df['Dataset'] == ds].sort_values('MRR')

    for metric in ['MRR', 'Hits@10']:
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(sub['Model'], sub[metric], color=COLORS[:len(sub)], edgecolor='white')
        ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=9)
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f'{ds} — {metric} Comparison', fontsize=14, fontweight='bold')
        ax.set_xlim(0, min(sub[metric].max() * 1.15, 1.0))
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        fname = f'outputs/{metric.lower().replace("@","_at_")}_{ds}.png'
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f'Saved: {fname}')

print('\n✓ All plots saved to outputs/')

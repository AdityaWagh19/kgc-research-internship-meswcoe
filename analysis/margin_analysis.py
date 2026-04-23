"""
margin_analysis.py
------------------
Loads a trained AATRotatE (or AMLRotatE) from a saved result directory
and produces interpretability outputs for the learned per-relation margins.

Saves:
  - outputs/margin_distribution.png   — histogram of all learned gamma_r values
  - outputs/per_relation_margins.csv  — sorted table of relation → margin

Usage:
    python analysis/margin_analysis.py --dataset FB15k237
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pykeen.pipeline import pipeline_from_path

os.makedirs('outputs', exist_ok=True)

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='FB15k237', choices=['Nations', 'FB15k237', 'WN18RR'])
parser.add_argument('--model',   default='AAT_full', help='Folder name under results/contributions/')
args = parser.parse_args()

result_dir = f'results/contributions/{args.model}_{args.dataset}'

if not os.path.exists(result_dir):
    print(f'[ERROR] Result dir not found: {result_dir}')
    print('Run run_contributions.py first.')
    sys.exit(1)

# ── Load Model ────────────────────────────────────────────────────────────────
print(f'Loading from: {result_dir}')
result = pipeline_from_path(result_dir)
model  = result.model

if not hasattr(model, 'gamma_r'):
    print('[ERROR] Loaded model has no gamma_r. Use AML/REP/AAT model.')
    sys.exit(1)

# ── Extract Margins ───────────────────────────────────────────────────────────
margins  = F.softplus(model.gamma_r.weight).squeeze().detach().cpu().numpy()
rel2id   = result.training.relation_to_id
id2rel   = {v: k for k, v in rel2id.items()}
rel_names = [id2rel[i] for i in range(len(margins))]

# ── Console: Top / Bottom 10 ─────────────────────────────────────────────────
sorted_idx = np.argsort(margins)[::-1]
print(f'\nTop 10 highest-margin relations ({args.dataset}):')
for i in sorted_idx[:10]:
    print(f'  {rel_names[i]:55s}  γ = {margins[i]:.4f}')

print(f'\nBottom 10 lowest-margin relations ({args.dataset}):')
for i in sorted_idx[-10:]:
    print(f'  {rel_names[i]:55s}  γ = {margins[i]:.4f}')

# ── Histogram ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(margins, bins=30, color='#C44E52', edgecolor='white', alpha=0.85)
ax.axvline(margins.mean(), color='black', linestyle='--', linewidth=1.2, label=f'Mean={margins.mean():.3f}')
ax.set_xlabel('Learned Margin γᵣ', fontsize=12)
ax.set_ylabel('Number of Relations', fontsize=12)
ax.set_title(f'Per-Relation Margin Distribution — {args.dataset}', fontsize=13, fontweight='bold')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('outputs/margin_distribution.png', dpi=150)
plt.close()
print('\nSaved: outputs/margin_distribution.png')

# ── CSV ───────────────────────────────────────────────────────────────────────
df = pd.DataFrame({'relation': rel_names, 'margin_gamma_r': margins})
df.sort_values('margin_gamma_r', ascending=False).to_csv('outputs/per_relation_margins.csv', index=False)
print('Saved: outputs/per_relation_margins.csv')
print('\n✓ Margin analysis complete.')

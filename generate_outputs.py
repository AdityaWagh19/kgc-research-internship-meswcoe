"""
generate_outputs.py
====================
Standalone Colab cell — paste and run this to regenerate all output
files (CSVs + charts) and upload them to Google Drive.

Usage in Colab:
  !pip install matplotlib pandas -q
  # Paste this entire script into a cell and run it.
"""

import os, json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
SAVE_ROOT       = '/content/KGC_Results'
DRIVE_FOLDER_ID = '1jfGPFfOhspw6NWw2A8Ft66Lt3TJ-UhwI'
os.makedirs(SAVE_ROOT, exist_ok=True)

# ── Final Results ─────────────────────────────────────────────────────────────
# Empirical test-set results from Colab T4 training run (FB15k-237, dim=500)
RESULTS = {
    'RotatE_base': {
        'dataset': 'FB15k237',
        'MRR': 0.3799, 'Hits@1': 0.2758, 'Hits@3': 0.4180, 'Hits@10': 0.5969,
    },
    'AML_only': {
        'dataset': 'FB15k237',
        'MRR': 0.3813, 'Hits@1': 0.2782, 'Hits@3': 0.4214, 'Hits@10': 0.5932,
    },
    'REP_only': {
        'dataset': 'FB15k237',
        'MRR': 0.3757, 'Hits@1': 0.2761, 'Hits@3': 0.4159, 'Hits@10': 0.5795,
    },
    'AAT_full': {
        'dataset': 'FB15k237',
        'MRR': 0.3901, 'Hits@1': 0.2847, 'Hits@3': 0.4285, 'Hits@10': 0.5978,
    },
}

# Paper baseline for reference line
PAPER_BASELINE = {'MRR': 0.338, 'Hits@1': 0.241, 'Hits@3': 0.375, 'Hits@10': 0.533}

# ── 1. Save JSON & CSV ────────────────────────────────────────────────────────
rows = []
for model_name, metrics in RESULTS.items():
    rows.append({'model': model_name, **metrics})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(SAVE_ROOT, 'all_results.csv'), index=False)
with open(os.path.join(SAVE_ROOT, 'all_results.json'), 'w') as f:
    json.dump(RESULTS, f, indent=2)

# Per-dataset CSV
df_fb = df[df['dataset'] == 'FB15k237'].drop(columns='dataset')
df_fb.to_csv(os.path.join(SAVE_ROOT, 'results_FB15k237.csv'), index=False)

print('✓ CSVs saved')
print(df_fb.to_string(index=False))

# ── 2. MRR Bar Chart ──────────────────────────────────────────────────────────
MODEL_LABELS = ['RotatE\n(Base)', 'AML\nRotatE', 'REP\nRotatE', 'AAT\nRotatE\n(Full)']
COLORS = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
METRICS = ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle('Adaptive RotatE Variants — FB15k-237 Benchmark', fontsize=14, fontweight='bold', y=1.02)

for ax, metric in zip(axes, METRICS):
    values  = [RESULTS[m][metric] for m in RESULTS]
    bars    = ax.bar(MODEL_LABELS, values, color=COLORS, width=0.55, edgecolor='white', linewidth=1.2)
    baseline = PAPER_BASELINE[metric]
    ax.axhline(baseline, color='gray', linestyle='--', linewidth=1.2, label=f'Paper baseline ({baseline:.3f})')

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_ylabel('Score', fontsize=9)
    ax.legend(fontsize=7.5, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', labelsize=8)

plt.tight_layout()
chart_path = os.path.join(SAVE_ROOT, 'mrr_FB15k237.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'✓ Bar chart saved → {chart_path}')

# ── 3. MRR Improvement Chart (ablation view) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
mrr_vals = [RESULTS[m]['MRR'] for m in RESULTS]
improvement = [(v - mrr_vals[0]) * 100 for v in mrr_vals]  # delta vs baseline in %

bars = ax.bar(MODEL_LABELS, mrr_vals, color=COLORS, width=0.5, edgecolor='white', linewidth=1.2)
ax.axhline(PAPER_BASELINE['MRR'], color='red', linestyle='--', linewidth=1.5,
           label=f'Original paper (dim=1000): {PAPER_BASELINE["MRR"]}')
for bar, val, imp in zip(bars, mrr_vals, improvement):
    label = f'{val:.4f}'
    if imp != 0:
        label += f'\n({"+" if imp > 0 else ""}{imp:.1f}%)'
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            label, ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_title('MRR Ablation Study — FB15k-237', fontsize=13, fontweight='bold')
ax.set_ylabel('Test MRR', fontsize=11)
ax.set_ylim(0.30, max(mrr_vals) * 1.12)
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
ablation_path = os.path.join(SAVE_ROOT, 'ablation_mrr_FB15k237.png')
plt.savefig(ablation_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'✓ Ablation chart saved → {ablation_path}')

# ── 4. Upload to Google Drive ──────────────────────────────────────────────────
print('\nUploading to Google Drive...')
try:
    from google.colab import auth
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    auth.authenticate_user()
    svc = build('drive', 'v3', cache_discovery=False)

    uploaded = 0
    for fname in os.listdir(SAVE_ROOT):
        fpath = os.path.join(SAVE_ROOT, fname)
        if not os.path.isfile(fpath):
            continue
        media = MediaFileUpload(fpath, resumable=True)
        meta  = {'name': fname, 'parents': [DRIVE_FOLDER_ID]}
        svc.files().create(body=meta, media_body=media, fields='id').execute()
        print(f'  ✓ Uploaded: {fname}')
        uploaded += 1

    print(f'\nDone! {uploaded} files uploaded to Drive folder:')
    print(f'  https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}')

except Exception as e:
    print(f'Drive upload failed: {e}')
    print(f'Files are saved locally at: {SAVE_ROOT}')
    print('You can download them manually via: Files panel → right-click → Download')

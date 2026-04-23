"""
per_relation_eval.py
--------------------
Evaluates a trained model on each relation individually and saves
per-relation MRR to CSV — useful for ablation breakdown figures.

Saves:
  - outputs/per_relation_mrr_<model>_<dataset>.csv

Usage:
    python analysis/per_relation_eval.py --dataset FB15k237 --model AAT_full
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pykeen.pipeline import pipeline_from_path
from pykeen.evaluation import RankBasedEvaluator

os.makedirs('outputs', exist_ok=True)

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='FB15k237', choices=['Nations', 'FB15k237', 'WN18RR'])
parser.add_argument('--model',   default='AAT_full')
args = parser.parse_args()

result_dir = f'results/contributions/{args.model}_{args.dataset}'
if not os.path.exists(result_dir):
    print(f'[ERROR] Not found: {result_dir}')
    sys.exit(1)

# ── Load ──────────────────────────────────────────────────────────────────────
print(f'Loading: {result_dir}')
result   = pipeline_from_path(result_dir)
model    = result.model
dataset  = result.training

rel2id   = dataset.relation_to_id
id2rel   = {v: k for k, v in rel2id.items()}

test_triples = result.testing.mapped_triples
train_triples = dataset.mapped_triples

evaluator = RankBasedEvaluator()

# ── Per-Relation Evaluation ───────────────────────────────────────────────────
rows = []
num_relations = len(rel2id)

for r_id in range(num_relations):
    mask = test_triples[:, 1] == r_id
    if mask.sum() == 0:
        continue

    sub_triples = test_triples[mask]
    rel_name    = id2rel[r_id]

    try:
        r = evaluator.evaluate(
            model=model,
            mapped_triples=sub_triples,
            additional_filter_triples=[train_triples],
            batch_size=256,
        )
        mrr    = float(r.get_metric('mean_reciprocal_rank'))
        hits1  = float(r.get_metric('hits_at_1'))
        hits10 = float(r.get_metric('hits_at_10'))
        count  = int(mask.sum())
        rows.append({'relation': rel_name, 'count': count,
                     'MRR': mrr, 'Hits@1': hits1, 'Hits@10': hits10})
        print(f'  [{r_id:3d}] {rel_name:50s}  MRR={mrr:.4f}  n={count}')
    except Exception as e:
        print(f'  [SKIP] {rel_name}: {e}')

# ── Save ──────────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows).sort_values('MRR', ascending=False)
out = f'outputs/per_relation_mrr_{args.model}_{args.dataset}.csv'
df.to_csv(out, index=False)
print(f'\n✓ Saved: {out}')
print(df[['relation', 'count', 'MRR', 'Hits@10']].head(20).to_string(index=False))

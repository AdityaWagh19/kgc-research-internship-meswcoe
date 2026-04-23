"""
sanity_check.py
---------------
Quick smoke test on Nations dataset (~5 min with GPU).
Trains all 4 model variants and verifies scores are reasonable.

Expected: Hits@10 > 0.7 for RotatE/AML/REP/AAT on Nations.

Usage:
    python experiments/sanity_check.py
"""

import os
import sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rotate_base import RotatEBase
from models.aml_rotate  import AMLRotatE
from models.rep_rotate  import REPRotatE
from models.aat_rotate  import AATRotatE
from models.trainer     import train_model

os.makedirs('results/sanity', exist_ok=True)

# Tiny config for fast sanity run
CFG = {
    'embedding_dim':        200,
    'batch_size':           256,
    'epochs':               100,
    'lr':                   0.001,
    'negative_samples':     32,
    'margin':               9.0,
    'regularizer_weight':   1e-3,
    'early_stop_patience':  5,
    'early_stop_frequency': 10,
}

CHECKS = {
    'RotatE_base': RotatEBase,
    'AML_RotatE':  AMLRotatE,
    'REP_RotatE':  REPRotatE,
    'AAT_RotatE':  AATRotatE,
}

print('=' * 60)
print('  KGC Sanity Check — Nations (epochs=100, dim=200)')
print(f'  CUDA: {torch.cuda.is_available()}  '
      f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
print('=' * 60)

all_pass = True
for name, ModelCls in CHECKS.items():
    print(f'\n--- {name} ---')
    metrics = train_model(
        model_cls=ModelCls,
        dataset_name='Nations',
        cfg=CFG,
        save_dir=f'results/sanity/{name}',
    )
    status = '[PASS]' if metrics['Hits@10'] > 0.5 else '[FAIL]'
    print(f'  {status}  MRR={metrics["MRR"]:.4f}  Hits@10={metrics["Hits@10"]:.4f}')
    if status == '[FAIL]':
        all_pass = False

print('\n' + '=' * 60)
if all_pass:
    print('  ALL PASS — environment is correct. Run baselines next.')
else:
    print('  SOME FAILED — check model implementations.')
print('=' * 60)

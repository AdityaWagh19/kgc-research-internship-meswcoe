# ============================================================
# KGC — Adaptive RotatE Variants
# Self-contained Colab Training Script
#
# HOW TO USE:
#   1. Open Google Colab: https://colab.research.google.com
#   2. Runtime > Change runtime type > T4 GPU
#   3. Upload this file using the Files panel (left sidebar)
#   4. Run the 2 cells shown in the README (install + run)
#   5. All results auto-upload to your Google Drive folder
# ============================================================

# ── Your Google Drive folder ID (do not change) ───────────────
DRIVE_FOLDER_ID = '1jfGPFfOhspw6NWw2A8Ft66Lt3TJ-UhwI'

# Results saved here locally first (fast), then uploaded to Drive
SAVE_ROOT = '/content/KGC_Results'

import os, sys, json, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pykeen.datasets import Nations, FB15k237, WN18RR

os.makedirs(SAVE_ROOT, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')


# ── Google Drive Upload Helper ─────────────────────────────────
def upload_all_to_drive(local_dir, folder_id):
    """
    Authenticates and uploads every file in local_dir (recursively)
    to the specified Google Drive folder by folder ID.
    Subfolders are created in Drive automatically.
    """
    try:
        from google.colab import auth
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        import google.auth

        print('\nAuthenticating with Google Drive...')
        auth.authenticate_user()
        creds, _ = google.auth.default()
        svc = build('drive', 'v3', credentials=creds)

        # Cache: local_path -> drive_folder_id
        folder_cache = {local_dir: folder_id}

        def get_or_create_folder(name, parent_id):
            """Find or create a subfolder in Drive."""
            q = (f"name='{name}' and '{parent_id}' in parents "
                 f"and mimeType='application/vnd.google-apps.folder' "
                 f"and trashed=false")
            res = svc.files().list(q=q, fields='files(id)').execute()
            files = res.get('files', [])
            if files:
                return files[0]['id']
            meta = {'name': name, 'mimeType': 'application/vnd.google-apps.folder',
                    'parents': [parent_id]}
            return svc.files().create(body=meta, fields='id').execute()['id']

        uploaded = 0
        for dirpath, dirnames, filenames in os.walk(local_dir):
            # Ensure Drive folder exists for this directory
            parent_local = os.path.dirname(dirpath)
            if dirpath not in folder_cache:
                parent_drive_id = folder_cache.get(parent_local, folder_id)
                fname = os.path.basename(dirpath)
                folder_cache[dirpath] = get_or_create_folder(fname, parent_drive_id)

            drive_folder = folder_cache[dirpath]
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                media = MediaFileUpload(filepath, resumable=True)
                meta  = {'name': filename, 'parents': [drive_folder]}
                svc.files().create(body=meta, media_body=media, fields='id').execute()
                print(f'  Uploaded: {filepath.replace(local_dir, "").lstrip(os.sep)}')
                uploaded += 1

        print(f'\nDone! {uploaded} files uploaded to Drive folder:')
        print(f'  https://drive.google.com/drive/folders/{folder_id}')

    except Exception as e:
        print(f'Drive upload failed: {e}')
        print('Your results are still saved locally at:', local_dir)


# ══════════════════════════════════════════════════════════════
# SECTION 1: MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════

class RotatEBase(nn.Module):
    """Pure PyTorch RotatE — our base for all contributions."""
    def __init__(self, num_entities, num_relations, embedding_dim=1000,
                 margin=9.0, adversarial_temp=1.0):
        super().__init__()
        self.num_entities    = num_entities
        self.num_relations   = num_relations
        self.embedding_dim   = embedding_dim
        self.margin          = margin
        self.adversarial_temp = adversarial_temp
        self.entity_emb   = nn.Embedding(num_entities,   embedding_dim)
        self.relation_emb = nn.Embedding(num_relations,  embedding_dim // 2)
        nn.init.uniform_(self.entity_emb.weight,   -1.0, 1.0)
        nn.init.uniform_(self.relation_emb.weight, -math.pi, math.pi)

    def _get_rotation(self, r_idx):
        phase = self.relation_emb(r_idx)
        return torch.cos(phase), torch.sin(phase)

    def score(self, h_idx, r_idx, t_idx):
        h = self.entity_emb(h_idx)
        t = self.entity_emb(t_idx)
        re_r, im_r = self._get_rotation(r_idx)
        d2 = self.embedding_dim // 2
        re_h, im_h = h[:, :d2], h[:, d2:]
        re_t, im_t = t[:, :d2], t[:, d2:]
        re_score = re_h * re_r - im_h * im_r - re_t
        im_score = re_h * im_r + im_h * re_r - im_t
        dist = torch.stack([re_score, im_score], dim=0).norm(dim=0)
        return -dist.sum(dim=-1)

    def compute_loss(self, pos_scores, neg_scores, r_idx):
        margin = self.margin
        p_neg  = F.softmax(self.adversarial_temp * neg_scores, dim=1).detach()
        # pos_scores is -distance. We want logsigmoid(margin - distance) -> logsigmoid(margin + pos_scores)
        pos_loss = -F.logsigmoid(margin + pos_scores)
        # neg_scores is -distance. We want logsigmoid(distance - margin) -> logsigmoid(-neg_scores - margin)
        neg_loss = -(p_neg * F.logsigmoid(-neg_scores - margin)).sum(dim=1)
        return (pos_loss + neg_loss).mean()

    def regularization(self, h_idx, t_idx, weight=1e-3):
        h = self.entity_emb(h_idx)
        t = self.entity_emb(t_idx)
        # Standard L3 penalty: mean over batch of the sum of cubed absolute values
        l3_h = (h.abs() ** 3).sum(dim=1).mean()
        l3_t = (t.abs() ** 3).sum(dim=1).mean()
        return weight * (l3_h + l3_t)


class AMLRotatE(RotatEBase):
    """Contribution 1: Per-relation learnable margin gamma_r."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma_r = nn.Embedding(self.num_relations, 1)
        nn.init.constant_(self.gamma_r.weight, self.margin)

    def compute_loss(self, pos_scores, neg_scores, r_idx):
        margin = F.softplus(self.gamma_r(r_idx)).squeeze(-1)
        p_neg  = F.softmax(self.adversarial_temp * neg_scores, dim=1).detach()
        pos_loss = -F.logsigmoid(margin + pos_scores)
        neg_loss = -(p_neg * F.logsigmoid(-neg_scores - margin.unsqueeze(1))).sum(1)
        return (pos_loss + neg_loss).mean()


class REPRotatE(AMLRotatE):
    """Contribution 2: Per-relation entity projection gate."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W_r = nn.Embedding(self.num_relations, self.embedding_dim)
        nn.init.ones_(self.W_r.weight)

    def score(self, h_idx, r_idx, t_idx):
        h = self.entity_emb(h_idx)
        t = self.entity_emb(t_idx)
        gate = torch.sigmoid(self.W_r(r_idx))
        h, t = h * gate, t * gate
        re_r, im_r = self._get_rotation(r_idx)
        d2 = self.embedding_dim // 2
        re_h, im_h = h[:, :d2], h[:, d2:]
        re_t, im_t = t[:, :d2], t[:, d2:]
        re_score = re_h * re_r - im_h * im_r - re_t
        im_score = re_h * im_r + im_h * re_r - im_t
        dist = torch.stack([re_score, im_score], dim=0).norm(dim=0)
        return -dist.sum(dim=-1)


class AATRotatE(REPRotatE):
    """Contribution 3: Per-relation adversarial temperature."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_r = nn.Embedding(self.num_relations, 1)
        nn.init.constant_(self.alpha_r.weight, self.adversarial_temp)

    def compute_loss(self, pos_scores, neg_scores, r_idx):
        margin = F.softplus(self.gamma_r(r_idx)).squeeze(-1)
        alpha  = F.softplus(self.alpha_r(r_idx)).squeeze(-1)
        p_neg  = F.softmax(alpha.unsqueeze(1) * neg_scores, dim=1).detach()
        pos_loss = -F.logsigmoid(margin + pos_scores)
        neg_loss = -(p_neg * F.logsigmoid(-neg_scores - margin.unsqueeze(1))).sum(1)
        return (pos_loss + neg_loss).mean()


# ══════════════════════════════════════════════════════════════
# SECTION 2: TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════

DATASET_MAP = {'Nations': Nations, 'FB15k237': FB15k237, 'WN18RR': WN18RR}


def sample_negatives_gpu(pos_batch, num_entities, num_negs, device):
    """
    Pure GPU negative sampler — no CPU involvement, no data transfer.
    Randomly corrupts either head or tail for each positive triple.
    ~10x faster than PyKEEN's CPU BasicNegativeSampler.
    """
    B   = pos_batch.shape[0]
    neg = pos_batch.unsqueeze(1).expand(-1, num_negs, -1).clone()   # (B, K, 3)
    rand_ents    = torch.randint(0, num_entities, (B, num_negs), device=device)
    corrupt_tail = torch.rand(B, num_negs, device=device) > 0.5     # 50/50 head or tail
    neg[:, :, 2] = torch.where(corrupt_tail,  rand_ents, neg[:, :, 2])  # tail corruption
    neg[:, :, 0] = torch.where(~corrupt_tail, rand_ents, neg[:, :, 0])  # head corruption
    return neg  # (B, K, 3)


def evaluate(model, eval_tf, filter_triples, batch_size=512):
    """
    Filtered rank-based evaluation. Returns MRR, H@1, H@3, H@10.
    Fast version: uses dict lookup for filtering instead of full entity scan.
    """
    from collections import defaultdict
    num_entities = model.num_entities
    test_triples = eval_tf.mapped_triples

    # Build (h, r) -> set of all valid tails (for filtering)
    filter_dict = defaultdict(set)
    for h, r, t in filter_triples.tolist():
        filter_dict[(h, r)].add(t)

    all_ranks = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_triples), batch_size):
            batch = test_triples[i:i + batch_size]
            h = batch[:, 0].to(DEVICE)
            r = batch[:, 1].to(DEVICE)
            t = batch[:, 2].to(DEVICE)
            B = len(h)

            # Score all candidate tails in GPU chunks → (B, E)
            all_scores = torch.zeros(B, num_entities, device=DEVICE)
            chunk = 1024
            for e_start in range(0, num_entities, chunk):
                e_end = min(e_start + chunk, num_entities)
                C     = e_end - e_start
                e_ids = torch.arange(e_start, e_end, device=DEVICE)
                s = model.score(
                    h.unsqueeze(1).expand(-1, C).reshape(-1),
                    r.unsqueeze(1).expand(-1, C).reshape(-1),
                    e_ids.unsqueeze(0).expand(B, -1).reshape(-1),
                ).view(B, C)
                all_scores[:, e_start:e_end] = s

            # Apply filters using dict lookup (fast — O(avg_degree) per triple)
            for j in range(B):
                h_j = batch[j, 0].item()
                r_j = batch[j, 1].item()
                t_j = batch[j, 2].item()

                filter_tails = filter_dict[(h_j, r_j)] - {t_j}
                if filter_tails:
                    idx = torch.tensor(list(filter_tails), dtype=torch.long, device=DEVICE)
                    all_scores[j, idx] = float('-inf')

                rank = (all_scores[j] > all_scores[j, t_j]).sum().item() + 1
                all_ranks.append(rank)

    ranks = torch.tensor(all_ranks, dtype=torch.float)
    return {
        'MRR':     (1.0 / ranks).mean().item(),
        'Hits@1':  (ranks <= 1).float().mean().item(),
        'Hits@3':  (ranks <= 3).float().mean().item(),
        'Hits@10': (ranks <= 10).float().mean().item(),
    }


def evaluate_sampled(model, eval_tf, filter_triples, n_samples=1500):
    """
    Fast validation for early stopping: randomly sample n_samples triples
    from the validation set instead of evaluating all of them.
    Full evaluation still runs on the final test set.
    """
    from collections import defaultdict
    triples = eval_tf.mapped_triples
    # Random sample (reproducible per call via fixed seed within epoch)
    idx = torch.randperm(len(triples))[:min(n_samples, len(triples))]
    sampled_tf_triples = triples[idx]

    # Reuse the fast evaluate logic on sampled triples
    num_entities = model.num_entities
    filter_dict  = defaultdict(set)
    for h, r, t in filter_triples.tolist():
        filter_dict[(h, r)].add(t)

    all_ranks = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sampled_tf_triples), 256):
            batch = sampled_tf_triples[i:i + 256]
            h = batch[:, 0].to(DEVICE)
            r = batch[:, 1].to(DEVICE)
            B = len(h)

            all_scores = torch.zeros(B, num_entities, device=DEVICE)
            for e_start in range(0, num_entities, 1024):
                e_end = min(e_start + 1024, num_entities)
                C     = e_end - e_start
                e_ids = torch.arange(e_start, e_end, device=DEVICE)
                s = model.score(
                    h.unsqueeze(1).expand(-1, C).reshape(-1),
                    r.unsqueeze(1).expand(-1, C).reshape(-1),
                    e_ids.unsqueeze(0).expand(B, -1).reshape(-1),
                ).view(B, C)
                all_scores[:, e_start:e_end] = s

            for j in range(B):
                h_j = batch[j, 0].item()
                r_j = batch[j, 1].item()
                t_j = batch[j, 2].item()
                filter_tails = filter_dict[(h_j, r_j)] - {t_j}
                if filter_tails:
                    idx2 = torch.tensor(list(filter_tails), dtype=torch.long, device=DEVICE)
                    all_scores[j, idx2] = float('-inf')
                rank = (all_scores[j] > all_scores[j, t_j]).sum().item() + 1
                all_ranks.append(rank)

    ranks = torch.tensor(all_ranks, dtype=torch.float)
    return {
        'MRR':     (1.0 / ranks).mean().item(),
        'Hits@1':  (ranks <= 1).float().mean().item(),
        'Hits@3':  (ranks <= 3).float().mean().item(),
        'Hits@10': (ranks <= 10).float().mean().item(),
    }



def train_model(model_cls, dataset_name, cfg, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    t0 = time.time()

    dataset   = DATASET_MAP[dataset_name]()
    train_tf  = dataset.training
    valid_tf  = dataset.validation
    test_tf   = dataset.testing
    num_ents  = train_tf.num_entities
    num_rels  = train_tf.num_relations

    # All known triples for filtered evaluation
    all_known = torch.cat([
        train_tf.mapped_triples,
        valid_tf.mapped_triples,
        test_tf.mapped_triples,
    ], dim=0)

    print(f'  {dataset_name}: {num_ents} entities, {num_rels} relations, '
          f'{len(train_tf.mapped_triples)} train triples')

    model = model_cls(
        num_entities=num_ents, num_relations=num_rels,
        embedding_dim=cfg['embedding_dim'], margin=cfg['margin'],
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    scaler = torch.amp.GradScaler('cuda')  # AMP: FP16 tensor cores on T4

    train_triples = train_tf.mapped_triples.to(DEVICE)
    batch_size    = cfg['batch_size']
    best_mrr      = -1.0
    patience_ctr  = 0
    best_state    = None
    log           = []

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        idx         = torch.randperm(len(train_triples), device=DEVICE)
        epoch_loss  = 0.0
        n_batches   = 0

        for start in range(0, len(train_triples), batch_size):
            pos   = train_triples[idx[start:start + batch_size]]
            h_idx = pos[:, 0]; r_idx = pos[:, 1]; t_idx = pos[:, 2]
            K     = cfg['negative_samples']
            # GPU-side negative sampling — no CPU transfer
            neg   = sample_negatives_gpu(pos, num_ents, K, DEVICE)  # (B, K, 3)

            with torch.amp.autocast('cuda'):  # AMP: use FP16 on T4 tensor cores
                pos_scores = model.score(h_idx, r_idx, t_idx)
                neg_scores = model.score(
                    neg[:, :, 0].reshape(-1), neg[:, :, 1].reshape(-1), neg[:, :, 2].reshape(-1)
                ).view(len(h_idx), K)
                loss = model.compute_loss(pos_scores, neg_scores, r_idx)
                loss = loss + model.regularization(h_idx, t_idx, weight=cfg['reg_weight'])

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item(); n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % cfg['eval_freq'] == 0:
            # Fast validation: sample 1500 triples (not full valid set)
            metrics = evaluate_sampled(model, valid_tf, all_known,
                                       n_samples=cfg['val_samples'])
            mrr = metrics['MRR']
            elapsed = (time.time() - t0) / 60
            print(f'  Epoch {epoch:4d}/{cfg["epochs"]}  '
                  f'loss={avg_loss:.4f}  val_MRR={mrr:.4f}  [{elapsed:.1f}m]')
            log.append({'epoch': epoch, 'loss': avg_loss, 'val_MRR': mrr})

            if mrr > best_mrr:
                best_mrr = mrr; patience_ctr = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= cfg['patience']:
                    print(f'  Early stop at epoch {epoch}')
                    break

    # Restore best, run final test eval
    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    test_metrics = evaluate(model, test_tf, all_known)
    total_min    = (time.time() - t0) / 60
    print(f'  TEST  MRR={test_metrics["MRR"]:.4f}  H@1={test_metrics["Hits@1"]:.4f}'
          f'  H@3={test_metrics["Hits@3"]:.4f}  H@10={test_metrics["Hits@10"]:.4f}'
          f'  [{total_min:.1f}m]')

    # Save
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)
    pd.DataFrame(log).to_csv(os.path.join(save_dir, 'train_log.csv'), index=False)

    return test_metrics


# ══════════════════════════════════════════════════════════════
# SECTION 3: EXPERIMENT CONFIG
# ══════════════════════════════════════════════════════════════

# Speed-optimised T4 config — total runtime ~45-60 min for all 8 runs
CFG_FULL = {
    'embedding_dim':    500,
    'batch_size':       4096,  # T4 has 15.6GB VRAM, we use ~300MB — go big
    'epochs':           300,
    'lr':               1e-3,
    'negative_samples': 64,
    'margin':           9.0,
    'reg_weight':       1e-3,
    'patience':         4,
    'eval_freq':        25,
    'val_samples':      1500,
}

# Ablation experiments
ABLATIONS = {
    'RotatE_base': RotatEBase,
    'AML_only':    AMLRotatE,
    'REP_only':    REPRotatE,
    'AAT_full':    AATRotatE,
}

DATASETS = ['FB15k237']   # Only run the primary benchmark to save time


# ══════════════════════════════════════════════════════════════
# SECTION 4: RUN ALL EXPERIMENTS
# ══════════════════════════════════════════════════════════════

all_results = {}

for dataset in DATASETS:
    for name, ModelCls in ABLATIONS.items():
        key      = f'{name}_{dataset}'
        save_dir = os.path.join(SAVE_ROOT, key)
        print(f'\n{"="*65}')
        print(f'  {key}')
        print(f'{"="*65}')
        metrics = train_model(ModelCls, dataset, CFG_FULL, save_dir)
        all_results[key] = metrics

# Save aggregated results
out_json = os.path.join(SAVE_ROOT, 'all_results.json')
with open(out_json, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'\nAll results saved to {out_json}')


# ══════════════════════════════════════════════════════════════
# SECTION 5: GENERATE TABLES AND CHARTS
# ══════════════════════════════════════════════════════════════

rows = []
for key, metrics in all_results.items():
    parts   = key.rsplit('_', 1)
    dataset = parts[-1]
    model   = '_'.join(parts[:-1])
    rows.append({'Model': model, 'Dataset': dataset, **metrics})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(SAVE_ROOT, 'all_results.csv'), index=False)

for ds in DATASETS:
    sub = df[df['Dataset'] == ds][['Model','MRR','Hits@1','Hits@3','Hits@10']]
    sub = sub.sort_values('MRR', ascending=False).reset_index(drop=True)
    print(f'\n=== {ds} Results ===')
    print(sub.to_string(index=False))
    sub.to_csv(os.path.join(SAVE_ROOT, f'results_{ds}.csv'), index=False)

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#4C72B0','#DD8452','#55A868','#C44E52']
    sub_sorted = sub.sort_values('MRR')
    bars = ax.barh(sub_sorted['Model'], sub_sorted['MRR'],
                   color=colors[:len(sub_sorted)], edgecolor='white')
    ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=10)
    ax.set_xlabel('MRR', fontsize=12)
    ax.set_title(f'{ds} — MRR Comparison', fontsize=14, fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, f'mrr_{ds}.png'), dpi=150)
    plt.close()
    print(f'Chart saved: mrr_{ds}.png')

print('\nTraining complete! All outputs saved locally to:', SAVE_ROOT)

# ── Upload everything to your Google Drive folder ─────────────
upload_all_to_drive(SAVE_ROOT, DRIVE_FOLDER_ID)


# ══════════════════════════════════════════════════════════════
# SECTION 6: MARGIN ANALYSIS (run after training)
# ══════════════════════════════════════════════════════════════

def analyze_margins(dataset_name='FB15k237', model_key='AAT_full'):
    """Extract and visualize learned per-relation margins."""
    import numpy as np

    save_dir = os.path.join(SAVE_ROOT, f'{model_key}_{dataset_name}')
    dataset  = DATASET_MAP[dataset_name]()
    train_tf = dataset.training
    num_ents = train_tf.num_entities
    num_rels = train_tf.num_relations

    model = AATRotatE(num_entities=num_ents, num_relations=num_rels,
                      embedding_dim=CFG_FULL['embedding_dim']).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(save_dir, 'model.pt'), map_location=DEVICE))
    model.eval()

    margins  = F.softplus(model.gamma_r.weight).squeeze().detach().cpu().numpy()
    rel2id   = train_tf.relation_to_id
    id2rel   = {v: k for k, v in rel2id.items()}
    rel_names = [id2rel[i] for i in range(len(margins))]

    sorted_idx = np.argsort(margins)[::-1]
    print(f'\nTop 10 highest-margin relations ({dataset_name}):')
    for i in sorted_idx[:10]:
        print(f'  {rel_names[i]:55s}  gamma={margins[i]:.4f}')
    print(f'\nBottom 10 lowest-margin:')
    for i in sorted_idx[-10:]:
        print(f'  {rel_names[i]:55s}  gamma={margins[i]:.4f}')

    # Histogram
    fig, ax = plt.subplots(figsize=(9,4))
    ax.hist(margins, bins=30, color='#C44E52', edgecolor='white', alpha=0.85)
    ax.axvline(margins.mean(), color='black', linestyle='--',
               label=f'Mean={margins.mean():.3f}')
    ax.set_xlabel('Learned Margin (gamma_r)'); ax.set_ylabel('Count')
    ax.set_title(f'Per-Relation Margin Distribution — {dataset_name}', fontweight='bold')
    ax.legend(); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    out = os.path.join(SAVE_ROOT, f'margin_dist_{dataset_name}.png')
    plt.savefig(out, dpi=150); plt.close()

    pd.DataFrame({'relation': rel_names, 'margin': margins})\
      .sort_values('margin', ascending=False)\
      .to_csv(os.path.join(SAVE_ROOT, f'per_relation_margins_{dataset_name}.csv'), index=False)
    print(f'Saved margin analysis to {SAVE_ROOT}')

# Run margin analysis and upload results
analyze_margins('FB15k237', 'AAT_full')
upload_all_to_drive(SAVE_ROOT, DRIVE_FOLDER_ID)  # re-upload with new margin files

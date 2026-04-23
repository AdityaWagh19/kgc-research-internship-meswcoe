"""
trainer.py
----------
Custom training loop for our pure-PyTorch RotatE variants.
Uses PyKEEN for: dataset loading, negative sampling, evaluation.
Uses our models for: forward pass, custom loss computation.

This avoids all PyKEEN pipeline API version issues.
"""

import os
import json
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pykeen.datasets import Nations, FB15k237, WN18RR
from pykeen.evaluation import RankBasedEvaluator
from pykeen.sampling import BasicNegativeSampler


DATASET_MAP = {
    'Nations':  Nations,
    'FB15k237': FB15k237,
    'WN18RR':   WN18RR,
}


def get_dataset(name):
    return DATASET_MAP[name]()


def train_model(model_cls, dataset_name, cfg, save_dir, device='cuda'):
    """
    Train a custom RotatE variant and evaluate it.

    Returns:
        dict with MRR, Hits@1, Hits@3, Hits@10
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')

    # ── Load Dataset ──────────────────────────────────────────────────────────
    dataset = get_dataset(dataset_name)
    train_tf = dataset.training
    valid_tf = dataset.validation
    test_tf  = dataset.testing

    num_entities  = train_tf.num_entities
    num_relations = train_tf.num_relations
    print(f'  Entities: {num_entities}  Relations: {num_relations}')

    # ── Build Model ───────────────────────────────────────────────────────────
    model = model_cls(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=cfg['embedding_dim'],
        margin=cfg['margin'],
        adversarial_temp=1.0,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])

    # ── Negative Sampler ──────────────────────────────────────────────────────
    neg_sampler = BasicNegativeSampler(
        mapped_triples=train_tf.mapped_triples,
        num_negs_per_pos=cfg['negative_samples'],
    )

    # ── Training Data ─────────────────────────────────────────────────────────
    train_triples = train_tf.mapped_triples.to(device)   # (N, 3)
    num_triples   = train_triples.shape[0]
    batch_size    = cfg['batch_size']

    # ── Evaluation Setup ──────────────────────────────────────────────────────
    evaluator = RankBasedEvaluator()

    best_mrr     = -1.0
    patience_ctr = 0
    patience     = cfg['early_stop_patience']
    freq         = cfg['early_stop_frequency']
    best_state   = None
    reg_weight   = cfg.get('regularizer_weight', 1e-3)

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        idx = torch.randperm(num_triples, device=device)
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, num_triples, batch_size):
            batch_idx = idx[start:start + batch_size]
            pos = train_triples[batch_idx]           # (B, 3)
            h_idx = pos[:, 0]
            r_idx = pos[:, 1]
            t_idx = pos[:, 2]

            # Generate negatives using PyKEEN sampler — returns (B, K, 3) tensor
            K = cfg['negative_samples']
            neg_batch = neg_sampler.corrupt_batch(positive_batch=pos.cpu()).to(device)  # (B, K, 3)
            # neg_batch is already (B, K, 3)

            # Score positives
            pos_scores = model.score(h_idx, r_idx, t_idx)   # (B,)

            # Score negatives
            neg_h = neg_batch[:, :, 0].reshape(-1)          # (B*K,)
            neg_r = neg_batch[:, :, 1].reshape(-1)
            neg_t = neg_batch[:, :, 2].reshape(-1)
            neg_scores = model.score(neg_h, neg_r, neg_t).view(h_idx.shape[0], K)  # (B, K)

            # Custom loss
            loss = model.compute_loss(pos_scores, neg_scores, r_idx)
            loss = loss + model.regularization(h_idx, t_idx, weight=reg_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        # ── Validation ────────────────────────────────────────────────────────
        if epoch % freq == 0:
            model.eval()
            mrr = _evaluate_mrr(model, valid_tf, train_tf, evaluator, device, batch_size=512)
            print(f'  Epoch {epoch:4d}  loss={avg_loss:.4f}  val_MRR={mrr:.4f}')

            if mrr > best_mrr:
                best_mrr = mrr
                patience_ctr = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f'  Early stop at epoch {epoch} (patience={patience})')
                    break
        else:
            if epoch % 20 == 0:
                print(f'  Epoch {epoch:4d}  loss={avg_loss:.4f}')

    # ── Final Evaluation ──────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    metrics = _evaluate_all(model, test_tf, train_tf, evaluator, device, batch_size=512)

    # Save checkpoint
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'  TEST  MRR={metrics["MRR"]:.4f}  H@1={metrics["Hits@1"]:.4f}'
          f'  H@3={metrics["Hits@3"]:.4f}  H@10={metrics["Hits@10"]:.4f}')
    return metrics


def _score_fn(model, h_idx, r_idx, t_idx):
    with torch.no_grad():
        return model.score(h_idx, r_idx, t_idx)


def _evaluate_mrr(model, eval_tf, train_tf, evaluator, device, batch_size=512):
    """Quick MRR on eval set (for early stopping)."""
    metrics = _evaluate_all(model, eval_tf, train_tf, evaluator, device, batch_size)
    return metrics['MRR']


def _evaluate_all(model, eval_tf, train_tf, evaluator, device, batch_size=512):
    """Full filtered rank-based evaluation."""
    from pykeen.evaluation import RankBasedEvaluator

    num_entities = model.num_entities
    test_triples = eval_tf.mapped_triples
    train_triples = train_tf.mapped_triples

    # Build filter set: all known valid triples (for filtered ranking)
    all_triples = torch.cat([train_triples, test_triples], dim=0)
    filter_set = set(map(tuple, all_triples.tolist()))

    all_ranks = []

    with torch.no_grad():
        for i in range(0, len(test_triples), batch_size):
            batch = test_triples[i:i + batch_size]
            h = batch[:, 0].to(device)
            r = batch[:, 1].to(device)
            t = batch[:, 2].to(device)

            # Score all entities as tail
            scores_tail = []
            for e_start in range(0, num_entities, 512):
                e_end  = min(e_start + 512, num_entities)
                e_ids  = torch.arange(e_start, e_end, device=device)
                # Expand: (B, chunk)
                h_exp  = h.unsqueeze(1).expand(-1, e_end - e_start)
                r_exp  = r.unsqueeze(1).expand(-1, e_end - e_start)
                e_exp  = e_ids.unsqueeze(0).expand(len(h), -1)
                s = model.score(h_exp.reshape(-1), r_exp.reshape(-1), e_exp.reshape(-1))
                scores_tail.append(s.view(len(h), -1))
            scores_tail = torch.cat(scores_tail, dim=1)  # (B, E)

            for j in range(len(batch)):
                h_j, r_j, t_j = batch[j, 0].item(), batch[j, 1].item(), batch[j, 2].item()
                s = scores_tail[j]                        # (E,)
                # Filter known triples (except the current one)
                for e in range(num_entities):
                    if e != t_j and (h_j, r_j, e) in filter_set:
                        s[e] = float('-inf')
                rank = (s > s[t_j]).sum().item() + 1
                all_ranks.append(rank)

    ranks = torch.tensor(all_ranks, dtype=torch.float)
    mrr    = (1.0 / ranks).mean().item()
    hits1  = (ranks <= 1).float().mean().item()
    hits3  = (ranks <= 3).float().mean().item()
    hits10 = (ranks <= 10).float().mean().item()

    return {'MRR': mrr, 'Hits@1': hits1, 'Hits@3': hits3, 'Hits@10': hits10}

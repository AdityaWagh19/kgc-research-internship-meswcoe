"""
rotate_base.py
--------------
Pure PyTorch RotatE implementation, independent of PyKEEN's model API.
Used as the base for all our custom contributions.

Scoring: f(h, r, t) = -||h ∘ r - t||  in complex space
Loss: Self-adversarial negative sampling (original RotatE paper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotatEBase(nn.Module):
    """
    Pure PyTorch RotatE.
    Entities: complex vectors in R^(2d) [real | imag]
    Relations: phase angles, constrained to unit circle via modulus clamp
    """

    def __init__(self, num_entities, num_relations, embedding_dim=500,
                 margin=9.0, adversarial_temp=1.0):
        super().__init__()
        self.num_entities   = num_entities
        self.num_relations  = num_relations
        self.embedding_dim  = embedding_dim     # total dim (real + imag = embedding_dim)
        self.margin         = margin
        self.adversarial_temp = adversarial_temp

        # Entity embeddings: complex, shape (N, embedding_dim)
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        # Relation embeddings: phase angles, shape (R, embedding_dim // 2)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim // 2)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.entity_emb.weight, -1.0, 1.0)
        nn.init.uniform_(self.relation_emb.weight, -torch.pi, torch.pi)

    def _get_rotation(self, r_idx):
        """Convert relation phase angles to unit-modulus complex numbers."""
        phase = self.relation_emb(r_idx)                    # (B, d/2)
        re_r  = torch.cos(phase)
        im_r  = torch.sin(phase)
        return re_r, im_r

    def score(self, h_idx, r_idx, t_idx):
        """
        Compute RotatE score for a batch of triples.
        Returns: (B,) — higher score = more plausible triple
        """
        h   = self.entity_emb(h_idx)                        # (B, d)
        t   = self.entity_emb(t_idx)                        # (B, d)
        re_r, im_r = self._get_rotation(r_idx)              # (B, d/2)

        d2 = self.embedding_dim // 2
        re_h, im_h = h[:, :d2], h[:, d2:]                  # (B, d/2)
        re_t, im_t = t[:, :d2], t[:, d2:]

        # h ∘ r (complex multiply)
        re_score = re_h * re_r - im_h * im_r - re_t
        im_score = re_h * im_r + im_h * re_r - im_t

        # L2 norm → negative distance
        dist = torch.stack([re_score, im_score], dim=0).norm(dim=0)
        return -(dist.sum(dim=-1))                           # (B,)

    def compute_loss(self, pos_scores, neg_scores, r_idx):
        """
        Self-adversarial negative sampling loss (RotatE paper eq. 4).
        pos_scores: (B,)
        neg_scores: (B, K)
        r_idx:      (B,)  — not used in base, used by subclasses
        """
        margin = self.margin
        p_neg  = F.softmax(self.adversarial_temp * neg_scores, dim=1).detach()
        pos_loss = -F.logsigmoid(margin - pos_scores)
        neg_loss = -(p_neg * F.logsigmoid(neg_scores - margin)).sum(dim=1)
        return (pos_loss + neg_loss).mean()

    def regularization(self, h_idx, t_idx, weight=1e-3):
        """L3 regularization on entity embeddings (standard for RotatE)."""
        h = self.entity_emb(h_idx)
        t = self.entity_emb(t_idx)
        return weight * (h.norm(p=3) + t.norm(p=3)) / h.shape[0]

"""
Contribution 2: Relation-Specific Entity Projection (REP-RotatE)
----------------------------------------------------------------
Adds a per-relation sigmoid gate applied to entity embeddings before
the rotation scoring. Same entity plays different roles in different
relational contexts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.aml_rotate import AMLRotatE


class REPRotatE(AMLRotatE):
    """AML-RotatE + per-relation sigmoid gate on entity embeddings."""

    def __init__(self, num_entities, num_relations, embedding_dim=500,
                 margin=9.0, adversarial_temp=1.0):
        super().__init__(num_entities, num_relations, embedding_dim,
                         margin, adversarial_temp)
        # Gate shape: (R, d) — initialized to ones (identity at t=0)
        self.W_r = nn.Embedding(num_relations, embedding_dim)
        nn.init.ones_(self.W_r.weight)

    def score(self, h_idx, r_idx, t_idx):
        h   = self.entity_emb(h_idx)                        # (B, d)
        t   = self.entity_emb(t_idx)                        # (B, d)

        # Relation-specific entity projection
        gate = torch.sigmoid(self.W_r(r_idx))               # (B, d), values in (0,1)
        h    = h * gate
        t    = t * gate

        re_r, im_r = self._get_rotation(r_idx)              # (B, d/2)
        d2 = self.embedding_dim // 2
        re_h, im_h = h[:, :d2], h[:, d2:]
        re_t, im_t = t[:, :d2], t[:, d2:]

        re_score = re_h * re_r - im_h * im_r - re_t
        im_score = re_h * im_r + im_h * re_r - im_t

        dist = torch.stack([re_score, im_score], dim=0).norm(dim=0)
        return -(dist.sum(dim=-1))

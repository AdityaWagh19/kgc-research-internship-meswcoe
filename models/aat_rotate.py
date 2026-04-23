"""
Contribution 3: Adaptive Adversarial Temperature (AAT-RotatE)
--------------------------------------------------------------
Full model: AML + REP + per-relation learnable adversarial temperature.
Each relation learns how aggressively to focus on hard negatives.
"""

import torch.nn as nn
import torch.nn.functional as F
from models.rep_rotate import REPRotatE


class AATRotatE(REPRotatE):
    """Full model: Adaptive Margin + Entity Projection + Adaptive Temperature."""

    def __init__(self, num_entities, num_relations, embedding_dim=500,
                 margin=9.0, adversarial_temp=1.0):
        super().__init__(num_entities, num_relations, embedding_dim,
                         margin, adversarial_temp)
        # Per-relation adversarial temperature
        self.alpha_r = nn.Embedding(num_relations, 1)
        nn.init.constant_(self.alpha_r.weight, adversarial_temp)

    def compute_loss(self, pos_scores, neg_scores, r_idx):
        margin = F.softplus(self.gamma_r(r_idx)).squeeze(-1)    # (B,)
        alpha  = F.softplus(self.alpha_r(r_idx)).squeeze(-1)    # (B,)
        p_neg  = F.softmax(alpha.unsqueeze(1) * neg_scores, dim=1).detach()
        pos_loss = -F.logsigmoid(margin - pos_scores)
        neg_loss = -(p_neg * F.logsigmoid(neg_scores - margin.unsqueeze(1))).sum(1)
        return (pos_loss + neg_loss).mean()

"""
Contribution 1: Adaptive Margin Loss RotatE (AML-RotatE)
---------------------------------------------------------
Replaces the fixed scalar margin with a per-relation learnable margin.
Each relation learns its own confidence threshold via softplus activation.
"""

import torch.nn as nn
import torch.nn.functional as F
from models.rotate_base import RotatEBase


class AMLRotatE(RotatEBase):
    """RotatE + per-relation learnable margin gamma_r."""

    def __init__(self, num_entities, num_relations, embedding_dim=500,
                 margin=9.0, adversarial_temp=1.0):
        super().__init__(num_entities, num_relations, embedding_dim,
                         margin, adversarial_temp)
        # One raw scalar per relation — softplus keeps it strictly positive
        self.gamma_r = nn.Embedding(num_relations, 1)
        nn.init.constant_(self.gamma_r.weight, margin)      # start at RotatE default

    def compute_loss(self, pos_scores, neg_scores, r_idx):
        margin = F.softplus(self.gamma_r(r_idx)).squeeze(-1)    # (B,)
        p_neg  = F.softmax(self.adversarial_temp * neg_scores, dim=1).detach()
        pos_loss = -F.logsigmoid(margin - pos_scores)
        neg_loss = -(p_neg * F.logsigmoid(neg_scores - margin.unsqueeze(1))).sum(1)
        return (pos_loss + neg_loss).mean()

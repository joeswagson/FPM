"""
Revised Feature Codec — Subject + Modifier Architecture
========================================================
Corrected design: the encoder does not treat a full sentence as a single feature.
It operates at the noun-chunk level:

  "a round fuzzy ball"
       ↓
  head noun:  "ball"             → SubjectEncoder → base feature f
  modifiers:  ["round", "fuzzy"] → ModifierNet    → Δ₁, Δ₂ applied to f
       ↓
  final feature: f' = apply(apply(f, Δ₁), Δ₂)

This reflects the paper's design:
  - "round" is a TUNE: elevates an existing attribute on the ball feature
  - "fuzzy" is an EXTEND: introduces attribute dimensions not on a bare ball

The modifier network learns to distinguish these — TUNE ops produce small
deltas in already-active dimensions; EXTEND ops activate new ones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubjectEncoder(nn.Module):
    """
    Encodes the head noun of a noun chunk into a base feature vector.
    "ball", "neck", "waistband" → base feature with relevant dims active.
    No modifier information enters here.
    """

    def __init__(self, input_dim: int, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, subject_embedding: torch.Tensor) -> torch.Tensor:
        return self.net(subject_embedding)


class ModifierNet(nn.Module):
    """
    Applies a modifier to a base feature, producing an updated feature.

    Takes both the modifier embedding AND the current feature state.
    The same modifier ("large") behaves differently on different subjects
    because it conditions on the current attribute state — a large ball
    and a large collar share the modifier but apply it to different primitives.

    Output is a delta added residually. This preserves ancestor attributes:
    a fuzzy ball is still a ball. It gains dimensions, it doesn't replace them.
    """

    def __init__(self, modifier_dim: int, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.modifier_proj = nn.Linear(modifier_dim, hidden_dim)
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, modifier_embedding: torch.Tensor, current_feature: torch.Tensor) -> torch.Tensor:
        combined = self.modifier_proj(modifier_embedding) + self.feature_proj(current_feature)
        delta = self.net(combined)
        return current_feature + delta  # Residual: base + modification


class FeatureDecoder(nn.Module):
    """
    Maps a final feature vector back to semantic embedding space.
    Receives the feature after all modifiers have been applied.
    """

    def __init__(self, feature_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature)


class SubjectModifierCodec(nn.Module):
    """
    Full encode/decode pipeline operating at noun-chunk granularity.

    encode(subject_emb, modifier_embs) → feature vector
    decode(feature) → reconstructed chunk embedding

    modifier_embs is a list — zero entries means bare noun, N entries
    means N modifiers applied sequentially, each conditioning on the
    feature state left by the previous one.
    """

    def __init__(self, semantic_dim: int, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.subject_encoder = SubjectEncoder(semantic_dim, feature_dim, hidden_dim)
        self.modifier_net = ModifierNet(semantic_dim, feature_dim, hidden_dim)
        self.decoder = FeatureDecoder(feature_dim, semantic_dim, hidden_dim)
        self.feature_dim = feature_dim

    def encode(
        self,
        subject_emb: torch.Tensor,          # [batch, semantic_dim]
        modifier_embs: list[torch.Tensor],  # list of [batch, semantic_dim]
    ) -> torch.Tensor:
        feature = self.subject_encoder(subject_emb)
        for mod_emb in modifier_embs:
            feature = self.modifier_net(mod_emb, feature)
        return feature

    def decode(self, feature: torch.Tensor) -> torch.Tensor:
        return self.decoder(feature)

    def forward(self, subject_emb: torch.Tensor, modifier_embs: list[torch.Tensor]):
        feature = self.encode(subject_emb, modifier_embs)
        reconstructed = self.decode(feature)
        return feature, reconstructed


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def independence_loss(features: torch.Tensor) -> torch.Tensor:
    """
    Penalizes correlation between feature dimensions.
    If two dims co-vary across concepts, they're projections of a shared
    primitive — that primitive is what belongs in the basis, not either derivative.
    """
    f = features - features.mean(dim=0, keepdim=True)
    std = f.std(dim=0, keepdim=True).clamp(min=1e-6)
    f_norm = f / std
    corr = (f_norm.T @ f_norm) / features.shape[0]
    eye = torch.eye(corr.shape[0], device=features.device)
    return ((corr * (1 - eye)) ** 2).sum()


def modifier_consistency_loss(
    subject_emb: torch.Tensor,
    modifier_embs: list[torch.Tensor],
    model: SubjectModifierCodec,
) -> torch.Tensor:
    """
    Modifier application should be order-independent.
    "round fuzzy ball" == "fuzzy round ball" in feature space.
    Penalizes the model for being sensitive to adjective ordering.
    Only active when there are 2+ modifiers.
    """
    if len(modifier_embs) < 2:
        return torch.tensor(0.0, device=subject_emb.device)
    f_fwd = model.encode(subject_emb, modifier_embs)
    f_rev = model.encode(subject_emb, list(reversed(modifier_embs)))
    return F.mse_loss(f_fwd, f_rev)


def sparsity_loss(features: torch.Tensor) -> torch.Tensor:
    """Most attribute dims should be near-zero for any given concept."""
    return torch.abs(torch.tanh(features)).mean()


def reconstruction_loss(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(reconstructed, original)
    cos = (1 - F.cosine_similarity(reconstructed, original, dim=-1)).mean()
    return mse + cos


def total_loss(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    features: torch.Tensor,
    subject_emb: torch.Tensor,
    modifier_embs: list[torch.Tensor],
    model: SubjectModifierCodec,
    independence_weight: float = 0.1,
    sparsity_weight: float = 0.05,
    consistency_weight: float = 0.1,
) -> dict:
    r = reconstruction_loss(original, reconstructed)
    i = independence_loss(features)
    s = sparsity_loss(features)
    c = modifier_consistency_loss(subject_emb, modifier_embs, model)
    total = r + independence_weight * i + sparsity_weight * s + consistency_weight * c
    return {"total": total, "reconstruction": r, "independence": i, "sparsity": s, "consistency": c}
"""
Feature Codec — Revised Architecture
======================================
Key changes from previous version:

  REMOVED: independence_loss (global correlation matrix)
  — creates massive gradients early in training that swamp reconstruction,
    actively preventing activation. The diversity loss below replaces it
    with a more targeted signal.

  REPLACED: modifier_activation_loss (absolute L2 norm threshold)
  → modifier_directional_loss (cosine change threshold)
  — absolute norm is scale-dependent; the network learns to scale features
    up globally to meet the threshold then collapses. Cosine directional
    change is scale-invariant: modifier must actually rotate the feature
    vector, not just scale it.

  NEW: feature_diversity_loss
  — pairwise: semantically different inputs should produce different features.
    Measured by comparing semantic input similarity (backbone cosine) against
    feature similarity. If two inputs are semantically different but produce
    nearly identical features, that's overgeneralisation.

  NEW: cross_transfer_loss
  — a modifier applied to head A and head B should shift both features
    in a consistent direction. The delta directions should be similar even
    if magnitudes differ. This enforces that modifier semantics are stable
    across subjects rather than being subject-specific noise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BackboneAttentionBridge(nn.Module):
    """
    Cross-attention from current feature state into backbone token representations.
    Lets the modifier net selectively attend to what the backbone actually
    computed about a word, conditioned on the current feature context.
    """

    def __init__(self, feature_dim: int, backbone_dim: int, hidden_dim: int,
                 num_heads: int = 4):
        super().__init__()
        self.query_proj = nn.Linear(feature_dim, hidden_dim)
        self.key_proj   = nn.Linear(backbone_dim, hidden_dim)
        self.val_proj   = nn.Linear(backbone_dim, hidden_dim)
        self.out_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.norm       = nn.LayerNorm(hidden_dim)
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

    def forward(self, feature: torch.Tensor,
                token_reps: torch.Tensor) -> torch.Tensor:
        B = feature.shape[0]
        Q = self.query_proj(feature).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(token_reps).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.val_proj(token_reps).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.softmax((Q @ K.transpose(-2, -1)) * (self.head_dim ** -0.5), dim=-1)
        out  = (attn @ V).transpose(1, 2).contiguous().view(B, self.num_heads * self.head_dim)
        return self.norm(self.out_proj(out))


class SubjectEncoder(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int, hidden_dim: int = 256,
                 use_bridge: bool = False, num_heads: int = 4):
        super().__init__()
        self.use_bridge = use_bridge
        if use_bridge:
            self.bridge = BackboneAttentionBridge(input_dim, input_dim, hidden_dim, num_heads)
            in_dim = hidden_dim
        else:
            in_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x: torch.Tensor, token_reps: torch.Tensor = None) -> torch.Tensor:
        if self.use_bridge and token_reps is not None:
            x = self.bridge(x, token_reps)
        return self.net(x)


class ModifierNet(nn.Module):
    """
    Applies modifier to feature state, returning (updated_feature, delta).

    Gate controls which dimensions are touched.
    Delta is returned separately so directional change can be measured.
    """

    def __init__(self, modifier_dim: int, feature_dim: int, hidden_dim: int = 256,
                 use_bridge: bool = False, num_heads: int = 4):
        super().__init__()
        self.use_bridge    = use_bridge

        if use_bridge:
            self.bridge    = BackboneAttentionBridge(feature_dim, modifier_dim,
                                                     hidden_dim, num_heads)
            combined_dim   = hidden_dim + hidden_dim
        else:
            combined_dim   = hidden_dim

        self.modifier_proj = nn.Linear(modifier_dim, hidden_dim)
        self.feature_proj  = nn.Linear(feature_dim, hidden_dim)

        # Both paths converge to hidden_dim before delta_net/gate_net.
        # Bridge path: cat([attended, feature_proj]) = combined_dim → project down.
        # Fallback path: sum(modifier_proj, feature_proj) = hidden_dim already.
        self.norm_bridge    = nn.LayerNorm(combined_dim) if use_bridge else None
        self.bridge_proj    = nn.Linear(combined_dim, hidden_dim) if use_bridge else None
        self.norm_fallback  = nn.LayerNorm(hidden_dim)

        # delta_net and gate_net always receive hidden_dim regardless of path
        self.delta_net = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Tanh(),
        )
        self.gate_net = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, modifier_emb: torch.Tensor, current_feature: torch.Tensor,
                modifier_token_reps: torch.Tensor = None):
        if self.use_bridge and modifier_token_reps is not None:
            attended = self.bridge(current_feature, modifier_token_reps)
            cat_out  = torch.cat([attended, self.feature_proj(current_feature)], dim=-1)
            combined = self.bridge_proj(self.norm_bridge(cat_out))
        else:
            # Fallback (no token reps): pooled modifier embedding path
            combined = self.norm_fallback(
                self.modifier_proj(modifier_emb) + self.feature_proj(current_feature)
            )
        direction = self.delta_net(combined)
        gate      = self.gate_net(combined)
        delta     = gate * direction
        return current_feature + delta, delta


class ReasoningCortex(nn.Module):
    """Placeholder iterative processor. Near-identity until full cortex is added."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256, n_steps: int = 1):
        super().__init__()
        self.n_steps = n_steps
        self.step    = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor):
        for _ in range(self.n_steps):
            features = features + self.residual_scale.tanh() * self.step(features)
        return features, self.n_steps


class SemanticDecoder(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SubjectModifierCodec(nn.Module):
    def __init__(self, semantic_dim: int, feature_dim: int, hidden_dim: int = 256,
                 use_bridge: bool = False, num_heads: int = 4):
        super().__init__()
        self.subject_encoder = SubjectEncoder(semantic_dim, feature_dim, hidden_dim,
                                              use_bridge, num_heads)
        self.modifier_net    = ModifierNet(semantic_dim, feature_dim, hidden_dim,
                                           use_bridge, num_heads)
        self.cortex          = ReasoningCortex(feature_dim, hidden_dim)
        self.decoder         = SemanticDecoder(feature_dim, semantic_dim, hidden_dim)
        self.feature_dim     = feature_dim
        self.semantic_dim    = semantic_dim

    def encode(self, subject_emb, modifier_embs, subject_token_reps=None,
               modifier_token_reps_list=None):
        feature = self.subject_encoder(subject_emb, subject_token_reps)
        deltas  = []
        for i, mod_emb in enumerate(modifier_embs):
            mod_reps = modifier_token_reps_list[i] if modifier_token_reps_list else None
            feature, delta = self.modifier_net(mod_emb, feature, mod_reps)
            deltas.append(delta)
        return feature, deltas

    def reason(self, features):
        return self.cortex(features)

    def decode(self, features):
        return self.decoder(features)

    def forward(self, subject_emb, modifier_embs,
                subject_token_reps=None, modifier_token_reps_list=None):
        features, deltas = self.encode(subject_emb, modifier_embs,
                                        subject_token_reps, modifier_token_reps_list)
        cortex_out, _    = self.reason(features)
        reconstructed    = self.decode(cortex_out)
        return features, cortex_out, reconstructed, deltas


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def reconstruction_loss(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(reconstructed, original)
    cos = (1 - F.cosine_similarity(reconstructed, original, dim=-1)).mean()
    return mse + cos


def collapse_loss(features: torch.Tensor, min_var: float = 0.01) -> torch.Tensor:
    """
    Penalise feature dimensions with near-zero variance across the batch.
    A dead dimension carries no information regardless of input concept.
    """
    return F.relu(min_var - features.var(dim=0)).mean()


def feature_diversity_loss(
    features: torch.Tensor,
    input_embs: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """
    Semantically different inputs should produce different feature vectors.

    For random pairs in the batch:
      - if their input embeddings are dissimilar (cos_sim < 0.5), their
        features should differ by at least `margin` in cosine distance
      - if their inputs are similar, no penalty (similar inputs can map similarly)

    This replaces independence_loss. Independence_loss penalised global
    correlation structure and created enormous destabilising gradients.
    This loss is local (pairwise) and only fires on genuinely different concepts.
    """
    B = features.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=features.device)

    # Sample pairs — O(B) not O(B²)
    idx    = torch.randperm(B, device=features.device)
    f_a, f_b = features, features[idx]
    e_a, e_b = input_embs, input_embs[idx]

    feat_sim  = F.cosine_similarity(f_a, f_b, dim=-1)       # [B]
    input_sim = F.cosine_similarity(e_a, e_b, dim=-1)       # [B]

    # Only penalise pairs where inputs are clearly different
    different = (input_sim < 0.5).float()
    # Penalise when features are too similar despite different inputs
    loss = F.relu(feat_sim - (1.0 - margin)) * different
    return loss.mean()


def modifier_directional_loss(
    features_before: torch.Tensor,
    features_after: torch.Tensor,
    min_cos_change: float = 0.1,
) -> torch.Tensor:
    """
    Modifier must meaningfully rotate the feature vector, not just scale it.

    Measures cosine distance between pre- and post-modifier features.
    Scale-invariant: a large feature vector shifted by a tiny fraction fails
    just as much as a small one shifted by nothing.

    min_cos_change = 0.1 means the modifier must change the direction
    by at least cos⁻¹(0.9) ≈ 26° — a meaningful conceptual shift.
    """
    cos_change = 1 - F.cosine_similarity(features_before, features_after, dim=-1)
    return F.relu(min_cos_change - cos_change).mean()


def cross_transfer_loss(
    head_embs_a: torch.Tensor,
    head_embs_b: torch.Tensor,
    modifier_emb: torch.Tensor,
    model: "SubjectModifierCodec",
) -> torch.Tensor:
    """
    A modifier should shift features in a consistent direction regardless of subject.

    Apply the same modifier to two different heads. The delta directions
    (unit vectors) should be similar — modifier semantics shouldn't flip
    or become orthogonal depending on what they're applied to.

    'rough' on 'surface' and 'rough' on 'stone' should both activate
    similar attribute dimensions, even if the base features differ.
    """
    base_a = model.subject_encoder(head_embs_a)
    base_b = model.subject_encoder(head_embs_b)

    mod_emb_a = modifier_emb.expand(head_embs_a.shape[0], -1)
    mod_emb_b = modifier_emb.expand(head_embs_b.shape[0], -1)

    _, delta_a = model.modifier_net(mod_emb_a, base_a)
    _, delta_b = model.modifier_net(mod_emb_b, base_b)

    # Normalise deltas to unit vectors, compare directions
    d_a_norm = F.normalize(delta_a, dim=-1)
    d_b_norm = F.normalize(delta_b, dim=-1)

    # High cos sim between delta directions = consistent modifier semantics
    # We want this to be high, so loss = 1 - similarity
    dir_sim = F.cosine_similarity(d_a_norm, d_b_norm, dim=-1)
    return (1 - dir_sim).mean()


def feature_roundtrip_loss(features: torch.Tensor,
                           model: "SubjectModifierCodec") -> torch.Tensor:
    """features → decode → re-encode → should recover original features."""
    decoded    = model.decode(features)
    re_encoded = model.subject_encoder(decoded)
    return F.mse_loss(re_encoded, features.detach())


def modifier_consistency_loss(subject_emb, modifier_embs,
                               model: "SubjectModifierCodec") -> torch.Tensor:
    """round fuzzy ball == fuzzy round ball."""
    if len(modifier_embs) < 2:
        return torch.tensor(0.0, device=subject_emb.device)
    f_fwd, _ = model.encode(subject_emb, modifier_embs)
    f_rev, _ = model.encode(subject_emb, list(reversed(modifier_embs)))
    return F.mse_loss(f_fwd, f_rev)


def modifier_contrastive_loss(subject_embs, mod_a, mod_b,
                               model: "SubjectModifierCodec",
                               margin: float = 0.4) -> torch.Tensor:
    """Different modifiers → different feature outputs."""
    base      = model.subject_encoder(subject_embs)
    feat_a, _ = model.modifier_net(mod_a, base)
    feat_b, _ = model.modifier_net(mod_b, base)
    sim       = F.cosine_similarity(feat_a, feat_b, dim=-1)
    return F.relu(sim - (1.0 - margin)).mean()


def total_loss(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    features: torch.Tensor,
    features_before_mod: torch.Tensor,       # pre-modifier feature for directional check
    features_after_mod: torch.Tensor | None, # post-modifier feature (None if no modifiers)
    input_embs: torch.Tensor,                # original backbone embeddings for diversity
    subject_emb: torch.Tensor,
    modifier_embs: list,
    model: "SubjectModifierCodec",
    mod_embs_a: torch.Tensor = None,
    mod_embs_b: torch.Tensor = None,
    head_embs_alt: torch.Tensor = None,      # second set of heads for cross-transfer
    # weights
    collapse_weight:      float = 1.0,
    diversity_weight:     float = 1.0,
    directional_weight:   float = 2.0,
    contrastive_weight:   float = 2.0,
    transfer_weight:      float = 1.0,
    consistency_weight:   float = 0.1,
    roundtrip_weight:     float = 0.5,
    # thresholds
    contrastive_margin:    float = 0.4,
    min_cos_change:        float = 0.1,
    min_var:               float = 0.01,
    diversity_margin:      float = 0.3,
) -> dict:
    r   = reconstruction_loss(original, reconstructed)
    col = collapse_loss(features, min_var)
    div = feature_diversity_loss(features, input_embs, diversity_margin)
    rt  = feature_roundtrip_loss(features, model)
    c   = modifier_consistency_loss(subject_emb, modifier_embs, model)

    # Directional: modifier must actually change feature direction
    if features_after_mod is not None:
        dirn = modifier_directional_loss(features_before_mod, features_after_mod,
                                         min_cos_change)
    else:
        dirn = torch.tensor(0.0, device=original.device)

    # Contrastive: different modifiers → different outputs
    if mod_embs_a is not None and mod_embs_b is not None:
        ct = modifier_contrastive_loss(
            subject_emb.expand(mod_embs_a.shape[0], -1),
            mod_embs_a, mod_embs_b, model, contrastive_margin,
        )
    else:
        ct = torch.tensor(0.0, device=original.device)

    # Cross-transfer: modifier direction consistent across different subjects
    if head_embs_alt is not None and modifier_embs:
        mod_emb_single = modifier_embs[0]
        xfer = cross_transfer_loss(
            subject_emb.expand_as(head_embs_alt),
            head_embs_alt,
            mod_emb_single,
            model,
        )
    else:
        xfer = torch.tensor(0.0, device=original.device)

    total = (r
             + collapse_weight    * col
             + diversity_weight   * div
             + directional_weight * dirn
             + contrastive_weight * ct
             + transfer_weight    * xfer
             + consistency_weight * c
             + roundtrip_weight   * rt)

    return {
        "total": total, "reconstruction": r, "collapse": col,
        "diversity": div, "directional": dirn, "contrastive": ct,
        "transfer": xfer, "consistency": c, "roundtrip": rt,
    }
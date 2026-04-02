"""
Inference for the Subject+Modifier codec.

Tests are now expressed as (head, modifiers) pairs — the natural
granularity of the architecture.
"""

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from model import SubjectModifierCodec

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

checkpoint = torch.load("feature_codec.pt", map_location="cpu")
model = SubjectModifierCodec(
    semantic_dim=checkpoint["semantic_dim"],
    feature_dim=checkpoint["feature_dim"],
    hidden_dim=checkpoint["hidden_dim"],
)
model.load_state_dict(checkpoint["model_state"])
model.eval()

corpus      = checkpoint["corpus"]
corpus_embs = checkpoint["embeddings"]  # [N, semantic_dim]

backbone = SentenceTransformer("all-MiniLM-L6-v2")
backbone.eval()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(DEVICE)
corpus_embs = corpus_embs.to(DEVICE)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def embed(texts: list[str]) -> torch.Tensor:
    with torch.no_grad():
        return torch.tensor(
            backbone.encode(texts, convert_to_numpy=True),
            dtype=torch.float32,
            device=DEVICE,
        )

def encode(head: str, modifiers: list[str] = []) -> torch.Tensor:
    """Encode a (head noun, modifiers) pair to a feature vector."""
    subj_emb = embed([head])
    feature  = model.subject_encoder(subj_emb)
    for mod in modifiers:
        mod_emb = embed([mod])
        feature = model.modifier_net(mod_emb, feature)
    return feature

def nearest(embedding: torch.Tensor, n: int = 3) -> list[tuple[float, str]]:
    sims = F.cosine_similarity(embedding.view(1, -1), corpus_embs, dim=-1)
    top  = sims.topk(n)
    return [(sims[i].item(), corpus[i]) for i in top.indices]

def roundtrip(head: str, modifiers: list[str] = []):
    label = f"'{head}'" if not modifiers else f"'{head}' + {modifiers}"
    print(f"\nInput:  {label}")
    feature = encode(head, modifiers)
    recon   = model.decode(feature)
    print(f"Features (first 8): {feature[0, :8].tolist()}")
    print("Nearest matches:")
    for sim, text in nearest(recon):
        print(f"  {sim:.4f}  '{text}'")

def compare_modifiers(head: str, mod_sets: list[list[str]]):
    """
    Encode the same head noun with different modifier sets and compare.
    Shows how modifiers shift the feature vector and what they retrieve.
    """
    print(f"\nComparing modifiers on head: '{head}'")
    base_feat = encode(head)
    base_recon = model.decode(base_feat)

    for mods in mod_sets:
        feat  = encode(head, mods)
        recon = model.decode(feat)
        delta = (feat - base_feat).norm().item()
        best_sim, best_match = nearest(recon, n=1)[0]
        mod_label = ", ".join(mods) if mods else "(none)"
        print(f"  modifiers={mod_label:<35}  Δfeature={delta:.3f}  → '{best_match}' ({best_sim:.4f})")

def interpolate(head_a: str, mods_a: list[str], head_b: str, mods_b: list[str], steps: int = 5):
    """Interpolate between two encoded concepts in feature space."""
    label_a = f"'{head_a}' {mods_a}"
    label_b = f"'{head_b}' {mods_b}"
    print(f"\nInterpolating: {label_a} → {label_b}")
    f_a = encode(head_a, mods_a)
    f_b = encode(head_b, mods_b)
    for i in range(steps + 1):
        t = i / steps
        f_interp = (1 - t) * f_a + t * f_b
        recon = model.decode(f_interp)
        best_sim, best_match = nearest(recon, n=1)[0]
        print(f"  t={t:.2f}  →  '{best_match}'  ({best_sim:.4f})")

def modifier_order_test(head: str, modifiers: list[str]):
    """
    Check if modifier order changes the output — it shouldn't.
    The consistency loss trains against this, so delta should be small.
    """
    if len(modifiers) < 2:
        print("Need 2+ modifiers for order test.")
        return
    f_fwd = encode(head, modifiers)
    f_rev = encode(head, list(reversed(modifiers)))
    delta = (f_fwd - f_rev).norm().item()
    print(f"\nOrder consistency: '{head}' + {modifiers}")
    print(f"  Forward:  {nearest(model.decode(f_fwd), n=1)[0][1]}")
    print(f"  Reversed: {nearest(model.decode(f_rev), n=1)[0][1]}")
    print(f"  Feature delta: {delta:.4f}  (0.0 = perfectly order-independent)")

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

print("=" * 60)
print("ROUNDTRIP TESTS")
print("=" * 60)

roundtrip("object", ["round"])
roundtrip("object", ["sharp", "metallic"])
roundtrip("surface", ["rough"])
roundtrip("surface", ["smooth", "reflective"])
roundtrip("container", ["large"])
roundtrip("container", ["sealed", "transparent"])
roundtrip("structure", ["rigid"])
roundtrip("structure", ["flexible", "lightweight"])

print("\n" + "=" * 60)
print("MODIFIER DELTA TESTS")
print("=" * 60)

compare_modifiers("object", [
    [],
    ["large"],
    ["small"],
    ["heavy"],
    ["fragile"],
    ["large", "heavy"],
])

compare_modifiers("surface", [
    [],
    ["rough"],
    ["smooth"],
    ["wet"],
    ["sticky"],
    ["rough", "wet"],
])

print("\n" + "=" * 60)
print("INTERPOLATION TESTS")
print("=" * 60)

interpolate("object", ["small"], "object", ["large"])
interpolate("surface", ["rough"], "surface", ["smooth"])
interpolate("structure", ["rigid"], "structure", ["flexible"])

print("\n" + "=" * 60)
print("MODIFIER ORDER CONSISTENCY")
print("=" * 60)

modifier_order_test("object", ["large", "heavy"])
modifier_order_test("surface", ["rough", "wet"])
modifier_order_test("structure", ["flexible", "lightweight"])
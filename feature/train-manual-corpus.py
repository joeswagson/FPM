"""
Training script for the feature autoencoder.

This trains the encoder and decoder jointly on a corpus of sentences.
The backbone (sentence transformer) is frozen — we're not relearning
what language means, we're learning to compress meaning into a feature space
that respects the non-proportionality and sparsity constraints.
"""

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from model import FeatureAutoencoder, total_loss

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FEATURE_DIM = 64        # Size of the feature vector. Start small, scale up.
HIDDEN_DIM = 512
INDEPENDENCE_WEIGHT = 0.1
SPARSITY_WEIGHT = 0.05
LR = 1e-4
EPOCHS = 500
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BACKBONE_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast, good semantic quality

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
# This is your semantic grounding dataset. In a full system this would be
# much larger and more diverse — you want coverage of the concept space
# you care about, not just language patterns.

CORPUS = [
    # Shape / physical form
    "a round ball",
    "a round fuzzy ball",
    "a round short-haired fuzzy ball",
    "a perfectly spherical marble",
    "a slightly deflated ball",
    "a cube",
    "a flat disc",

    # Color and light
    "a red bucket",
    "a small red bucket",
    "a bright red car",
    "a dark blue sky",
    "the sky is blue",
    "a yellow sunflower",
    "a green leaf",
    "a white wall",
    "something glowing orange",

    # Material
    "a wooden table",
    "a glass bottle",
    "a metal rod",
    "a soft cotton blanket",
    "a rough stone surface",
    "a smooth ceramic bowl",

    # Size and scale
    "a tiny pebble",
    "a large boulder",
    "a massive ship",
    "a microscopic organism",

    # Temperature and state
    "a block of ice",
    "boiling water",
    "warm sunlight",
    "a cold metal surface",
    "steam rising from a cup",

    # Motion
    "a ball rolling down a hill",
    "a leaf falling slowly",
    "a car moving at high speed",
    "still water in a pond",

    # Composite / modified subjects
    "a soccer ball",
    "a deflated soccer ball",
    "an inflated basketball",
    "a worn leather football",
    "a transparent glass sphere",
    "a rough wooden cube",
    "a cold steel ball bearing",
]

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

print(f"Using device: {DEVICE}")
print(f"Loading backbone: {BACKBONE_MODEL}")

backbone = SentenceTransformer(BACKBONE_MODEL)
backbone.eval()

# Encode the full corpus once — backbone is frozen
with torch.no_grad():
    embeddings = torch.tensor(
        backbone.encode(CORPUS, show_progress_bar=False),
        dtype=torch.float32,
        device=DEVICE,
    )

SEMANTIC_DIM = embeddings.shape[1]
print(f"Semantic dim: {SEMANTIC_DIM}, Feature dim: {FEATURE_DIM}, Corpus size: {len(CORPUS)}")

model = FeatureAutoencoder(SEMANTIC_DIM, FEATURE_DIM, HIDDEN_DIM).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

print("\nTraining...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()

    # Shuffle and batch
    perm = torch.randperm(len(CORPUS), device=DEVICE)
    epoch_losses = {"total": 0, "reconstruction": 0, "independence": 0, "sparsity": 0}
    n_batches = 0

    for start in range(0, len(CORPUS), BATCH_SIZE):
        idx = perm[start:start + BATCH_SIZE]
        batch = embeddings[idx]

        features, reconstructed = model(batch)
        losses = total_loss(
            batch,
            reconstructed,
            features,
            independence_weight=INDEPENDENCE_WEIGHT,
            sparsity_weight=SPARSITY_WEIGHT,
        )

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k in epoch_losses:
            epoch_losses[k] += losses[k].item()
        n_batches += 1

    scheduler.step()

    if epoch % 50 == 0 or epoch == 1:
        avg = {k: v / n_batches for k, v in epoch_losses.items()}
        cos_sim = F.cosine_similarity(
            model(embeddings)[1], embeddings
        ).mean().item()
        print(
            f"Epoch {epoch:4d} | "
            f"total={avg['total']:.4f} | "
            f"recon={avg['reconstruction']:.4f} | "
            f"indep={avg['independence']:.4f} | "
            f"sparse={avg['sparsity']:.4f} | "
            f"cos_sim={cos_sim:.4f}"
        )

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

torch.save({
    "model_state": model.state_dict(),
    "semantic_dim": SEMANTIC_DIM,
    "feature_dim": FEATURE_DIM,
    "hidden_dim": HIDDEN_DIM,
    "corpus": CORPUS,
    "embeddings": embeddings.cpu(),
}, "feature_codec.pt")

print("\nSaved to feature_codec.pt")
print("Run inference.py to test encoding and decoding.")
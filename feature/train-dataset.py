"""
Training on noun chunks extracted from Bosio/pacman_descriptions.

Each training sample is a (head_noun, [modifiers], full_chunk) triple.
The model learns to:
  1. Encode the head noun to a base feature
  2. Apply each modifier sequentially as an attribute operation
  3. Decode the final feature back to the full chunk embedding

This is the correct granularity — features are subjects, not sentences.
"""

import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from model import SubjectModifierCodec, total_loss
from chunk_extractor import extract_chunks_batch, NounChunk

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# DATASET_NAME    = "Bosio/pacman_descriptions"
DATASET_NAMES = [
    # "Salesforce/wikitext",
    "agentlans/high-quality-english-sentences",
]
MAX_SENTENCES   = 10_000
MIN_CHUNK_LEN   = 3
MAX_MODIFIERS   = 8

FEATURE_DIM          = 128
HIDDEN_DIM           = 256
INDEPENDENCE_WEIGHT  = 0.1
SPARSITY_WEIGHT      = 0.05
CONSISTENCY_WEIGHT   = 0.1
LR                   = 1e-4
EPOCHS               = 200
BATCH_SIZE           = 256

BACKBONE_MODEL  = "all-MiniLM-L6-v2"
CACHE_FILE      = "chunk_embeddings_cache.pt"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Load dataset and extract noun chunks
# ---------------------------------------------------------------------------

print(f"Device: {DEVICE}")
# print(f"Loading {DATASET_NAME}...")
#
# ds = load_dataset(DATASET_NAME, split="train")
# text_col = next((c for c in ["text", "description", "sentence"] if c in ds.column_names), ds.column_names[0])
#
# sentences = []
# seen = set()
# for row in ds:
#     t = str(row[text_col]).strip()
#     if t not in seen and len(t) > 10:
#         seen.add(t)
#         sentences.append(t)
#     if len(sentences) >= MAX_SENTENCES:
#         break

print(f"Loading datasets: {DATASET_NAMES}...")

def extract_texts(ds):
    text_col = next(
        (c for c in ["text", "sentence", "content"] if c in ds.column_names),
        ds.column_names[0]
    )
    for row in ds:
        t = str(row[text_col]).strip()
        if len(t) > 10:
            yield t

sentences = []
seen = set()

for name in DATASET_NAMES:
    print(f"  → {name}")
    if "wikitext" in name.lower():
        ds = load_dataset(name, "wikitext-103-raw-v1", split="train")
    else:
        ds = load_dataset(name, split="train")
    # ds = load_dataset(name, split="train")

    # wikitext cleanup
    if "wikitext" in name.lower():
        ds = ds.filter(lambda x: x["text"].strip() != "" and not x["text"].startswith("="))

    for t in extract_texts(ds):
        if t not in seen:
            seen.add(t)
            sentences.append(t)

        if MAX_SENTENCES and len(sentences) >= MAX_SENTENCES:
            break



print(f"Sentences: {len(sentences)}")
print("Extracting noun chunks (spaCy)...")

raw_chunks = extract_chunks_batch(sentences, batch_size=512)

# chunks: list[NounChunk] = []
# for c in raw_chunks:
#     if len(c.head) < MIN_CHUNK_LEN:
#         continue
#     c.modifiers = c.modifiers[:MAX_MODIFIERS]
#     chunks.append(c)

filtered = []
for c in raw_chunks:
    if len(c.head) < MIN_CHUNK_LEN:
        continue

    # drop garbage modifiers (common in wikitext)
    mods = [m for m in c.modifiers if m.isalpha() and len(m) > 2]
    mods = mods[:MAX_MODIFIERS]

    c.modifiers = mods
    filtered.append(c)

chunks = filtered

print(f"Noun chunks extracted: {len(chunks)}")
print(f"  With modifiers:      {sum(1 for c in chunks if c.modifiers)}")
print(f"  Bare nouns:          {sum(1 for c in chunks if not c.modifiers)}")
print(f"  Avg modifiers:       {sum(len(c.modifiers) for c in chunks) / len(chunks):.2f}")

# ---------------------------------------------------------------------------
# Embed heads, modifiers, and full chunks (or load from cache)
# ---------------------------------------------------------------------------

if os.path.exists(CACHE_FILE):
    print(f"\nLoading cache: {CACHE_FILE}")
    cache = torch.load(CACHE_FILE, map_location="cpu")
    if cache.get("n_chunks") == len(chunks) and cache.get("model") == BACKBONE_MODEL:
        head_embs     = cache["head_embs"].to(DEVICE)
        chunk_embs    = cache["chunk_embs"].to(DEVICE)
        modifier_embs = [[m.to(DEVICE) for m in mods] for mods in cache["modifier_embs"]]
        print(f"Cache hit — {len(chunks)} chunks loaded")
    else:
        print("Cache mismatch — re-embedding")
        os.remove(CACHE_FILE)
        head_embs = None
else:
    head_embs = None

if head_embs is None:
    print(f"\nEmbedding with {BACKBONE_MODEL}...")
    backbone = SentenceTransformer(BACKBONE_MODEL)
    backbone.eval()

    def embed(texts):
        return torch.tensor(
            backbone.encode(texts, batch_size=512, show_progress_bar=True, convert_to_numpy=True),
            dtype=torch.float32,
        )

    head_embs  = embed([c.head for c in chunks]).to(DEVICE)
    chunk_embs = embed([c.full_text for c in chunks]).to(DEVICE)

    all_mod_tokens = sorted(set(m for c in chunks for m in c.modifiers))
    print(f"Unique modifier tokens: {len(all_mod_tokens)}")
    mod_token_embs = embed(all_mod_tokens) if all_mod_tokens else torch.zeros(0, head_embs.shape[1])
    mod_token_index = {tok: i for i, tok in enumerate(all_mod_tokens)}

    modifier_embs = []
    for c in chunks:
        mods = [mod_token_embs[mod_token_index[m]].to(DEVICE) for m in c.modifiers]
        modifier_embs.append(mods)

    torch.save({
        "head_embs":      head_embs.cpu(),
        "chunk_embs":     chunk_embs.cpu(),
        "modifier_embs":  [[m.cpu() for m in mods] for mods in modifier_embs],
        "n_chunks":       len(chunks),
        "model":          BACKBONE_MODEL,
    }, CACHE_FILE)
    print(f"Cached to {CACHE_FILE}")

SEMANTIC_DIM = head_embs.shape[1]
N = len(chunks)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = SubjectModifierCodec(SEMANTIC_DIM, FEATURE_DIM, HIDDEN_DIM).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nSemantic dim: {SEMANTIC_DIM} | Feature dim: {FEATURE_DIM} | Chunks: {N} | Params: {n_params:,}\n")

# ---------------------------------------------------------------------------
# Encode a batch given head embeddings and per-sample modifier lists
# ---------------------------------------------------------------------------

def encode_batch(model, subj_embs, batch_modifier_embs):
    feature = model.subject_encoder(subj_embs)
    max_mods = max((len(m) for m in batch_modifier_embs), default=0)
    for step in range(max_mods):
        has_mod = [j for j, m in enumerate(batch_modifier_embs) if step < len(m)]
        if not has_mod:
            break
        mod_batch = torch.stack([batch_modifier_embs[j][step] for j in has_mod])
        feat_batch = feature[has_mod]
        updated = model.modifier_net(mod_batch, feat_batch)
        feature = feature.clone()
        feature[has_mod] = updated
    return feature

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

print("Training...\n")

eval_idx   = list(range(min(512, N)))
eval_head  = head_embs[eval_idx]
eval_chunk = chunk_embs[eval_idx]
eval_mods  = [modifier_embs[i] for i in eval_idx]

for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(N)
    epoch_losses = {k: 0.0 for k in ["total", "reconstruction", "independence", "sparsity", "consistency"]}
    n_batches = 0

    for start in range(0, N, BATCH_SIZE):
        idx = perm[start:start + BATCH_SIZE].tolist()
        subj   = head_embs[idx]
        target = chunk_embs[idx]
        batch_mods = [modifier_embs[i] for i in idx]

        feature = encode_batch(model, subj, batch_mods)
        reconstructed = model.decode(feature)

        # Pick one multi-modifier sample for consistency loss
        multi = next((m for m in batch_mods if len(m) >= 2), [])
        subj_single = subj[:1]

        losses = total_loss(
            target, reconstructed, feature,
            subj_single, multi, model,
            independence_weight=INDEPENDENCE_WEIGHT,
            sparsity_weight=SPARSITY_WEIGHT,
            consistency_weight=CONSISTENCY_WEIGHT,
        )

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k in epoch_losses:
            epoch_losses[k] += losses[k].item()
        n_batches += 1

    scheduler.step()

    if epoch % 10 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            eval_feature = encode_batch(model, eval_head, eval_mods)
            eval_recon   = model.decode(eval_feature)
            cos_sim = F.cosine_similarity(eval_recon, eval_chunk).mean().item()
        model.train()

        avg = {k: v / n_batches for k, v in epoch_losses.items()}
        print(
            f"Epoch {epoch:4d} | "
            f"recon={avg['reconstruction']:.4f} | "
            f"indep={avg['independence']:.4f} | "
            f"sparse={avg['sparsity']:.4f} | "
            f"consist={avg['consistency']:.4f} | "
            f"cos_sim={cos_sim:.4f}"
        )

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

torch.save({
    "model_state":   model.state_dict(),
    "semantic_dim":  SEMANTIC_DIM,
    "feature_dim":   FEATURE_DIM,
    "hidden_dim":    HIDDEN_DIM,
    "corpus":        [c.full_text for c in chunks[:2000]],
    "embeddings":    chunk_embs[:2000].cpu(),
    "datasets":       DATASET_NAMES,
}, "feature_codec.pt")

print("\nSaved to feature_codec.pt — run inference.py to test.")
"""
Training script — Subject+Modifier codec
=========================================
Supports multi-million sentence corpora via streaming chunk extraction.
Sentences are never fully loaded into memory alongside their parsed chunks —
spaCy processes them in small windows and chunks are embedded in batches,
then accumulated into the final tensor.
"""

import os
import gc
import hashlib
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from model import SubjectModifierCodec, total_loss
from chunk_extractor import collect_chunks, NounChunk

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_NAMES = [
    # "Salesforce/wikitext",
    "agentlans/high-quality-english-sentences",
]

MAX_SENTENCES   = 10_000    # None = use all available
MAX_CHUNKS      = 500_000 # cap extracted noun chunks — None = no cap
SPACY_BATCH     = 1024     # sentences per spaCy window (tune down if OOM)
MIN_CHUNK_LEN   = 2
MAX_MODIFIERS   = 4

FEATURE_DIM          = 128
HIDDEN_DIM           = 512
INDEPENDENCE_WEIGHT  = 0.1
SPARSITY_WEIGHT      = 0.05
CONSISTENCY_WEIGHT   = 0.1
LR                   = 1e-4
EPOCHS               = 200
BATCH_SIZE           = 256
EMBED_BATCH          = 512   # backbone embedding batch size
LOG_EPOCH_INTERVAL   = 1

BACKBONE_MODEL = "all-MiniLM-L6-v2"
CACHE_FILE     = "chunk_embeddings_cache.pt"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Dataset loading — your extraction logic, kept intact
# ---------------------------------------------------------------------------

def extract_texts(ds):
    text_col = next(
        (c for c in ["text", "sentence", "content"] if c in ds.column_names),
        ds.column_names[0]
    )
    for row in ds:
        t = str(row[text_col]).strip()
        if len(t) > 10:
            yield t


print(f"Device: {DEVICE}")
print("Loading datasets...")

sentences = []
seen = set()

for name in DATASET_NAMES:
    print(f"  → {name}")
    if "wikitext" in name.lower():
        ds = load_dataset(name, "wikitext-103-raw-v1", split="train")
        ds = ds.filter(lambda x: x["text"].strip() != "" and
                                  not x["text"].startswith("="))
    else:
        ds = load_dataset(name, split="train")

    for t in extract_texts(ds):
        if t not in seen:
            seen.add(t)
            sentences.append(t)
        if MAX_SENTENCES and len(sentences) >= MAX_SENTENCES:
            break

del seen  # free the dedup set — can be large
gc.collect()

print(f"Sentences: {len(sentences):,}")

# ---------------------------------------------------------------------------
# Cache key — invalidate if sentences or config changed
# ---------------------------------------------------------------------------

# Hash the first+last 1000 sentences and the count as a cheap fingerprint.
# Recomputing a hash over 2.8M strings would take too long.
_sample = sentences[:500] + sentences[-500:]
_fingerprint = hashlib.md5(
    (str(len(sentences)) + "".join(_sample) + BACKBONE_MODEL).encode()
).hexdigest()[:12]

cache_valid = False
if os.path.exists(CACHE_FILE):
    meta = torch.load(CACHE_FILE, map_location="cpu", weights_only=False)
    cache_valid = meta.get("fingerprint") == _fingerprint
    if not cache_valid:
        print("Cache fingerprint mismatch — re-extracting")
        os.remove(CACHE_FILE)

# ---------------------------------------------------------------------------
# Chunk extraction — streaming, memory-safe
# ---------------------------------------------------------------------------

if not cache_valid:
    print(f"\nExtracting noun chunks (spaCy batch={SPACY_BATCH})...")
    chunks: list[NounChunk] = collect_chunks(
        sentences,
        max_chunks=MAX_CHUNKS,
        spacy_batch=SPACY_BATCH,
        max_modifiers=MAX_MODIFIERS,
        min_head_len=MIN_CHUNK_LEN,
        show_progress=True,
        total=len(sentences),
    )

    print(f"\nChunks extracted:    {len(chunks):,}")
    print(f"  With modifiers:    {sum(1 for c in chunks if c.modifiers):,}")
    print(f"  Bare nouns:        {sum(1 for c in chunks if not c.modifiers):,}")
    print(f"  Avg modifiers:     {sum(len(c.modifiers) for c in chunks) / max(len(chunks),1):.2f}")

    # Free sentences — no longer needed
    del sentences
    gc.collect()

    # ---------------------------------------------------------------------------
    # Embed heads, modifiers, full chunks
    # ---------------------------------------------------------------------------

    print(f"\nEmbedding with {BACKBONE_MODEL}...")
    backbone = SentenceTransformer(BACKBONE_MODEL)
    backbone.eval()

    def embed(texts: list[str]) -> torch.Tensor:
        return torch.tensor(
            backbone.encode(
                texts,
                batch_size=EMBED_BATCH,
                show_progress_bar=True,
                convert_to_numpy=True,
            ),
            dtype=torch.float32,
        )

    print("  Embedding heads...")
    head_embs = embed([c.head for c in chunks])

    print("  Embedding full chunks...")
    chunk_embs = embed([c.full_text for c in chunks])

    # Embed unique modifier tokens only — avoids re-embedding duplicates
    all_mod_tokens = sorted(set(m for c in chunks for m in c.modifiers))
    print(f"  Embedding {len(all_mod_tokens):,} unique modifier tokens...")
    if all_mod_tokens:
        mod_token_embs = embed(all_mod_tokens)
    else:
        mod_token_embs = torch.zeros(0, head_embs.shape[1])
    mod_index = {tok: i for i, tok in enumerate(all_mod_tokens)}

    modifier_embs = [
        [mod_token_embs[mod_index[m]] for m in c.modifiers]
        for c in chunks
    ]

    # Free backbone — done embedding
    del backbone
    gc.collect()

    print(f"Saving cache to {CACHE_FILE}...")
    torch.save({
        "head_embs":      head_embs,
        "chunk_embs":     chunk_embs,
        "modifier_embs":  modifier_embs,
        "corpus":         [c.full_text for c in chunks[:2000]],
        "n_chunks":       len(chunks),
        "fingerprint":    _fingerprint,
        "model":          BACKBONE_MODEL,
    }, CACHE_FILE)
    print("Cached.")

else:
    print(f"\nLoading cache: {CACHE_FILE}")
    del sentences
    gc.collect()

    cache = torch.load(CACHE_FILE, map_location="cpu", weights_only=False)
    head_embs     = cache["head_embs"]
    chunk_embs    = cache["chunk_embs"]
    modifier_embs = cache["modifier_embs"]
    print(f"Cache hit — {len(head_embs):,} chunks loaded")

# Move to device
head_embs  = head_embs.to(DEVICE)
chunk_embs = chunk_embs.to(DEVICE)
modifier_embs = [[m.to(DEVICE) for m in mods] for mods in modifier_embs]

SEMANTIC_DIM = head_embs.shape[1]
N = len(head_embs)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = SubjectModifierCodec(SEMANTIC_DIM, FEATURE_DIM, HIDDEN_DIM).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nSemantic dim: {SEMANTIC_DIM} | Feature dim: {FEATURE_DIM} | "
      f"Chunks: {N:,} | Params: {n_params:,}\n")

# ---------------------------------------------------------------------------
# Encode a batch — applies modifiers sequentially, handles ragged lists
# ---------------------------------------------------------------------------

def encode_batch(subj_embs: torch.Tensor, batch_modifier_embs: list) -> torch.Tensor:
    feature  = model.subject_encoder(subj_embs)
    max_mods = max((len(m) for m in batch_modifier_embs), default=0)
    for step in range(max_mods):
        has_mod = [j for j, m in enumerate(batch_modifier_embs) if step < len(m)]
        if not has_mod:
            break
        mod_batch  = torch.stack([batch_modifier_embs[j][step] for j in has_mod])
        feat_batch = feature[has_mod]
        updated    = model.modifier_net(mod_batch, feat_batch)
        feature    = feature.clone()
        feature[has_mod] = updated
    return feature

# ---------------------------------------------------------------------------
# Eval subset — fixed 512 samples for stable per-epoch cos_sim tracking
# ---------------------------------------------------------------------------

eval_idx   = list(range(min(512, N)))
eval_head  = head_embs[eval_idx]
eval_chunk = chunk_embs[eval_idx]
eval_mods  = [modifier_embs[i] for i in eval_idx]

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

print("Training...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(N)
    epoch_losses = {k: 0.0 for k in
                    ["total", "reconstruction", "independence", "sparsity", "consistency"]}
    n_batches = 0

    for start in range(0, N, BATCH_SIZE):
        idx        = perm[start:start + BATCH_SIZE].tolist()
        subj       = head_embs[idx]
        target     = chunk_embs[idx]
        batch_mods = [modifier_embs[i] for i in idx]

        feature       = encode_batch(subj, batch_mods)
        reconstructed = model.decode(feature)

        # Consistency loss: one multi-modifier sample per batch
        multi       = next((m for m in batch_mods if len(m) >= 2), [])
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

    if epoch % LOG_EPOCH_INTERVAL == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            eval_feature = encode_batch(eval_head, eval_mods)
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

cache_data = torch.load(CACHE_FILE, map_location="cpu", weights_only=False)

torch.save({
    "model_state":  model.state_dict(),
    "semantic_dim": SEMANTIC_DIM,
    "feature_dim":  FEATURE_DIM,
    "hidden_dim":   HIDDEN_DIM,
    "corpus":       cache_data.get("corpus", []),
    "embeddings":   chunk_embs[:2000].cpu(),
    "dataset":      DATASET_NAMES,
}, "feature_codec.pt")

print("\nSaved to feature_codec.pt — run inference.py to test.")
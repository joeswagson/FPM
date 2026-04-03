"""
Inference — uses validation sentences from Flickr30k for realistic testing.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from model import SubjectModifierCodec

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------

ckpt = torch.load("feature_codec.pt", map_location="cpu", weights_only=False)

model = SubjectModifierCodec(
    semantic_dim=ckpt["semantic_dim"],
    feature_dim=ckpt["feature_dim"],
    hidden_dim=ckpt["hidden_dim"],
    use_bridge=ckpt.get("use_bridge", False),
    num_heads=ckpt.get("num_heads", 4),
).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

corpus_texts = ckpt["corpus"]                       # training chunk texts
corpus_embs  = ckpt["embeddings"].to(DEVICE)        # [N, semantic_dim]
val_sentences = ckpt.get("val_sentences", [])
val_embs      = ckpt.get("val_embs")
if val_embs is not None:
    val_embs = val_embs.to(DEVICE)

BACKBONE_MODEL = ckpt.get("backbone_model", "microsoft/harrier-oss-v1-0.6b")
print(f"Loading backbone: {BACKBONE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
backbone  = AutoModel.from_pretrained(BACKBONE_MODEL).to(DEVICE)
backbone.eval()

SEMANTIC_DIM = ckpt["semantic_dim"]

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def mean_pool(token_out, attention_mask):
    mask   = attention_mask.unsqueeze(-1).float()
    summed = (token_out * mask).sum(dim=1)
    return summed / mask.sum(dim=1).clamp(min=1e-9)

def embed(texts: list[str], max_len: int = 64) -> torch.Tensor:
    enc = tokenizer(texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_len)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = backbone(**enc)
    return mean_pool(out.last_hidden_state, enc["attention_mask"]).float()

def embed_tokens(texts: list[str], max_len: int = 16) -> torch.Tensor:
    enc = tokenizer(texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_len)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = backbone(**enc)
    return out.last_hidden_state.float()

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode(head: str, modifiers: list[str] = []) -> torch.Tensor:
    subj    = embed([head])
    feature = model.subject_encoder(subj)
    for mod in modifiers:
        mod_emb  = embed([mod])
        tok_reps = embed_tokens([mod]) if ckpt.get("use_bridge") else None
        feature, _ = model.modifier_net(mod_emb, feature, tok_reps)
    return feature

@torch.no_grad()
def nearest(emb: torch.Tensor, pool: torch.Tensor, texts: list, n: int = 3):
    sims = F.cosine_similarity(emb.view(1, -1), pool, dim=-1)
    top  = sims.topk(n)
    return [(sims[i].item(), texts[i]) for i in top.indices]

def roundtrip(head: str, modifiers: list[str] = [], ref_pool=corpus_embs, ref_texts=corpus_texts):
    label = f"'{head}'" if not modifiers else f"'{head}' + {modifiers}"
    print(f"\nInput:  {label}")
    feat    = encode(head, modifiers)
    co, _   = model.reason(feat)
    decoded = model.decode(co)
    for sim, txt in nearest(decoded, ref_pool, ref_texts):
        print(f"  {sim:.4f}  '{txt}'")

def compare(head: str, mod_sets: list[list[str]]):
    print(f"\nModifier comparison — head: '{head}'")
    base = encode(head)
    for mods in mod_sets:
        feat    = encode(head, mods)
        delta   = (feat - base).norm().item()
        cos_chg = (1 - F.cosine_similarity(feat, base).item())
        co, _   = model.reason(feat)
        dec     = model.decode(co)
        best    = nearest(dec, corpus_embs, corpus_texts, n=1)[0]
        label   = mods if mods else ["(none)"]
        print(f"  {str(label):<35}  Δcos={cos_chg:.3f}  → '{best[1]}' ({best[0]:.3f})")

def interpolate(head_a, mods_a, head_b, mods_b, steps=5):
    print(f"\nInterpolating: '{head_a}'{mods_a} → '{head_b}'{mods_b}")
    f_a = encode(head_a, mods_a)
    f_b = encode(head_b, mods_b)
    for i in range(steps + 1):
        t       = i / steps
        f_i     = (1 - t) * f_a + t * f_b
        co, _   = model.reason(f_i)
        dec     = model.decode(co)
        sim, tx = nearest(dec, corpus_embs, corpus_texts, n=1)[0]
        print(f"  t={t:.1f}  → '{tx}' ({sim:.3f})")

# ---------------------------------------------------------------------------
# Validation set tests
# ---------------------------------------------------------------------------

@torch.no_grad()
def val_roundtrip(n: int = 10):
    """
    Encode val sentence embeddings as bare subjects, decode, find nearest
    in the val set itself — checks that the feature space generalises
    to unseen sentences, not just training chunks.
    """
    if val_embs is None or len(val_sentences) == 0:
        print("No validation data in checkpoint.")
        return
    print(f"\n{'='*60}")
    print(f"VALIDATION ROUNDTRIP (unseen sentences)")
    print(f"{'='*60}")
    for i in range(min(n, len(val_sentences))):
        emb  = val_embs[i:i+1]
        feat = model.subject_encoder(emb)
        co, _ = model.reason(feat)
        dec  = model.decode(co)
        # Find nearest in full val set
        top  = nearest(dec, val_embs, val_sentences, n=3)
        print(f"\nInput: '{val_sentences[i][:80]}'")
        for sim, tx in top:
            print(f"  {sim:.4f}  '{tx[:80]}'")

@torch.no_grad()
def val_modifier_transfer(n_pairs: int = 5):
    """
    Sample random val sentences, extract a modifier from one,
    apply it to a head from another, verify the result is coherent.
    Shows cross-concept transfer on unseen data.
    """
    if not val_sentences:
        return
    from chunk_extractor import stream_chunks
    print(f"\n{'='*60}")
    print(f"VALIDATION MODIFIER TRANSFER")
    print(f"{'='*60}")

    parsed = []
    for chunk in stream_chunks(val_sentences[:200], spacy_batch=64, show_progress=False):
        parsed.append((chunk.source, chunk))

    import random
    rng = random.Random(0)
    # Group chunks by source sentence
    by_source = {}
    for source, chunk in parsed:
        by_source.setdefault(source, []).append(chunk)
    sources = list(by_source.keys())

    count = 0
    for _ in range(n_pairs * 10):
        if count >= n_pairs or len(sources) < 2:
            break
        src_key = rng.choice(sources)
        tgt_key = rng.choice(sources)
        chunks_with_mods = [c for c in by_source[src_key] if c.modifiers]
        tgt_chunks = by_source[tgt_key]
        if not chunks_with_mods or not tgt_chunks:
            continue
        src_chunk = rng.choice(chunks_with_mods)
        tgt_chunk = rng.choice(tgt_chunks)
        mod = rng.choice(src_chunk.modifiers)
        print(f"\nModifier '{mod}' (from '{src_chunk.full_text}') applied to '{tgt_chunk.head}':")
        compare(tgt_chunk.head, [[], [mod]])
        count += 1

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("ROUNDTRIP TESTS (training domain)")
print(f"{'='*60}")

roundtrip("dog", ["large"])
roundtrip("dog", ["small", "white"])
roundtrip("woman", ["young"])
roundtrip("man", ["tall", "bearded"])
roundtrip("field", ["green"])
roundtrip("sky", ["cloudy"])

print(f"\n{'='*60}")
print("MODIFIER COMPARISON")
print(f"{'='*60}")

compare("dog", [[], ["large"], ["small"], ["white"], ["black"], ["large", "white"]])
compare("person", [[], ["young"], ["old"], ["tall"], ["running"]])
compare("sky", [[], ["blue"], ["cloudy"], ["dark"], ["clear"]])

print(f"\n{'='*60}")
print("INTERPOLATION")
print(f"{'='*60}")

interpolate("dog", ["large"], "dog", ["small"])
interpolate("sky", ["blue"], "sky", ["cloudy"])
interpolate("woman", ["young"], "man", ["old"])

val_roundtrip(n=8)
val_modifier_transfer(n_pairs=4)
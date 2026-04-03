def main():
    """
    Training — Flickr30k captions, harrier backbone.
    """

    import os
    import gc
    import hashlib
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModel

    from model import SubjectModifierCodec, total_loss
    from chunk_extractor import collect_chunks, NounChunk

    # ---------------------------------------------------------------------------
    # Config
    # ---------------------------------------------------------------------------

    N_PROCESS = max(1, os.cpu_count() - 1)

    DATASET_NAMES = [
        "AnyModal/flickr30k",
    ]

    MAX_SENTENCES     = 25_000        # None = use all. Cap train sentences before split.
    MAX_VAL_SENTENCES = 2_000     # hard cap on validation set size
    MAX_CHUNKS      = 500_000
    SPACY_BATCH     = 1024
    MIN_CHUNK_LEN   = 2
    MAX_MODIFIERS   = 4

    USE_BRIDGE      = True
    BRIDGE_HEADS    = 8
    FEATURE_DIM     = 256
    HIDDEN_DIM      = 1024

    # Independence removed — replaced by diversity + directional
    # COLLAPSE_WEIGHT      = 1.0
    # DIVERSITY_WEIGHT     = 1.0
    # DIRECTIONAL_WEIGHT   = 2.0
    # CONTRASTIVE_WEIGHT   = 2.0
    # TRANSFER_WEIGHT      = 1.0
    # CONSISTENCY_WEIGHT   = 0.1
    # ROUNDTRIP_WEIGHT     = 0.5
    # CONTRASTIVE_MARGIN   = 0.4
    # MIN_COS_CHANGE       = 0.1
    # MIN_VAR              = 0.01
    # DIVERSITY_MARGIN     = 0.3
    # CONTRASTIVE_PAIRS    = 16
    COLLAPSE_WEIGHT       = 0.5
    DIVERSITY_WEIGHT      = 0.4
    DIRECTIONAL_WEIGHT    = 2.5
    CONTRASTIVE_WEIGHT    = 1.5
    TRANSFER_WEIGHT       = 2.0
    CONSISTENCY_WEIGHT    = 0.2
    ROUNDTRIP_WEIGHT      = 5.5
    CONTRASTIVE_MARGIN    = 0.1
    MIN_COS_CHANGE        = 0.05
    MIN_VAR               = 0.015
    DIVERSITY_MARGIN      = 0.2
    CONTRASTIVE_PAIRS     = 32

    LR                 = 1e-4
    EPOCHS             = 200
    BATCH_SIZE         = 256
    EMBED_BATCH        = 128
    LOG_EPOCH_INTERVAL = 1

    BACKBONE_MODEL = "microsoft/harrier-oss-v1-0.6b"
    CACHE_FILE     = "chunk_embeddings_cache.pt"
    DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------------------------
    # Backbone: harrier via AutoModel with mean pooling
    # ---------------------------------------------------------------------------

    print(f"Device: {DEVICE}")
    print(f"Loading backbone tokenizer: {BACKBONE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def mean_pool(token_out, attention_mask):
        """Mean pool over non-padding tokens."""
        mask   = attention_mask.unsqueeze(-1).float()
        summed = (token_out * mask).sum(dim=1)
        count  = mask.sum(dim=1).clamp(min=1e-9)
        return summed / count

    def embed_texts_pooled(backbone, texts: list, max_len: int = 64) -> torch.Tensor:
        """Return mean-pooled embeddings [N, hidden_dim]."""
        all_embs = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i+EMBED_BATCH]
            enc   = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_len)
            enc   = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                out = backbone(**enc)
            emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
            all_embs.append(emb.cpu().float())
        return torch.cat(all_embs, dim=0)

    def embed_texts_tokens(backbone, texts: list, max_len: int = 16) -> torch.Tensor:
        """Return token-level representations [N, seq, hidden_dim]."""
        all_reps = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i+EMBED_BATCH]
            enc   = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_len)
            enc   = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                out = backbone(**enc)
            all_reps.append(out.last_hidden_state.cpu().float())
        # Pad to same seq length across batches
        max_s = max(r.shape[1] for r in all_reps)
        padded = []
        for r in all_reps:
            pad = max_s - r.shape[1]
            if pad > 0:
                r = F.pad(r, (0, 0, 0, pad))
            padded.append(r)
        return torch.cat(padded, dim=0)

    # ---------------------------------------------------------------------------
    # Dataset loading — Flickr30k
    # ---------------------------------------------------------------------------

    def extract_texts_flickr(ds) -> list[str]:
        texts = []
        seen  = set()

        col = next((c for c in ["original_alt_text", "alt_text", "text", "content"]
                    if c in ds.column_names), ds.column_names[0])
        print(f"  Caption column: '{col}'")
        for row in ds:
            val = row[col]
            # Some datasets nest captions as lists
            if isinstance(val, list):
                items = val
            else:
                items = [val]
            for t in items:
                t = str(t).strip()
                if len(t) > 10 and t not in seen:
                    seen.add(t)
                    texts.append(t)
        return texts

    print("Loading Flickr30k...")
    sentences = []
    for name in DATASET_NAMES:
        print(f"  → {name}")
        try:
            ds = load_dataset(name, split="train")
        except Exception:
            ds = load_dataset(name, split="train", trust_remote_code=True)
        sentences.extend(extract_texts_flickr(ds))

    # Apply MAX_SENTENCES cap before split
    if MAX_SENTENCES and len(sentences) > MAX_SENTENCES:
        sentences = sentences[:MAX_SENTENCES]
    print(f"Sentences (after cap): {len(sentences):,}")

    # ---------------------------------------------------------------------------
    # Train / validation split
    # ---------------------------------------------------------------------------

    import random
    random.seed(42)
    random.shuffle(sentences)
    val_size        = min(MAX_VAL_SENTENCES, max(1, int(len(sentences) * 0.05)))
    val_sentences   = sentences[:val_size]
    train_sentences = sentences[val_size:]
    print(f"Train: {len(train_sentences):,} | Val: {len(val_sentences):,}")

    # ---------------------------------------------------------------------------
    # Cache
    # ---------------------------------------------------------------------------

    _sample      = train_sentences[:500] + train_sentences[-500:]
    _bridge_flag = "bridge" if USE_BRIDGE else "nobridge"
    _fingerprint = hashlib.md5(
        (str(len(train_sentences)) + "".join(_sample) + BACKBONE_MODEL + _bridge_flag).encode()
    ).hexdigest()[:12]

    cache_valid = False
    if os.path.exists(CACHE_FILE):
        meta = torch.load(CACHE_FILE, map_location="cpu", weights_only=False)
        cache_valid = meta.get("fingerprint") == _fingerprint
        if not cache_valid:
            print("Cache mismatch — rebuilding")
            os.remove(CACHE_FILE)

    # ---------------------------------------------------------------------------
    # Chunk extraction + embedding
    # ---------------------------------------------------------------------------

    if not cache_valid:
        print(f"\nExtracting noun chunks...")
        chunks: list[NounChunk] = collect_chunks(
            train_sentences,
            max_chunks=MAX_CHUNKS,
            spacy_batch=SPACY_BATCH,
            max_modifiers=MAX_MODIFIERS,
            min_head_len=MIN_CHUNK_LEN,
            show_progress=True,
            total=len(train_sentences),
            # n_process=N_PROCESS,       
        )
        print(f"Chunks: {len(chunks):,}  (with modifiers: {sum(1 for c in chunks if c.modifiers):,})")
        del train_sentences
        gc.collect()

        print(f"\nLoading backbone: {BACKBONE_MODEL}")
        backbone = AutoModel.from_pretrained(BACKBONE_MODEL).to(DEVICE)
        backbone.eval()

        BACKBONE_DIM = backbone.config.hidden_size
        print(f"Backbone hidden dim: {BACKBONE_DIM}")

        print("  Embedding heads (pooled)...")
        head_embs  = embed_texts_pooled(backbone, [c.head for c in chunks])
        print("  Embedding chunks (pooled)...")
        chunk_embs = embed_texts_pooled(backbone, [c.full_text for c in chunks])

        all_mod_tokens = sorted(set(m for c in chunks for m in c.modifiers))
        print(f"  Embedding {len(all_mod_tokens):,} modifier tokens (pooled)...")
        mod_pool_embs = embed_texts_pooled(backbone, all_mod_tokens) \
                        if all_mod_tokens else torch.zeros(0, BACKBONE_DIM)
        mod_pool_idx  = {tok: i for i, tok in enumerate(all_mod_tokens)}

        modifier_embs = [
            [mod_pool_embs[mod_pool_idx[m]] for m in c.modifiers]
            for c in chunks
        ]

        if USE_BRIDGE:
            print(f"  Embedding {len(all_mod_tokens):,} modifier tokens (token-level)...")
            mod_tok_reps = embed_texts_tokens(backbone, all_mod_tokens, max_len=8) \
                        if all_mod_tokens else torch.zeros(0, 1, BACKBONE_DIM)
        else:
            mod_tok_reps = None

        # Embed validation sentences for inference
        print("  Embedding validation set...")
        val_embs = embed_texts_pooled(backbone, val_sentences)

        del backbone
        gc.collect()

        torch.save({
            "head_embs":      head_embs,
            "chunk_embs":     chunk_embs,
            "modifier_embs":  modifier_embs,
            "mod_pool_embs":  mod_pool_embs,
            "mod_tok_reps":   mod_tok_reps,
            "mod_pool_idx":   mod_pool_idx,
            "val_sentences":  val_sentences,
            "val_embs":       val_embs,
            "corpus":         [c.full_text for c in chunks[:2000]],
            "n_chunks":       len(chunks),
            "fingerprint":    _fingerprint,
            "model":          BACKBONE_MODEL,
            "use_bridge":     USE_BRIDGE,
            "backbone_dim":   BACKBONE_DIM,
        }, CACHE_FILE)
        print("Cached.")

    else:
        print(f"\nLoading cache...")
        del train_sentences
        gc.collect()
        cache         = torch.load(CACHE_FILE, map_location="cpu", weights_only=False)
        head_embs     = cache["head_embs"]
        chunk_embs    = cache["chunk_embs"]
        modifier_embs = cache["modifier_embs"]
        mod_pool_embs = cache.get("mod_pool_embs")
        mod_tok_reps  = cache.get("mod_tok_reps")
        mod_pool_idx  = cache.get("mod_pool_idx", {})
        val_sentences = cache.get("val_sentences", [])
        val_embs      = cache.get("val_embs")
        BACKBONE_DIM  = cache.get("backbone_dim", head_embs.shape[1])
        print(f"Cache hit — {len(head_embs):,} chunks | bridge={cache.get('use_bridge')}")

    # Move tensors to device
    head_embs     = head_embs.to(DEVICE)
    chunk_embs    = chunk_embs.to(DEVICE)
    modifier_embs = [[m.to(DEVICE) for m in mods] for mods in modifier_embs]
    if mod_pool_embs is not None and len(mod_pool_embs) > 0:
        mod_pool_embs = mod_pool_embs.to(DEVICE)
    if mod_tok_reps is not None and len(mod_tok_reps) > 0:
        mod_tok_reps = mod_tok_reps.to(DEVICE)
    if val_embs is not None:
        val_embs = val_embs.to(DEVICE)

    SEMANTIC_DIM = head_embs.shape[1]
    N            = len(head_embs)

    # ---------------------------------------------------------------------------
    # Modifier index lookup for bridge
    # ---------------------------------------------------------------------------

    if USE_BRIDGE and mod_tok_reps is not None and len(mod_pool_embs) > 0:
        print("Building modifier index lookup...")
        mod_pool_cpu = mod_pool_embs.cpu()
        modifier_indices = []
        for mods in modifier_embs:
            indices = []
            for m_emb in mods:
                sims = F.cosine_similarity(m_emb.cpu().unsqueeze(0), mod_pool_cpu, dim=-1)
                indices.append(sims.argmax().item())
            modifier_indices.append(indices)
        print("Done.")
    else:
        modifier_indices = [[] for _ in range(N)]

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------

    model = SubjectModifierCodec(
        semantic_dim=SEMANTIC_DIM,
        feature_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        use_bridge=USE_BRIDGE,
        num_heads=BRIDGE_HEADS,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nSemantic: {SEMANTIC_DIM} | Feature: {FEATURE_DIM} | Hidden: {HIDDEN_DIM} | "
        f"Params: {n_params:,} | Bridge: {USE_BRIDGE}\n")

    # ---------------------------------------------------------------------------
    # Encode batch
    # ---------------------------------------------------------------------------

    def encode_batch(subj_embs, batch_mods, batch_midx=None):
        feature    = model.subject_encoder(subj_embs)
        base_feat  = feature.detach().clone()  # pre-modifier snapshot
        all_deltas = []
        first_post = None
        max_mods   = max((len(m) for m in batch_mods), default=0)

        for step in range(max_mods):
            has = [j for j, m in enumerate(batch_mods) if step < len(m)]
            if not has:
                break
            mod_b    = torch.stack([batch_mods[j][step] for j in has])
            feat_b   = feature[has]
            tok_reps = None
            if USE_BRIDGE and mod_tok_reps is not None and batch_midx:
                idxs = [batch_midx[j][step] for j in has if step < len(batch_midx[j])]
                if len(idxs) == len(has):
                    tok_reps = mod_tok_reps[idxs]
            updated, delta = model.modifier_net(mod_b, feat_b, tok_reps)
            feature        = feature.clone()
            feature[has]   = updated
            all_deltas.append(delta)
            if first_post is None:
                first_post = feature.clone()

        return feature, base_feat, first_post, all_deltas

    # ---------------------------------------------------------------------------
    # Contrastive + transfer sampling
    # ---------------------------------------------------------------------------

    def sample_contrastive_pairs(n):
        if mod_pool_embs is None or len(mod_pool_embs) < 2:
            return None, None
        ia = torch.randint(len(mod_pool_embs), (n,), device=DEVICE)
        ib = torch.randint(len(mod_pool_embs), (n,), device=DEVICE)
        same = ia == ib
        if same.any():
            ib[same] = (ib[same] + 1) % len(mod_pool_embs)
        return mod_pool_embs[ia], mod_pool_embs[ib]

    # ---------------------------------------------------------------------------
    # Validation metric — nearest-neighbour recall on val set
    # ---------------------------------------------------------------------------

    @torch.no_grad()
    def val_recall(k: int = 5) -> float:
        """
        For each val sentence embedding, encode it as a bare noun chunk,
        decode, and check if the nearest k corpus chunks include something
        semantically relevant. Quick proxy: cos-sim of decoded embedding
        to the original val embedding.
        """
        if val_embs is None or len(val_embs) == 0:
            return 0.0
        # Just measure decode quality: encode val embs as bare subjects, decode, compare
        n   = min(256, len(val_embs))
        emb = val_embs[:n]
        # Treat val embeddings as subject embeddings directly
        feat = model.subject_encoder(emb)
        co, _ = model.reason(feat)
        dec  = model.decode(co)
        return F.cosine_similarity(dec, emb, dim=-1).mean().item()

    # ---------------------------------------------------------------------------
    # Frequency-balanced sampler — solves subject/modifier skew
    # ---------------------------------------------------------------------------
    # Flickr30k is heavily skewed: "person", "dog", "man", "woman" dominate.
    # Without reweighting, the model overtunes to common concepts and rare ones
    # (bicycle, instrument, building) get almost no gradient signal.
    #
    # Strategy: compute per-head-noun sampling weight = 1 / sqrt(frequency).
    # sqrt rather than 1/freq prevents extreme upweighting of single-occurrence heads.
    # The contrastive sampler uses inverse-frequency weights for modifier pairs
    # so rare modifiers ("corrugated", "iridescent") get more contrastive signal.

    from collections import Counter
    import math

    # Count head noun occurrences across all chunks
    head_texts  = [head_embs[i] for i in range(N)]  # tensors, use index map
    # We need the string heads — rebuild from cache
    _cache      = torch.load(CACHE_FILE, map_location="cpu", weights_only=False)
    # The cache stores full_text for corpus, but we need head strings.
    # Re-derive from modifier_embs structure: heads are embedded at head_embs[i].
    # Since we don't store head strings in cache, approximate by clustering:
    # instead, build weights from modifier count as a proxy for specificity.
    # Chunks with 0 modifiers are generic (likely high-frequency) — downweight them.
    # Chunks with modifiers are more specific — upweight them.

    mod_counts      = torch.tensor([len(modifier_embs[i]) for i in range(N)], dtype=torch.float)
    # Weight: more modifiers = rarer/more specific = higher sampling weight
    # Base weight 1.0 for bare nouns, up to 2.0 for 4-modifier chunks
    sample_weights  = 1.0 + mod_counts / MAX_MODIFIERS
    sample_weights  = sample_weights / sample_weights.sum()   # normalise to probability

    # Inverse-frequency weights for modifier contrastive sampling
    # Count how many chunks each modifier token appears in
    if mod_pool_embs is not None and len(mod_pool_embs) > 0:
        mod_freq = torch.zeros(len(mod_pool_embs))
        for idx_list in modifier_indices:
            for mi in idx_list:
                mod_freq[mi] += 1
        mod_freq    = mod_freq.clamp(min=1)
        # Inverse sqrt frequency — rare modifiers get higher contrastive weight
        mod_ct_weights = (1.0 / mod_freq.sqrt())
        mod_ct_weights = mod_ct_weights / mod_ct_weights.sum()
        mod_ct_weights = mod_ct_weights.to(DEVICE)
    else:
        mod_ct_weights = None

    def sample_contrastive_pairs_weighted(n: int):
        """Sample modifier pairs with inverse-frequency weighting."""
        if mod_pool_embs is None or len(mod_pool_embs) < 2:
            return None, None
        if mod_ct_weights is not None:
            ia = torch.multinomial(mod_ct_weights, n, replacement=True)
            ib = torch.multinomial(mod_ct_weights, n, replacement=True)
        else:
            ia = torch.randint(len(mod_pool_embs), (n,), device=DEVICE)
            ib = torch.randint(len(mod_pool_embs), (n,), device=DEVICE)
        same = ia == ib
        if same.any():
            ib[same] = (ib[same] + 1) % len(mod_pool_embs)
        return mod_pool_embs[ia], mod_pool_embs[ib]

    # ---------------------------------------------------------------------------
    # Eval subset (training data)
    # ---------------------------------------------------------------------------

    eval_idx   = list(range(min(512, N)))
    eval_head  = head_embs[eval_idx]
    eval_chunk = chunk_embs[eval_idx]
    eval_mods  = [modifier_embs[i] for i in eval_idx]
    eval_midx  = [modifier_indices[i] for i in eval_idx]

    # ---------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------

    print("Training...\n")

    loss_keys = ["total", "reconstruction", "collapse", "diversity",
                "directional", "contrastive", "transfer", "consistency", "roundtrip"]

    for epoch in range(1, EPOCHS + 1):
        model.train()
        # Weighted sampling without replacement for this epoch
        perm         = torch.multinomial(sample_weights, N, replacement=False)
        epoch_losses = {k: 0.0 for k in loss_keys}
        n_batches    = 0

        for start in range(0, N, BATCH_SIZE):
            idx        = perm[start:start + BATCH_SIZE].tolist()
            subj       = head_embs[idx]
            target     = chunk_embs[idx]
            batch_mods = [modifier_embs[i] for i in idx]
            batch_midx = [modifier_indices[i] for i in idx]

            feature, base_feat, first_post, deltas = encode_batch(subj, batch_mods, batch_midx)
            cortex_out, _ = model.reason(feature)
            reconstructed = model.decode(cortex_out)

            multi    = next((m for m in batch_mods if len(m) >= 2), [])
            mod_a, mod_b = sample_contrastive_pairs_weighted(CONTRASTIVE_PAIRS)

            # Cross-transfer: use a shuffled version of the same batch as alt heads
            alt_idx  = torch.randperm(len(idx))[:min(8, len(idx))]
            alt_heads = subj[alt_idx]

            losses = total_loss(
                target, reconstructed,
                feature,
                base_feat, first_post,
                target,           # input_embs = target chunk embs for diversity signal
                subj[:1], multi, model,
                mod_embs_a=mod_a,
                mod_embs_b=mod_b,
                head_embs_alt=alt_heads if len(batch_mods[0]) > 0 else None,
                collapse_weight=COLLAPSE_WEIGHT,
                diversity_weight=DIVERSITY_WEIGHT,
                directional_weight=DIRECTIONAL_WEIGHT,
                contrastive_weight=CONTRASTIVE_WEIGHT,
                transfer_weight=TRANSFER_WEIGHT,
                consistency_weight=CONSISTENCY_WEIGHT,
                roundtrip_weight=ROUNDTRIP_WEIGHT,
                contrastive_margin=CONTRASTIVE_MARGIN,
                min_cos_change=MIN_COS_CHANGE,
                min_var=MIN_VAR,
                diversity_margin=DIVERSITY_MARGIN,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in loss_keys:
                epoch_losses[k] += losses[k].item()
            n_batches += 1

        scheduler.step()

        if epoch % LOG_EPOCH_INTERVAL == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                ef, _, _, _ = encode_batch(eval_head, eval_mods, eval_midx)
                co, _       = model.reason(ef)
                er          = model.decode(co)
                train_cos   = F.cosine_similarity(er, eval_chunk).mean().item()
                val_cos     = val_recall()
            model.train()

            avg = {k: v / n_batches for k, v in epoch_losses.items()}
            print(
                f"Epoch {epoch:4d} | "
                f"recon={avg['reconstruction']:.4f} | "
                f"div={avg['diversity']:.4f} | "
                f"dirn={avg['directional']:.4f} | "
                f"ct={avg['contrastive']:.4f} | "
                f"xfer={avg['transfer']:.4f} | "
                f"col={avg['collapse']:.4f} | "
                f"cos={train_cos:.4f} | "
                f"val={val_cos:.4f}"
            )

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------

    cache_data = torch.load(CACHE_FILE, map_location="cpu", weights_only=False)
    torch.save({
        "model_state":    model.state_dict(),
        "semantic_dim":   SEMANTIC_DIM,
        "feature_dim":    FEATURE_DIM,
        "hidden_dim":     HIDDEN_DIM,
        "use_bridge":     USE_BRIDGE,
        "num_heads":      BRIDGE_HEADS,
        "corpus":         cache_data.get("corpus", []),
        "embeddings":     chunk_embs[:2000].cpu(),
        "val_sentences":  val_sentences,
        "val_embs":       val_embs.cpu() if val_embs is not None else None,
        "backbone_model": BACKBONE_MODEL,
        "dataset":        DATASET_NAMES,
    }, "feature_codec.pt")

    print("\nSaved to feature_codec.pt")

    return

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    try:
        mp.set_start_method('forkserver', force=True)
    except RuntimeError:
        pass 
    main()
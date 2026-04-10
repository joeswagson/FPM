from transformers import AutoTokenizer, AutoModel, BertForMaskedLM
import torch
import torch.nn.functional as F
from itertools import combinations
import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
MLM_MODEL = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)
mlm_model = BertForMaskedLM.from_pretrained(MLM_MODEL)
mlm_tokenizer = AutoTokenizer.from_pretrained(MLM_MODEL)
mlm_model.eval()

# -----------------------------
# CORE FUNCTIONS
# -----------------------------

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

def embed(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        out = model(**inputs)
    emb = mean_pooling(out, inputs['attention_mask'])
    return F.normalize(emb, dim=1)

def cos(a, b):
    return F.cosine_similarity(a, b).item()

# -----------------------------
# CONFIG
# -----------------------------

NOUNS = ["ball", "car", "apple", "shirt", "sky"]
MODIFIERS = ["red", "blue", "green"]
BAD_MODIFIERS = ["wet fart", "quantum spaghetti"]
EDGE_CASES = [("red", "lambda"), ("blue", "democracy")]

TEMPLATES = [
    "{} {}",
    "a {} {}",
    "the {} {}",
    "this {} {}"
]

NULL_NOUNS = ["freedom", "democracy", "lambda", "theorem", "justice", "entropy"]

# -----------------------------
# PHRASE HANDLING
# -----------------------------

def embed_phrase(mod, noun):
    vecs = [embed(t.format(mod, noun)) for t in TEMPLATES]
    return torch.mean(torch.stack(vecs), dim=0)

def diff_avg(mod, noun):
    return embed_phrase(mod, noun) - embed(noun)

# -----------------------------
# METRIC 1 — COSINE SIMILARITY (original)
# -----------------------------

def test_modifier_consistency(mod):
    vecs = [F.normalize(diff_avg(mod, n), dim=1) for n in NOUNS]
    sims = [cos(a, b) for a, b in combinations(vecs, 2)]
    return {
        "pairwise": [round(s, 3) for s in sims],
        "mean": round(sum(sims) / len(sims), 4)
    }

def test_modifier_similarity_cos(mod_a, mod_b):
    va = F.normalize(torch.mean(torch.stack([diff_avg(mod_a, n) for n in NOUNS]), dim=0), dim=1)
    vb = F.normalize(torch.mean(torch.stack([diff_avg(mod_b, n) for n in NOUNS]), dim=0), dim=1)
    return round(cos(va, vb), 4)

def test_applicability_cos(mod, noun, reference_vec):
    v = F.normalize(diff_avg(mod, noun), dim=1)
    return round(cos(v, reference_vec), 4)

# -----------------------------
# METRIC 2 — MAHALANOBIS DISTANCE
# -----------------------------

from sklearn.decomposition import PCA

PCA_DIMS = 8

# OLD
def fit_modifier_distribution(mod, nouns):
    vecs = np.stack([diff_avg(mod, n).squeeze().detach().numpy() for n in nouns])
    pca = PCA(n_components=min(PCA_DIMS, len(nouns) - 1))
    projected = pca.fit_transform(vecs)
    mu = projected.mean(axis=0)
    cov = np.cov(projected.T) + np.eye(projected.shape[1]) * 1e-4
    return mu, cov, pca


def mahalanobis_loo(mod, noun, all_nouns, pca=None, mu=None, cov=None):
    train_nouns = [n for n in all_nouns if n != noun]
    vecs = np.stack([diff_avg(mod, n).squeeze().detach().numpy() for n in train_nouns])

    pca = PCA(n_components=min(PCA_DIMS, len(train_nouns) - 1))
    projected = pca.fit_transform(vecs)
    mu = projected.mean(axis=0)
    cov = np.cov(projected.T) + np.eye(projected.shape[1]) * 1e-4

    # project test vec into the same PCA space
    test_vec = diff_avg(mod, noun).squeeze().detach().numpy()
    test_proj = pca.transform(test_vec.reshape(1, -1))[0]

    delta = test_proj - mu
    return float(np.sqrt(delta @ inv(cov) @ delta))

# -----------------------------
# METRIC 3 — GAUSSIAN LOG-LIKELIHOOD TYPICALITY
# -----------------------------

def typicality_score(mod, noun, mu, cov, pca):
    v = diff_avg(mod, noun).squeeze().detach().numpy()
    v_proj = pca.transform(v.reshape(1, -1))[0]
    rv = multivariate_normal(mean=mu, cov=cov, allow_singular=True)
    self_score = rv.logpdf(mu)
    target_score = rv.logpdf(v_proj)
    return round(float(np.exp(target_score - self_score)), 4)

# -----------------------------
# METRIC 4 — MLM TOKEN PROBABILITY
# -----------------------------

def mlm_compatibility(mod, noun):
    phrase = f"the {mod} [MASK]"
    inputs = mlm_tokenizer(phrase, return_tensors='pt')
    mask_idx = (inputs['input_ids'] == mlm_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    with torch.no_grad():
        logits = mlm_model(**inputs).logits

    probs = torch.softmax(logits[0, mask_idx], dim=-1)

    # handle multi-token nouns — average prob of first token only
    noun_ids = mlm_tokenizer.encode(noun, add_special_tokens=False)
    if not noun_ids:
        return 0.0
    noun_id = noun_ids[0]
    return round(probs[0, noun_id].item(), 6)

# -----------------------------
# METRIC 5 — CONTRASTIVE RANK SCORE
# -----------------------------

def rank_applicability(mod, noun, null_nouns=NULL_NOUNS):
    target_dist = mahalanobis_loo(mod, noun, NOUNS)
    null_dists = [mahalanobis_loo(mod, n, NOUNS) for n in null_nouns]
    rank = sum(1 for d in null_dists if d > target_dist)
    return round(rank / len(null_dists), 4)


# -----------------------------
# COMPOSITE SCORE
# -----------------------------

# Calibrated from empirical run — adjust if you expand NOUNS/MODIFIERS
COS_FLOOR = 0.70   # below this = clearly wrong
COS_CEIL  = 0.94   # above this = perfect (very color ball territory)

# Typicality is inverted: ~0.20 for sensible pairs, ~0.85 for nonsense subjects
# (1 - typ) gives ~0.80 for sensible, ~0.15 for nonsense — use as a multiplier
TYP_PENALTY_THRESHOLD = 0.50  # above this = subject is suspicious

def stretched_cos(cs):
    """
    Pull the compressed [0.70, 0.94] band into [0.0, 1.0].
    Anything below floor → 0, above ceil → 1.
    """
    return max(0.0, min(1.0, (cs - COS_FLOOR) / (COS_CEIL - COS_FLOOR)))

def mah_to_affinity(mah, scale=1.0):
    """
    Soft exponential decay. mah=0 → 1.0, mah=1 → 0.37, mah=2 → 0.14.
    Scale controls how harshly distance is penalized.
    """
    return float(np.exp(-mah / scale))

def base_alignment_score(mod, noun):
    """cos(phrase, bare noun) — already computed, just expose it"""
    phrase = embed_phrase(mod, noun)
    base = embed(noun)
    return round(cos(phrase, base), 4)

def composite_score(cs, mah, typ, base_align, weights=(0.50, 0.10, 0.15, 0.25)):
    w_cos, w_mah, w_typ, w_base = weights

    s_cos   = stretched_cos(cs)
    s_mah   = mah_to_affinity(mah)
    s_typ   = 1.0 - typ        # inverted: high = bad subject
    s_base  = base_align       # already in [0,1], no transform needed

    raw = w_cos * s_cos + w_mah * s_mah + w_typ * s_typ + w_base * s_base

    if typ > TYP_PENALTY_THRESHOLD:
        raw *= 0.5

    return round(raw, 4)

# -----------------------------
# ORIGINAL TESTS (kept intact)
# -----------------------------

def test_reconstruction(mod, noun):
    base = embed(noun)
    mod_vec = torch.mean(torch.stack([diff_avg(mod, n) for n in NOUNS]), dim=0)
    recon = F.normalize(base + mod_vec, dim=1)
    actual = embed_phrase(mod, noun)
    return round(cos(recon, actual), 4)

def test_base_alignment(mod, noun):
    phrase = embed_phrase(mod, noun)
    base = embed(noun)
    return round(cos(phrase, base), 4)

def test_intensity_scaling(mod, noun):
    base = embed(noun)
    normal = embed_phrase(mod, noun)
    strong = embed_phrase(f"very {mod}", noun)
    vn = F.normalize(normal - base, dim=1)
    vs = F.normalize(strong - base, dim=1)
    return {
        "dir_cos": round(cos(vn, vs), 4),
        "mag_normal": round(torch.norm(normal - base).item(), 4),
        "mag_very": round(torch.norm(strong - base).item(), 4),
    }

def test_very_consistency(modifiers, noun="ball"):
    vecs = []
    for m in modifiers:
        v = F.normalize(embed_phrase(f"very {m}", noun) - embed_phrase(m, noun), dim=1)
        vecs.append(v)
    sims = [cos(a, b) for a, b in combinations(vecs, 2)]
    return {
        "pairwise": [round(s, 3) for s in sims],
        "mean": round(sum(sims) / len(sims), 4)
    }

def test_very_transfer(mod, noun):
    base = embed_phrase(mod, noun)
    very_vec = torch.mean(torch.stack([
        embed_phrase(f"very {mod}", n) - embed_phrase(mod, n)
        for n in NOUNS
    ]), dim=0)
    recon = F.normalize(base + very_vec, dim=1)
    actual = embed_phrase(f"very {mod}", noun)
    return round(cos(recon, actual), 4)

# -----------------------------
# HELPERS
# -----------------------------

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def subsection(title):
    print(f"\n  ── {title}")

def row(label, val, width=40):
    print(f"    {label:<{width}} {val}")

# -----------------------------
# RUN
# -----------------------------

# Precompute modifier distributions once
print("\nFitting modifier distributions...")
distributions = {}
for m in MODIFIERS + BAD_MODIFIERS:
    distributions[m] = fit_modifier_distribution(m, NOUNS)
print("Done.\n")

# Precompute reference vecs for cos applicability
reference_vecs = {
    m: F.normalize(
        torch.mean(torch.stack([diff_avg(m, n) for n in NOUNS]), dim=0), dim=1
    )
    for m in MODIFIERS + BAD_MODIFIERS
}

# -------------------------------------------------------
section("1. MODIFIER CONSISTENCY  (diff-vector pairwise cos sim)")
# -------------------------------------------------------
for m in MODIFIERS + BAD_MODIFIERS:
    r = test_modifier_consistency(m)
    subsection(m)
    row("pairwise:", r["pairwise"])
    row("mean:", r["mean"])

# -------------------------------------------------------
section("2. MODIFIER SIMILARITY  (cos between averaged diff vecs)")
# -------------------------------------------------------
subsection("Within MODIFIERS")
for a, b in combinations(MODIFIERS, 2):
    row(f"{a} vs {b}:", test_modifier_similarity_cos(a, b))

subsection("BAD_MODIFIERS vs MODIFIERS")
for bad in BAD_MODIFIERS:
    for good in MODIFIERS:
        row(f"{bad} vs {good}:", test_modifier_similarity_cos(bad, good))

# -------------------------------------------------------
section("3. APPLICABILITY  (all 5 metrics per modifier × noun)")
# -------------------------------------------------------

ALL_TEST_PAIRS = (
    [(m, n) for m in MODIFIERS for n in NOUNS] +
    [(m, n) for m in BAD_MODIFIERS for n in NOUNS] +
    list(EDGE_CASES)
)

print(f"\n  {'Pair':<30} {'CosSim':>8} {'Mahala':>8} {'Typicty':>8} {'MLM_P':>10} {'Rank':>6}")
print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*6}")

for mod, noun in ALL_TEST_PAIRS:
    if mod not in distributions:
        # edge case modifiers — use nearest fitted modifier for distribution
        # (red for "red", blue for "blue")
        dist_key = mod if mod in distributions else MODIFIERS[0]
    else:
        dist_key = mod

    mu, cov, pca = distributions[dist_key]  # was: mu, cov = ...
    ref = reference_vecs[dist_key]

    cs = test_applicability_cos(mod, noun, ref)
    mah = round(mahalanobis_loo(mod, noun, NOUNS), 3)
    typ = typicality_score(mod, noun, mu, cov, pca)
    base_align = base_alignment_score(mod, noun)
    mlm = mlm_compatibility(mod, noun)
    score = composite_score(cs, mah, typ, base_align)

    # header
    print(f"\n  {'Pair':<30} {'CosSim':>8} {'Mahala':>8} {'Typicty':>8} {'Score':>7}")
    print(f"  {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 7}")

    # per pair
    score = composite_score(cs, mah, typ, base_align)
    label = f"{mod} {noun}"
    print(f"  {label:<30} {cs:>8} {mah:>8} {typ:>8} {base_align:>8} {score:>7}")
    # print(f"  {label:<30} {cs:>8} {mah:>8} {typ:>8} {mlm:>10.6f} {rnk:>6}")

# -------------------------------------------------------
section("4. RECONSTRUCTION  (cos between recon and actual phrase vec)")
# -------------------------------------------------------
for m in MODIFIERS:
    subsection(m)
    for n in NOUNS:
        row(f"{m} {n}:", test_reconstruction(m, n))

# -------------------------------------------------------
section("5. BASE ALIGNMENT  (cos between phrase and bare noun)")
# -------------------------------------------------------
for m in MODIFIERS + BAD_MODIFIERS:
    subsection(m)
    for n in ["ball", "lambda"]:
        row(f"{m} {n}:", test_base_alignment(m, n))

# -------------------------------------------------------
section("6. INTENSITY SCALING  (direction + magnitude of 'very')")
# -------------------------------------------------------
for m in MODIFIERS:
    subsection(m)
    for n in ["ball", "car"]:
        r = test_intensity_scaling(m, n)
        row(f"{m} {n} — dir_cos:", r["dir_cos"])
        row(f"{m} {n} — mag_normal:", r["mag_normal"])
        row(f"{m} {n} — mag_very:", r["mag_very"])

# -------------------------------------------------------
section("7. VERY CONSISTENCY  (do all 'very X' shifts point same way?)")
# -------------------------------------------------------
r = test_very_consistency(MODIFIERS)
row("pairwise:", r["pairwise"])
row("mean:", r["mean"])

# -------------------------------------------------------
section("8. VERY TRANSFER  (apply avg 'very' shift, compare to actual)")
# -------------------------------------------------------
for m in MODIFIERS:
    subsection(m)
    for n in ["ball", "car"]:
        row(f"very {m} {n}:", test_very_transfer(m, n))
"""
Noun Chunk Extractor
====================
Parses sentences into (head_noun, [modifiers], full_chunk_text) triples.

"a round fuzzy ball" → ("ball", ["round", "fuzzy"], "round fuzzy ball")
"elastic waistband"  → ("waistband", ["elastic"], "elastic waistband")
"zip-up front"       → ("front", ["zip-up"], "zip-up front")

This is the stage that dissolves grammar into subject+operator pairs.
Syntax is scaffolding — once we know which modifiers apply to which noun,
the sentence structure is discarded.
"""

import spacy
from dataclasses import dataclass

# Download: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")


@dataclass
class NounChunk:
    head: str           # The head noun — this becomes the feature node
    modifiers: list     # Adjectives/compounds — these become attribute operators
    full_text: str      # The complete chunk text — used for embedding the target
    source: str         # Original sentence, kept for debug/tracing


def extract_chunks(text: str) -> list[NounChunk]:
    """
    Extract all noun chunks from a sentence, splitting each into
    head noun and its modifiers.

    Compound modifiers (e.g. "zip-up") are kept as single tokens.
    Articles (a, an, the) are filtered — they carry no attribute information.
    """
    doc = nlp(text)
    chunks = []

    for chunk in doc.noun_chunks:
        head = chunk.root.lemma_.lower()

        modifiers = []
        for token in chunk:
            # Skip the head noun itself, articles, punctuation
            if token == chunk.root:
                continue
            if token.pos_ in ("DET",):  # articles: a, an, the
                continue
            if token.is_punct or token.is_space:
                continue
            # Keep adjectives, compounds, participial modifiers
            if token.pos_ in ("ADJ", "NOUN", "VERB") or token.dep_ in ("amod", "compound", "npadvmod"):
                modifiers.append(token.text.lower())

        # Skip chunks that are just a bare pronoun or very short noise
        if len(head) < 2:
            continue

        chunks.append(NounChunk(
            head=head,
            modifiers=modifiers,
            full_text=chunk.text.lower().strip(),
            source=text,
        ))

    return chunks


def extract_chunks_batch(texts: list[str], batch_size: int = 256) -> list[NounChunk]:
    """
    Batch extraction using spaCy's pipe for efficiency.
    """
    all_chunks = []
    for doc, text in zip(nlp.pipe(texts, batch_size=batch_size), texts):
        for chunk in doc.noun_chunks:
            head = chunk.root.lemma_.lower()
            modifiers = []
            for token in chunk:
                if token == chunk.root:
                    continue
                if token.pos_ in ("DET",):
                    continue
                if token.is_punct or token.is_space:
                    continue
                if token.pos_ in ("ADJ", "NOUN", "VERB") or token.dep_ in ("amod", "compound", "npadvmod"):
                    modifiers.append(token.text.lower())
            if len(head) < 2:
                continue
            all_chunks.append(NounChunk(
                head=head,
                modifiers=modifiers,
                full_text=chunk.text.lower().strip(),
                source=text,
            ))
    return all_chunks


if __name__ == "__main__":
    # Quick sanity check
    tests = [
        "a round ball",
        "a round fuzzy ball",
        "a round short-haired fuzzy ball",
        "Collared dress featuring sleeves below the elbow with cuffs and pleats.",
        "Midi dress with a lapel collar and crossover V-neckline, long sleeves.",
        "Oversized collared shirt with long cuffed sleeves.",
    ]

    for text in tests:
        print(f"\n'{text}'")
        for chunk in extract_chunks(text):
            print(f"  head='{chunk.head}'  modifiers={chunk.modifiers}  full='{chunk.full_text}'")
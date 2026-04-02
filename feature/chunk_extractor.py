"""
Noun Chunk Extractor
====================
Streaming design — never holds more than SPACY_BATCH sentences in memory
at once. Safe for multi-million sentence corpora.

Core idea: sentences come in as any iterable (list, generator, file lines).
The extractor slices them into small windows, runs spaCy on each window,
yields the resulting NounChunk objects, then drops the window before
loading the next one.
"""

import spacy
from dataclasses import dataclass, field
from typing import Iterable, Iterator
from tqdm import tqdm

# Disable components we don't need — saves ~40% memory and runs faster.
# We only need the tokenizer, tagger, and dependency parser for noun chunks.
nlp = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer"])
nlp.max_length = 2_000_000  # allow long docs if a sentence is huge


@dataclass
class NounChunk:
    head: str
    modifiers: list = field(default_factory=list)
    full_text: str = ""
    source: str = ""


def _parse_doc(doc: spacy.tokens.Doc, source: str) -> list[NounChunk]:
    """Extract NounChunk objects from a single parsed spaCy doc."""
    chunks = []
    for chunk in doc.noun_chunks:
        head = chunk.root.text.lower()  # skip lemmatizer — use raw text
        if len(head) < 2:
            continue

        modifiers = []
        for token in chunk:
            if token == chunk.root:
                continue
            if token.pos_ in ("DET", "PRON"):
                continue
            if token.is_punct or token.is_space:
                continue
            if token.pos_ in ("ADJ", "NOUN", "VERB") or \
               token.dep_ in ("amod", "compound", "npadvmod"):
                modifiers.append(token.text.lower())

        chunks.append(NounChunk(
            head=head,
            modifiers=modifiers,
            full_text=chunk.text.lower().strip(),
            source=source,
        ))
    return chunks


def stream_chunks(
    sentences: Iterable[str],
    spacy_batch: int = 512,    # sentences per spaCy pipe call — keep this small
    max_modifiers: int = 4,
    min_head_len: int = 2,
    show_progress: bool = True,
    total: int = None,         # optional: total sentence count for tqdm
) -> Iterator[NounChunk]:
    """
    Stream NounChunk objects from an iterable of sentences.

    Memory behaviour: at any point only `spacy_batch` sentences + their
    parsed docs exist in memory. Everything before the current window
    has been yielded and released.

    Args:
        sentences:      Any iterable of sentence strings. Can be a generator.
        spacy_batch:    Number of sentences processed per spaCy call.
                        512 is safe for ~3GB RAM. Lower if you're tight.
        max_modifiers:  Cap modifier count per chunk.
        min_head_len:   Skip head nouns shorter than this.
        show_progress:  Show a tqdm progress bar over sentences.
        total:          Total sentence count hint for tqdm (optional).
    """
    sentence_iter = iter(sentences)
    if show_progress:
        sentence_iter = tqdm(sentence_iter, total=total,
                             desc="Parsing chunks", unit=" sent")

    window = []
    for sentence in sentence_iter:
        window.append(sentence)

        if len(window) >= spacy_batch:
            # Process this window and immediately yield results
            for doc, src in zip(nlp.pipe(window, batch_size=spacy_batch), window):
                for chunk in _parse_doc(doc, src):
                    chunk.modifiers = chunk.modifiers[:max_modifiers]
                    if len(chunk.head) >= min_head_len:
                        yield chunk
            window = []  # drop references — GC can reclaim memory

    # Flush remaining sentences
    if window:
        for doc, src in zip(nlp.pipe(window, batch_size=spacy_batch), window):
            for chunk in _parse_doc(doc, src):
                chunk.modifiers = chunk.modifiers[:max_modifiers]
                if len(chunk.head) >= min_head_len:
                    yield chunk


def collect_chunks(
    sentences: Iterable[str],
    max_chunks: int = None,
    spacy_batch: int = 512,
    max_modifiers: int = 4,
    min_head_len: int = 2,
    show_progress: bool = True,
    total: int = None,
) -> list[NounChunk]:
    """
    Convenience wrapper: collect stream_chunks into a list, with optional cap.
    Use stream_chunks directly if you want to process without ever building
    a full list (e.g., writing to disk as you go).
    """
    chunks = []
    for chunk in stream_chunks(
        sentences,
        spacy_batch=spacy_batch,
        max_modifiers=max_modifiers,
        min_head_len=min_head_len,
        show_progress=show_progress,
        total=total,
    ):
        chunks.append(chunk)
        if max_chunks and len(chunks) >= max_chunks:
            break
    return chunks


if __name__ == "__main__":
    tests = [
        "a round ball",
        "a round fuzzy ball",
        "a round short-haired fuzzy ball",
        "Collared dress featuring sleeves below the elbow with cuffs and pleats.",
        "Midi dress with a lapel collar and crossover V-neckline, long sleeves.",
        "Oversized collared shirt with long cuffed sleeves.",
        "The large elastic waistband sits comfortably on the hips.",
    ]

    print("Streaming extractor test:\n")
    for chunk in stream_chunks(tests, spacy_batch=4, show_progress=False):
        print(f"  head='{chunk.head}'  mods={chunk.modifiers:<35}  full='{chunk.full_text}'")
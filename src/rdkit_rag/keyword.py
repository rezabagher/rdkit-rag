"""BM25 keyword search over chunks, plus rank fusion with vector search.

Embeddings are good at meaning ("how are rings perceived?") but often miss
exact identifiers such as ``MolToSmiles`` or ``sanitizeMol``. BM25 scores
chunks by how often the query's words appear in them, weighted by how rare
each word is across the index, which is exactly what identifier lookups need.

Code-aware tokenization: identifiers are kept whole and also split into
their camelCase / snake_case parts, so ``MolToSmiles`` matches a query for
"MolToSmiles" as well as one mentioning "smiles".

The BM25 weights are precomputed into a sparse (chunks x terms) matrix, so a
query is one sparse column slice and a sum.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

from .models import Chunk

# BM25 parameters: k1 controls term-frequency saturation, b controls how much
# longer chunks are penalized. These are the usual defaults.
BM25_K1 = 1.2
BM25_B = 0.75

# Test files get their keyword scores scaled by this factor. Test names
# (e.g. Catch2 TEST_CASE("github #5505: kekulization problem ...")) are
# written in plain words, so they otherwise outrank the implementation for
# most questions. They remain searchable, just ranked lower.
TEST_FILE_WEIGHT = 0.5
_TEST_DIRS = frozenset({"test", "tests", "testing", "test_data", "testdata", "bench"})
_TEST_NAME_RES = (
    re.compile(r"^(test|catch_)", re.IGNORECASE),  # testFoo.cpp, catch_mol.cpp
    re.compile(r"_tests?(_|\d|$)", re.IGNORECASE),  # rough_test.py, cxsmiles_test.cpp
    re.compile(r"Test"),  # UnitTestChem.py, MolOpsTest.cpp (but not "Contest")
    re.compile(r"tests?\d*$"),  # molopstest.cpp, smitest1.cpp
)

# Reciprocal rank fusion constant (60 is the value from the original paper).
RRF_K = 60

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_PART_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

STOPWORDS = frozenset(
    """
    a an and are as at be been but by can do does for from has have how if in
    into is it its of on or that the their then there these this to was what
    when where which while who why will with you your i we they them our not
    no yes so than too very just also should would could may might must shall
    implemented implement implementation used use using code function functions
    file files rdkit
    """.split()
)


# Suffixes stripped by the stemmer, longest first.
_SUFFIXES = (
    "izations", "isations", "ization", "isation", "ations", "ation",
    "izing", "ising", "izers", "izer", "ized", "ised", "izes", "ises",
    "ize", "ise", "ities", "ity", "ings", "ing", "ers", "er", "ed",
    "ies", "es", "s",
)
_VOWELS = frozenset("aeiou")
_MIN_STEM = 4


def _stem(token: str) -> str:
    """Crude suffix-stripping stemmer, applied identically to the index and
    the query, so related forms meet: kekulization / kekulize / kekulized
    -> "kekul", parser / parse / parsing -> "pars", fingerprints ->
    "fingerprint". Identifiers containing digits or underscores are left
    alone. It doesn't need to produce real words, only consistent ones."""
    if not token.isalpha() or len(token) <= _MIN_STEM:
        return token
    stem = token
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM:
            stem = token[: -len(suffix)]
            if suffix == "ies":
                stem += "y"
            break
    if len(stem) > _MIN_STEM and stem[-1] == stem[-2] and stem[-1] not in _VOWELS:
        stem = stem[:-1]  # embedding -> embedd -> embed
    if len(stem) > _MIN_STEM and stem.endswith("e"):
        stem = stem[:-1]  # parse -> pars, to match parser -> pars
    return stem


def _keep(token: str) -> bool:
    return len(token) > 1 and token not in STOPWORDS


def tokenize(text: str) -> list[str]:
    """Lowercased tokens: each identifier whole, plus its camelCase and
    snake_case parts when it has more than one."""
    tokens: list[str] = []
    for match in _IDENT_RE.finditer(text):
        word = match.group(0)
        whole = word.lower().strip("_")
        if _keep(whole):
            tokens.append(_stem(whole))
        parts = [p.lower() for piece in word.split("_") for p in _CAMEL_PART_RE.findall(piece)]
        if len(parts) > 1:
            tokens.extend(_stem(p) for p in parts if _keep(p))
    return tokens


def keyword_text(chunk: Chunk) -> str:
    """The text indexed for a chunk. The file path and symbol name are
    repeated so a match there counts more than one deep inside the body."""
    header = f"{chunk.file_path} {chunk.symbol_name or ''}"
    return f"{header} {header} {chunk.content}"


def is_test_path(path: str) -> bool:
    """Heuristic: does this repository path look like a test or benchmark?"""
    *dirs, name = path.split("/")
    if any(d.lower() in _TEST_DIRS for d in dirs):
        return True
    stem = name.rsplit(".", 1)[0]
    return any(r.search(stem) for r in _TEST_NAME_RES)


# Bump whenever tokenization, stemming or weighting changes, so cached
# keyword indexes built by older code are rebuilt automatically.
KEYWORD_INDEX_VERSION = 2


def chunks_fingerprint(chunks: list[Chunk]) -> str:
    """Identifies the exact chunks (and keyword-index format) an index was
    built for."""
    digest = hashlib.sha1(f"v{KEYWORD_INDEX_VERSION}\n".encode("utf-8"))
    for chunk in chunks:
        digest.update(chunk.id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class KeywordIndex:
    def __init__(
        self, weights: sparse.csc_matrix, vocab: dict[str, int], fingerprint: str
    ) -> None:
        self.weights = weights  # (num_chunks, num_terms) BM25 term weights
        self.vocab = vocab
        self.fingerprint = fingerprint

    @property
    def num_docs(self) -> int:
        return self.weights.shape[0]

    @classmethod
    def build(
        cls, chunks: list[Chunk], k1: float = BM25_K1, b: float = BM25_B
    ) -> KeywordIndex:
        vocab: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        tfs: list[int] = []
        doc_lens = np.zeros(len(chunks), dtype=np.float32)

        for i, chunk in enumerate(chunks):
            counts = Counter(tokenize(keyword_text(chunk)))
            doc_lens[i] = sum(counts.values())
            for term, count in counts.items():
                rows.append(i)
                cols.append(vocab.setdefault(term, len(vocab)))
                tfs.append(count)

        n_docs, n_terms = len(chunks), len(vocab)
        rows_a = np.asarray(rows, dtype=np.int32)
        cols_a = np.asarray(cols, dtype=np.int32)
        tf = np.asarray(tfs, dtype=np.float32)

        doc_freq = np.bincount(cols_a, minlength=n_terms).astype(np.float32)
        idf = np.log1p((n_docs - doc_freq + 0.5) / (doc_freq + 0.5)).astype(np.float32)
        avg_len = float(doc_lens.mean()) if n_docs and doc_lens.mean() > 0 else 1.0
        norm = k1 * (1.0 - b + b * doc_lens[rows_a] / avg_len)
        doc_weight = np.asarray(
            [TEST_FILE_WEIGHT if is_test_path(c.file_path) else 1.0 for c in chunks],
            dtype=np.float32,
        )
        values = doc_weight[rows_a] * idf[cols_a] * tf * (k1 + 1.0) / (tf + norm)

        weights = sparse.coo_matrix(
            (values, (rows_a, cols_a)), shape=(n_docs, n_terms), dtype=np.float32
        ).tocsc()
        return cls(weights, vocab, chunks_fingerprint(chunks))

    def scores(self, query: str) -> np.ndarray:
        counts = Counter(self.vocab[t] for t in tokenize(query) if t in self.vocab)
        if not counts:
            return np.zeros(self.num_docs, dtype=np.float32)
        term_ids = list(counts)
        query_weights = np.asarray([counts[t] for t in term_ids], dtype=np.float32)
        return np.asarray(self.weights[:, term_ids] @ query_weights).ravel()

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """(chunk index, BM25 score) pairs, best first; chunks sharing no
        word with the query are never returned."""
        if top_k <= 0 or self.num_docs == 0:
            return []
        scores = self.scores(query)
        matching = np.flatnonzero(scores > 0)
        if matching.size == 0:
            return []
        k = min(top_k, matching.size)
        top = matching[np.argpartition(-scores[matching], k - 1)[:k]]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(i), float(scores[i])) for i in top]

    def save(self, weights_path: Path, vocab_path: Path) -> None:
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(weights_path, self.weights, compressed=False)
        vocab_path.write_text(
            json.dumps({"fingerprint": self.fingerprint, "vocab": self.vocab}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, weights_path: Path, vocab_path: Path) -> KeywordIndex | None:
        if not weights_path.exists() or not vocab_path.exists():
            return None
        try:
            meta = json.loads(vocab_path.read_text(encoding="utf-8"))
            weights = sparse.load_npz(weights_path).tocsc()
        except (OSError, ValueError):
            return None
        return cls(weights, meta["vocab"], meta.get("fingerprint", ""))


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = RRF_K
) -> list[tuple[int, float]]:
    """Merge several ranked lists of chunk indices into one.

    Each list contributes 1 / (k + rank) for every chunk in it, so a chunk
    near the top of both lists beats one that tops only a single list.
    Ranks are used instead of raw scores because cosine similarities and
    BM25 scores are on unrelated scales.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))

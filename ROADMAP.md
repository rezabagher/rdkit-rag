# Roadmap

Where rdkit-rag is today and what comes next. Phases are ordered by value:
measure first, then improve retrieval, then deepen code understanding.

## Current state (v0.1)

- Index Python, C/C++, Markdown and RST; respects `.gitignore`
- Language-aware chunking: `ast` for Python, heuristic signature + brace
  matching for C++, headings for docs; ~99% of non-blank lines indexed on RDKit
- Local embeddings (sentence-transformers, BGE) and BM25 keyword search,
  combined into hybrid search
- `search` and `ask` (Ollama) with cited sources
- Incremental updates: re-running `index` embeds only new or changed code
- `rdkit-rag eval`: 45-question benchmark reporting Recall@k and MRR
- 114 unit tests, no model download needed to run them

Measured on RDKit (laptop, GTX 1650 Ti): 2,201 files, 25,004 chunks,
~20 min for a first full index, ~16 s for an update with no changes,
~7 s per search, ~44 s per answer with qwen3:1.7b.

## Phase 1: Measure and improve retrieval

- [x] **Evaluation set**: 45 RDKit questions (30 concept, 15 identifier)
  with the known correct file/function, and `rdkit-rag eval` reporting
  Recall@k and MRR. Baseline in `eval/baseline.json`: hybrid R@10 0.49,
  MRR 0.33 (identifier R@10 0.93, concept R@10 0.27).
- [x] **Hybrid search**: BM25 keyword search merged with vector results
  (reciprocal rank fusion), with code-aware tokenization. Targets exact
  identifiers such as `MolToSmiles` or `sanitizeMol` that embeddings often
  miss. Measured: hybrid has the best R@5/R@10, but keyword alone has a
  slightly higher MRR (0.35 vs 0.33).
- [ ] **Fix concept questions** (R@10 0.27): the baseline shows they mostly
  retrieve documentation and its Japanese translation (`Docs/Book_jp/`)
  instead of the implementing code, plus headers instead of `.cpp` files.
  Try: skipping or down-weighting duplicate translations, a code/docs
  filter, and linking a header declaration to its definition.
- [ ] **Reranking**: re-order the top ~50 candidates with a cross-encoder.
  *Done when* it improves MRR on the eval set without making search too slow.

## Phase 2: Deeper code understanding

- [ ] **tree-sitter C++ chunking**: replace the heuristic chunker with a real
  parse; keep the heuristic as a fallback.
- [ ] **Python ↔ C++ links**: map `rdkit.Chem` functions to the C++ code they
  wrap through the Boost.Python wrapper files, and retrieve both together.
- [ ] **Test ↔ implementation links**: connect Catch2 `TEST_CASE`s and Python
  tests to the functions they exercise.

## Phase 3: Day-to-day usability

- [x] **Incremental indexing**: every run re-chunks the repository and reuses
  the stored embedding of each chunk whose text is unchanged; only new or
  changed chunks are embedded. Updating a 21 Nov index (1,470 chunks changed)
  took 67 s instead of ~20 min. Still to do: re-chunk only files whose
  content hash changed, to cut the ~16 s no-change update further.
- [ ] **Persistent server mode**: keep the embedding model loaded, so search
  drops from ~7 s to under 1 s.
- [ ] **Streaming answers** from Ollama.
- [ ] **Citation check**: verify every `[file:start-end]` in an answer refers to
  a chunk that was actually in the context.

## Later / optional

- **cppcheck findings as metadata**: attach static-analysis warnings to chunks
  by line range. Only if a trial on `Code/GraphMol/` shows mostly real issues
  (without a compile database, expect many false positives).
- **Approximate vector index** (FAISS/HNSW) if the index grows to millions of
  chunks or multiple repositories.

## Out of scope

- Cloud LLM or embedding APIs; the tool stays local-first.
- Multi-user hosting.

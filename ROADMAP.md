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
- 71 unit tests, no model download needed to run them

Measured on RDKit (laptop, GTX 1650 Ti): 2,201 files, 24,015 chunks,
~20 min to index, ~7 s per search, ~44 s per answer with qwen3:1.7b.

## Phase 1: Measure and improve retrieval

- [ ] **Evaluation set**: 30–50 RDKit questions with the known correct
  file/function, and an `rdkit-rag eval` command reporting recall@k and MRR.
  *Done when* every later change can be compared against a baseline number.
- [x] **Hybrid search**: BM25 keyword search merged with vector results
  (reciprocal rank fusion), with code-aware tokenization. Targets exact
  identifiers such as `MolToSmiles` or `sanitizeMol` that embeddings often
  miss. Still to do: confirm the gain on the evaluation set above.
- [ ] **Reranking**: re-order the top ~50 candidates with a cross-encoder.
  *Done when* it improves MRR without making search too slow.

## Phase 2: Deeper code understanding

- [ ] **tree-sitter C++ chunking**: replace the heuristic chunker with a real
  parse; keep the heuristic as a fallback.
- [ ] **Python ↔ C++ links**: map `rdkit.Chem` functions to the C++ code they
  wrap through the Boost.Python wrapper files, and retrieve both together.
- [ ] **Test ↔ implementation links**: connect Catch2 `TEST_CASE`s and Python
  tests to the functions they exercise.

## Phase 3: Day-to-day usability

- [ ] **Incremental indexing**: re-embed only files whose content hash changed.
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

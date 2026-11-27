# Roadmap

Where rdkit-rag is today and what comes next. Phases are ordered by value:
measure first, then improve retrieval, then deepen code understanding.

## Current state (v0.1, beta)

- Index Python, C/C++, Markdown and RST; respects `.gitignore`
- Language-aware chunking: `ast` for Python, heuristic signature + brace
  matching for C++, headings for docs; ~99% of non-blank lines indexed on RDKit
- Local embeddings (sentence-transformers, BGE) and BM25 keyword search,
  combined into hybrid search
- `search` and `ask` (Ollama) with cited sources
- Incremental updates: re-running `index` embeds only new or changed code
- `rdkit-rag eval`: 45-question benchmark reporting Recall@k and MRR,
  plus 18 held-out questions
- Definition links: hybrid search follows code names in the top docs and
  headers to the code that defines them
- 127 unit tests, no model download needed to run them

Measured on RDKit (laptop, GTX 1650 Ti): 2,201 files, 25,004 chunks,
~20 min for a first full index, ~16 s for an update with no changes,
~7 s per search, ~44 s per answer with qwen3:1.7b.

## Phase 1: Measure and improve retrieval

- [x] **Evaluation set**: 45 RDKit questions (30 concept, 15 identifier)
  with the known correct file/function, and `rdkit-rag eval` reporting
  Recall@k and MRR. First baseline: hybrid R@10 0.49, MRR 0.33
  (identifier R@10 0.93, concept R@10 0.27). Plus 18 held-out concept
  questions (`eval/heldout_questions.toml`) for checking changes that were
  tuned on the main set.
- [x] **Hybrid search**: BM25 keyword search merged with vector results
  (reciprocal rank fusion), with code-aware tokenization. Targets exact
  identifiers such as `MolToSmiles` or `sanitizeMol` that embeddings often
  miss. Measured: hybrid has the best R@5/R@10, but keyword alone has a
  slightly higher MRR (0.35 vs 0.33).
- [x] **Definition links for concept questions**: concept questions mostly
  retrieved documentation and headers instead of the implementing code.
  Hybrid search now follows code names mentioned in the top doc/header
  results to their definitions. Concept R@10 0.27 -> 0.43, MRR 0.14 -> 0.30;
  held-out R@10 0.33 -> 0.44, MRR 0.11 -> 0.32; identifier R@10 0.93 -> 1.00.
  Tried and rejected (gain of at most one question): dropping the Japanese
  doc translations, down-weighting or capping docs, capping chunks per file,
  boosting the `.cpp` next to a matching header, and reranking with
  `ms-marco-MiniLM-L-6-v2` (also ~2 s per query).
- [ ] **Remaining concept misses** (R@10 0.43): often the explaining docs
  never name the function. Ideas: links from query words to symbol names,
  a code-specific embedding model.
- [ ] **Reranking**: re-order the top ~50 candidates with a cross-encoder.
  *Done when* it improves MRR on the eval set without making search too slow.
  A general web-search reranker (`ms-marco-MiniLM-L-6-v2`) didn't help;
  next to try: `bge-reranker-base` or a code-trained reranker.

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

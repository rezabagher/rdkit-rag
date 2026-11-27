# Changelog

All notable changes to rdkit-rag are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

First public release (beta).

### Added

- `rdkit-rag index`: scans Python, C/C++, Markdown and reStructuredText
  files, skipping git-ignored files, and splits them into chunks: Python by
  `ast`, C/C++ by signature and brace matching, docs by section.
- Local embeddings with sentence-transformers (BGE models by default).
- BM25 keyword index with code-aware tokenization (whole identifiers plus
  their camelCase and snake_case parts).
- `rdkit-rag search` with three modes: hybrid (default, reciprocal rank
  fusion of vector and keyword results), vector, and keyword.
- `rdkit-rag ask`: answers from the retrieved code with a local Ollama
  model, citing files and line ranges.
- Incremental indexing: re-running `index` embeds only new or changed
  chunks and reuses all other embeddings. Index files are written
  atomically, and the index records the git commit it was built from.
- `rdkit-rag eval`: a built-in set of 45 RDKit questions with known
  relevant locations, reporting Recall@k and MRR per search mode and
  category, with optional JSON output. Baseline in `eval/baseline.json`.

[0.1.0]: https://github.com/rezabagher/rdkit-rag/releases/tag/v0.1.0

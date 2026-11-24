# rdkit-rag

rdkit-rag is a local question-answering tool for the
[RDKit](https://github.com/rdkit/rdkit) cheminformatics codebase. RDKit is a
large mix of C++ and Python, about 2,200 source and doc files, and it's hard
to find where a feature is implemented: `grep` only finds exact words, and the
whole repo is far too big to paste into an AI model. rdkit-rag splits the code
into meaningful pieces (Python functions via the `ast` module, C++ functions
via signature and brace matching, docs by section). It turns each piece into
an embedding vector stored on disk, and when you ask a question in plain
English it finds the most relevant pieces by cosine similarity. A local LLM
running in [Ollama](https://ollama.com) then answers from that code and cites
the exact files and line numbers it used. Everything runs on your own machine
with no API keys or cloud services. Developers can explore an unfamiliar
codebase in natural language and check every claim against the real source.

This is an MVP. It favors simplicity and readability over completeness -
see [Current limitations](#current-limitations).

## Architecture

```
RDKit repository
        |
        v
File Scanner (.py, .cpp/.cc/.cxx/.h/.hpp, .md/.rst)
        |
        v
Code / Documentation Chunker
   (Python: ast module — Markdown/RST: sections)
   (C/C++: heuristic brace matching)
        |
        v
sentence-transformers Embeddings  +  BM25 Keyword Index
        |
        v
Local Storage (.data/)
   chunks.jsonl + embeddings.npy + bm25_weights.npz + index_metadata.json
        |
        v
Hybrid Search
   (NumPy cosine similarity + BM25, merged by reciprocal rank fusion)
        |
        v
Top Relevant Chunks
        |
        v
Ollama (local LLM)
        |
        v
Answer + Source Citations
```

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally (only needed
  for `rdkit-rag ask` - not for `index` or `search`)
- A local clone of the RDKit repository (or any codebase, really)

## Installation

```bash
git clone <this-repo> rdkit-rag
cd rdkit-rag
pip install -e ".[dev]"
```

This installs the `rdkit-rag` CLI along with its runtime dependencies
(`typer`, `pydantic`, `sentence-transformers`, `numpy`, `ollama`,
`python-dotenv`) and `pytest` for running the test suite.

## Ollama setup

1. Install Ollama: https://ollama.com/download
2. Start the server (usually automatic after install, or run `ollama serve`).
3. Pull a model, e.g.:

   ```bash
   ollama pull qwen3:8b
   ```

4. Make sure `OLLAMA_MODEL` (see below) matches the model you pulled.

`rdkit-rag ask` checks that Ollama is reachable and that the configured
model (exact tag, e.g. `qwen3:8b` is not satisfied by `qwen3:1.7b`) is available before doing any retrieval work, and prints a clear
error if not. `rdkit-rag index` and `rdkit-rag search` don't need Ollama
at all.

## Configuration

Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

```ini
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5   # any sentence-transformers model
OLLAMA_MODEL=qwen3:8b                    # any model you've `ollama pull`ed
OLLAMA_HOST=http://localhost:11434       # default Ollama host
```

No API keys are required anywhere.

## Usage

### 1. Index a repository

```bash
rdkit-rag index /path/to/rdkit
```

This scans supported files, chunks them, generates embeddings, and stores
everything under `.data/` in the current directory. If the target is a git
checkout, git-ignored files are skipped (e.g. the third-party sources RDKit's
CMake build downloads into `External/`). If an index already
exists there, re-running requires `--force`:

```bash
rdkit-rag index /path/to/rdkit --force
```

### 2. Search (no LLM required)

```bash
rdkit-rag search "Where is molecule sanitization implemented?"
rdkit-rag search "How are SMILES parsed?" --top-k 5
```

Prints the top matching chunks with file, symbol, and line-range info plus
the chunk content.

By default search is **hybrid**: it ranks chunks both by meaning (cosine
similarity of embeddings) and by keywords (BM25), then merges the two
rankings with reciprocal rank fusion. Each result shows its rank in both,
e.g. `(vector #4, keyword #1)`. Use `--mode` to pick one method:

```bash
rdkit-rag search "MolToSmiles" --mode keyword   # exact names; fast, no embedding model
rdkit-rag search "how are rings perceived" --mode vector
```

The keyword index is code-aware: identifiers are matched whole and by their
camelCase / snake_case parts (`MolToSmiles` also matches "smiles"), related
word forms are merged (kekulization / kekulize), and test files are ranked
below implementation files that use the same words. It is built during
`rdkit-rag index`; indexes made with older versions get it built
automatically, in a few seconds, on first search.

`ask` accepts the same `--mode` option.

### 3. Ask a question (uses Ollama)

```bash
rdkit-rag ask "How does RDKit sanitize a molecule?"
rdkit-rag ask "How does RDKit sanitize a molecule?" --top-k 8 --show-context
```

Retrieves relevant chunks, builds a bounded context window, sends it to the
local Ollama model along with the question, and prints the answer followed
by the exact sources it was grounded in:

```
ANSWER

RDKit sanitization is handled primarily by ...

SOURCES USED

1. Code/GraphMol/MolOps.cpp:123-180
2. Code/GraphMol/testMolOps.cpp:45-90
```

## Running tests

```bash
pytest
```

Tests use small fixture strings/files rather than the full RDKit repository,
and the embedding model is mocked in retrieval tests so no model download
is required to run the suite.

## Current limitations

- No reranking step after hybrid retrieval, and no evaluation set yet to
  measure retrieval quality systematically.
- C/C++ chunking is heuristic (signature detection + brace counting), not a
  real parser. It handles typical function/method definitions well but can
  miss unusual formatting; it falls back to overlapping line-based chunks
  when it can't confidently find functions in a file.
- No call graph or cross-reference analysis.
- No Python/C++ binding detection (e.g. linking `rdkit.Chem` Python calls to
  their C++ implementations).
- No test-to-implementation relationship detection.
- The index records which embedding model built it, and `search`/`ask`
  always embed queries with that model. Changing `EMBEDDING_MODEL` therefore
  requires `rdkit-rag index ... --force` to take effect.
- Indexing is full-rebuild only - there's no incremental/diff-based
  re-indexing yet.
- Designed for a single local repository, not for scaling to many
  repositories or many concurrent users.

See [ROADMAP.md](ROADMAP.md) for planned work.

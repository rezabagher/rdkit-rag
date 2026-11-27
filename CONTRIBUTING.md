# Contributing to rdkit-rag

Bug reports, questions and pull requests are welcome.

## Reporting a problem

Open an issue at https://github.com/rezabagher/rdkit-rag/issues and include:

- the command you ran and its full output
- your OS, Python version, and `EMBEDDING_MODEL` / `OLLAMA_MODEL` settings
- the RDKit commit you indexed (`git -C /path/to/rdkit rev-parse HEAD`)

For a wrong or missing search result, give the query and the file or
function you expected to find.

## Development setup

```bash
git clone https://github.com/rezabagher/rdkit-rag.git
cd rdkit-rag
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The tests use small fixture files and a fake embedding model, so they need
neither the RDKit repository nor a model download, nor Ollama.

## Making a change

1. Create a branch from `main`. Changes reach `main` only through a pull
   request.
2. Add or update tests for the behavior you change, and make sure `pytest`
   passes. CI runs the tests on Python 3.11, 3.12 and 3.13.
3. If the change affects chunking, embedding or ranking, run the retrieval
   benchmark before and after on the same index, and put both results in
   the pull request:

   ```bash
   rdkit-rag eval --output before.json
   # apply your change; re-run `rdkit-rag index` if it affects chunking
   rdkit-rag eval --output after.json
   ```

   The questions in `src/rdkit_rag/data/rdkit_questions.toml` were written
   against a pinned RDKit commit (`rdkit_commit` in that file). Please don't
   edit their relevant locations to make a result look better; if a label
   is wrong, fix it in a separate pull request that explains why.
4. Update `README.md` and `CHANGELOG.md` (under "Unreleased") when you change
   user-visible behavior.

## Code style

Keep it simple and readable: standard library and the existing dependencies
first, small functions, and comments that explain why rather than what.

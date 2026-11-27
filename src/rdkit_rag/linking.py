"""Definition links: follow code names mentioned in documentation and
headers to the code that defines them.

A plain-English question ("How does RDKit find the smallest set of smallest
rings?") is usually matched best by text that explains the concept: a
section of the RDKit Book or a commented declaration in a header. Those
texts name the functions involved (``findSSSR``, ``SanitizeMol``), but the
definitions themselves are written in code vocabulary and rank poorly. So
after hybrid search, the code names mentioned in the top documentation and
header results are looked up in a table of definitions, and the defining
chunks are merged into the ranking.

On the built-in benchmark this raised concept-question Recall@10 from 0.27 to
0.43, and from 0.33 to 0.44 on 18 held-out questions written afterwards
(eval/heldout_questions.toml).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .keyword import is_test_path
from .models import Chunk

DOC_LANGUAGES = frozenset({"Markdown", "reStructuredText"})
HEADER_SUFFIXES = (".h", ".hpp")

# Look for documentation/header chunks among this many top results...
LINK_WINDOW = 20
# ...and read code names from the first this-many of them.
LINK_SOURCES = 3

_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")
# camelCase / PascalCase with an inner capital (sanitizeMol, MolToSmiles,
# GetSSSR), or snake_case. Plain words such as "Molecule" or "RDKit" don't
# qualify, so prose doesn't link to unrelated functions with common names.
_CODE_NAME_RE = re.compile(r"[a-z][A-Z]|[A-Z][a-z]+[A-Z]")


def is_header(chunk: Chunk) -> bool:
    return chunk.file_path.endswith(HEADER_SUFFIXES)


def is_describing(chunk: Chunk) -> bool:
    """Documentation and headers describe code rather than implement it."""
    return chunk.language in DOC_LANGUAGES or is_header(chunk)


def looks_like_code_name(token: str) -> bool:
    return bool(_CODE_NAME_RE.search(token)) or "_" in token.strip("_")


def short_name(symbol_name: str) -> str:
    """The last component of a symbol: MolOps::sanitizeMol -> sanitizeMol,
    SaltRemover.StripMol -> StripMol."""
    return re.split(r"::|\.", symbol_name)[-1]


@dataclass(frozen=True)
class Link:
    definition: int  # chunk index of the defining code
    name: str  # the code name as it was written in the source
    source: int  # chunk index of the doc/header that mentions it


class DefinitionIndex:
    """Maps a function/class name to the chunk that defines it.

    Only implementation chunks count as definitions: not documentation, not
    headers (declarations), and test code only if nothing else defines the
    name. Names are compared case-insensitively, because docs often use the
    Python spelling (SanitizeMol) of a C++ function (sanitizeMol).
    """

    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        self._by_name: dict[str, int] = {}
        for i, chunk in enumerate(chunks):
            if not chunk.symbol_name or is_describing(chunk):
                continue
            name = short_name(chunk.symbol_name).lower()
            current = self._by_name.get(name)
            if current is None or (
                is_test_path(chunks[current].file_path)
                and not is_test_path(chunk.file_path)
            ):
                self._by_name[name] = i

    def links(self, ranking: list[int]) -> list[Link]:
        """Definitions of the code names mentioned by the top describing
        chunks in a ranking, most often mentioned first."""
        sources = [
            i for i in ranking[:LINK_WINDOW] if is_describing(self._chunks[i])
        ][:LINK_SOURCES]

        mentions: Counter[str] = Counter()
        first_seen: dict[str, tuple[str, int]] = {}
        for source in sources:
            names = set(_IDENTIFIER_RE.findall(self._chunks[source].content))
            # Sorted so that ties are broken the same way on every run.
            for name in sorted(names):
                key = name.lower()
                if key in self._by_name and looks_like_code_name(name):
                    mentions[key] += 1
                    first_seen.setdefault(key, (name, source))

        return [
            Link(definition=self._by_name[key], name=first_seen[key][0], source=first_seen[key][1])
            for key, _ in mentions.most_common()
        ]

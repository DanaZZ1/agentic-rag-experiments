"""Document loaders for text/markdown files and directories."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    doc_id: str
    text: str


def load_file(path: str | Path) -> Document:
    p = Path(path)
    return Document(p.stem, p.read_text(encoding="utf-8"))


def load_directory(path: str | Path, patterns: tuple[str, ...] = ("*.md", "*.txt")) -> list[Document]:
    base = Path(path)
    docs: list[Document] = []
    for pattern in patterns:
        for fp in sorted(base.rglob(pattern)):
            docs.append(Document(fp.stem, fp.read_text(encoding="utf-8")))
    return docs

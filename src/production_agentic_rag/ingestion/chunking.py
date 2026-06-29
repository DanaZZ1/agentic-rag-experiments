"""Chunking strategies producing retrievable units with metadata."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def fixed_size_chunks(text: str, *, size: int = 120, overlap: int = 24,
                      doc_id: str = "doc") -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    chunks: list[Chunk] = []
    step = size - overlap
    for i, start in enumerate(range(0, max(len(words), 1), step)):
        window = words[start:start + size]
        if not window:
            break
        chunks.append(Chunk(f"{doc_id}:{i}", " ".join(window),
                            {"doc_id": doc_id, "start": start, "strategy": "fixed"}))
        if start + size >= len(words):
            break
    return chunks


def markdown_section_chunks(text: str, *, doc_id: str = "doc") -> list[Chunk]:
    parts = re.split(r"\n(?=#{1,6}\s)", text.strip())
    chunks: list[Chunk] = []
    for i, part in enumerate(p.strip() for p in parts if p.strip()):
        title = part.splitlines()[0].lstrip("# ").strip() if part.startswith("#") else "intro"
        chunks.append(Chunk(f"{doc_id}:{i}", part,
                            {"doc_id": doc_id, "section": title, "strategy": "section"}))
    return chunks


CHUNKERS = {"fixed": fixed_size_chunks, "section": markdown_section_chunks}

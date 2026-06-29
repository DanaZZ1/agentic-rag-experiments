"""Shared text utilities: tokenization, light stemming, stopwords, cosine."""
from __future__ import annotations
import math
import re

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "in",
    "on", "for", "and", "or", "how", "what", "which", "who", "does", "do", "did",
    "with", "it", "this", "that", "these", "those", "i", "you", "we", "my", "our",
    "long", "can", "will", "from", "by", "at", "as", "if", "then", "than", "into",
}


def stem(word: str) -> str:
    """Tiny suffix-stripping stemmer (returns/returned/returning -> return)."""
    for suf in ("ing", "edly", "ed", "ies", "es", "s"):
        if len(word) > len(suf) + 2 and word.endswith(suf):
            base = word[: -len(suf)]
            return base + "y" if suf == "ies" else base
    return word


def tokenize(text: str, *, remove_stopwords: bool = True, do_stem: bool = True) -> list[str]:
    out = []
    for t in _TOKEN.findall(text.lower()):
        if remove_stopwords and t in STOPWORDS:
            continue
        out.append(stem(t) if do_stem else t)
    return out


def cosine(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def sentences(text: str) -> list[str]:
    text = re.sub(r"#{1,6}\s*[^\n]*", " ", text)          # strip markdown headers
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

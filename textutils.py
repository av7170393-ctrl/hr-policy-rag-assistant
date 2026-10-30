"""Pure-Python text helpers (no heavy dependencies, easy to unit test)."""
import re

STOP = set("a an the is are am of to in on for and or my me i do does can how what when where which who "
           "will be it at by with from as this that there their get much many".split())


def tokenize(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOP:
            continue
        if len(w) > 3 and w.endswith("s"):
            w = w[:-1]
        out.append(w)
    return out


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def split_text(text: str, size: int = 700, overlap_sentences: int = 1) -> list[str]:
    """Sentence-aware chunking with a small overlap so answers are not cut mid-rule."""
    chunks, cur = [], []
    for s in split_sentences(text):
        if cur and len(" ".join(cur)) + len(s) + 1 > size:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sentences:] if overlap_sentences else []
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def rrf(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion: merge several ranked id lists into one."""
    scores: dict[str, float] = {}
    for ranked in rank_lists:
        for pos, item in enumerate(ranked):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + pos + 1)
    return sorted(scores, key=scores.get, reverse=True)

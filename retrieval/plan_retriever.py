from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from retrieval.loader import load_json

_plan_data = load_json("section_targets/assessment_and_plan.json").get("data", [])

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "do",
    "does",
    "did",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "so",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "up",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
    # temporal/bridging words that often create accidental overlap
    "today",
    "yesterday",
    "tomorrow",
    "just",
    "recent",
    "recently",
    "past",
    "over",
    "ago",
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
    "time",
    "times",
    "been",
}


def _tokenize(text: str) -> set[str]:
    text = re.sub(r"\[(doctor|patient)\]", " ", text, flags=re.IGNORECASE)
    toks = {m.group(0).lower() for m in _WORD_RE.finditer(text)}
    return {t for t in toks if t not in _STOPWORDS}


# Precompute source token sets once (important for Streamlit responsiveness).
_plan_src_tokens: List[set[str]] = []
for _item in _plan_data:
    _plan_src_tokens.append(_tokenize(str(_item.get("src", "") or "")))


def get_dataset_plans(query_text: str, k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve dataset-based assessment/plan candidates.

    Returns structured, auditable candidates that may be used verbatim/light-cleaned.
    Output format:
      [{"case_id": str, "plan_text": str, "score": int}, ...]
    """
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return []

    scored: List[Tuple[int, int]] = []
    for idx, doc_tokens in enumerate(_plan_src_tokens):
        scored.append((len(query_tokens & doc_tokens), idx))

    scored.sort(key=lambda t: (-t[0], t[1]))

    # Prefer non-zero overlap; otherwise take top-k anyway.
    top = [idx for s, idx in scored if s > 0][:k]
    if not top:
        top = [idx for _, idx in scored[:k]]

    out: List[Dict[str, Any]] = []
    for i in top:
        item = _plan_data[i] if i < len(_plan_data) else {}
        plan_text = (item.get("tgt", "") or "").strip()
        if not plan_text:
            continue
        case_id = str(item.get("file") or "").strip() or str(item.get("id") or "").strip()
        # Recompute score from scored list (stable and deterministic).
        score = 0
        for s, idx in scored:
            if idx == i:
                score = int(s)
                break
        out.append({"case_id": case_id, "plan_text": plan_text, "score": score})

    return out


def get_plans(symptom_text: str, k: int = 3) -> List[str]:
    """Backward-compatible wrapper that returns only plan texts."""
    return [d["plan_text"] for d in get_dataset_plans(symptom_text, k=k) if d.get("plan_text")]

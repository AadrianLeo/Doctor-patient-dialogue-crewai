from __future__ import annotations

import re
from typing import List, Tuple

from retrieval.loader import load_json

_objective_data = load_json("section_targets/objective_exam.json")["data"]

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> set[str]:
    text = re.sub(r"\[(doctor|patient)\]", " ", text, flags=re.IGNORECASE)
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


def get_objective_exams(symptom_text: str, k: int = 3) -> List[str]:
    """Retrieve objective exam *examples* from similar dialogues.

    These are contextual references only (not auto-assumed).
    """
    query_tokens = _tokenize(symptom_text)
    if not query_tokens:
        return []

    scored: List[Tuple[int, int]] = []
    for idx, item in enumerate(_objective_data):
        src = item.get("src", "") or ""
        doc_tokens = _tokenize(src)
        scored.append((len(query_tokens & doc_tokens), idx))

    scored.sort(key=lambda t: (-t[0], t[1]))
    top = [idx for s, idx in scored if s > 0][:k]
    if not top:
        top = [idx for _, idx in scored[:k]]

    return [(_objective_data[i].get("tgt", "") or "").strip() for i in top if (_objective_data[i].get("tgt", "") or "").strip()]

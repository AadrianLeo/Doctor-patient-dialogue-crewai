from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `app.*` and `retrieval.*` imports work.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.patient_memory import PatientMemory
from app.followup_runtime import build_context_text, followup_topic, pick_next_followup


def preview_first_n_followups(*, patient_first_message: str, n: int = 5) -> list[str]:
    mem = PatientMemory()
    mem.update_from_patient_text(patient_first_message, is_first_message=True)

    patient_msgs: list[str] = [patient_first_message]
    context_text = build_context_text(patient_msgs)

    asked_lower: set[str] = set()
    asked_questions: list[str] = []
    topics_covered: set[str] = set(mem.inferred_topics())

    out: list[str] = []
    for _ in range(int(n)):
        q = pick_next_followup(
            context_text=context_text,
            query_text=mem.as_retrieval_query(include_history_snippets=False) or context_text,
            asked_lower=asked_lower,
            asked_questions=asked_questions,
            topics_covered=topics_covered,
            patient_memory=mem,
            k=80,
        )
        if not q:
            break
        out.append(q)
        asked_lower.add(q.lower())
        asked_questions.append(q)
        topics_covered.add(followup_topic(q))

    return out


if __name__ == "__main__":
    qs = preview_first_n_followups(patient_first_message="hello i have fever", n=5)
    for i, q in enumerate(qs, start=1):
        print(f"Q{i}: {q}")

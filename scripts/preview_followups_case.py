from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.patient_memory import PatientMemory
from app.followup_runtime import build_context_text, pick_next_followup, followup_topic


def simulate(patient_inputs: list[str], n_questions: int = 5) -> list[str]:
    mem = PatientMemory()
    patient_msgs: list[str] = []
    asked: list[str] = []
    topics_from_questions: list[str] = []

    if not patient_inputs:
        return []

    patient_msgs.append(patient_inputs[0])
    mem.update_from_patient_text(patient_inputs[0], is_first_message=True)

    def _next_question() -> str | None:
        ctx = build_context_text(patient_msgs)
        topics_covered = set(mem.inferred_topics()) | set(topics_from_questions)
        return pick_next_followup(
            context_text=ctx,
            query_text=mem.as_retrieval_query() or ctx,
            asked_lower={q.lower() for q in asked},
            topics_covered=topics_covered,
            patient_memory=mem,
            k=80,
        )

    for i in range(1, int(n_questions) + 1):
        q = _next_question()
        if not q:
            asked.append("<STOP>")
            break

        asked.append(q)
        topics_from_questions.append(followup_topic(q))

        # Feed the next patient message if provided.
        if i < len(patient_inputs):
            patient_msgs.append(patient_inputs[i])
            mem.update_from_patient_text(patient_inputs[i], is_first_message=False)

    return asked


if __name__ == "__main__":
    # Mirror the user's example.
    inputs = [
        "hello i have fever",
        "what?",
        "yes i have fever and vomiting",
    ]

    qs = simulate(inputs, n_questions=5)
    for i, q in enumerate(qs, start=1):
        print(f"Q{i}: {q}")

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path so imports like `app.*` work even when the
# script is executed from the `scripts/` folder.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.followup_runtime import followup_topic, pick_next_followup
from app.patient_memory import PatientMemory


def run_sequence(ctx: str, *, max_q: int = 10) -> None:
    mem = PatientMemory()
    mem.chief_complaint = "fever headache"
    mem.symptoms = {"fever", "headache", "chills"}

    asked: list[str] = []
    topics: set[str] = set()

    for i in range(max_q):
        q = pick_next_followup(
            context_text=ctx,
            query_text=ctx,
            asked_lower={a.lower() for a in asked},
            asked_questions=list(asked),
            topics_covered=set(topics),
            patient_memory=mem,
            k=80,
        )
        print(f"{i+1}. {q}")
        if not q:
            break
        asked.append(q)
        topics.add(followup_topic(q))


def main() -> int:
    ctx = (
        "I have fever and headache. "
        "No but I took paracetamol. "
        "Yes I have fever and chills."
    )
    run_sequence(ctx, max_q=12)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

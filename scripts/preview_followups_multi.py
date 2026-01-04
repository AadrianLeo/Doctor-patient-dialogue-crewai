from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.patient_memory import PatientMemory
from app.followup_runtime import build_context_text, pick_next_followup, followup_topic


def simulate(patient_inputs: list[str], n_questions: int = 5, k: int = 25) -> list[str]:
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
            k=int(k),
        )

    for i in range(1, int(n_questions) + 1):
        q = _next_question()
        if not q:
            asked.append("<STOP>")
            break

        asked.append(q)
        topics_from_questions.append(followup_topic(q))

        if i < len(patient_inputs):
            patient_msgs.append(patient_inputs[i])
            mem.update_from_patient_text(patient_inputs[i], is_first_message=False)

    return asked


def run_scenarios() -> None:
    scenarios: dict[str, list[str]] = {
        "Fever + vomiting (your case)": [
            "hello i have fever",
            "what?",
            "yes i have fever and vomiting",
        ],
        "Headache": [
            "hi i have a headache",
            "it started yesterday",
        ],
        "Abdominal pain": [
            "i have stomach pain",
            "it is getting worse",
        ],
        "Cough + sore throat": [
            "i have cough and sore throat",
            "for 3 days",
        ],
        "Rash": [
            "i have a rash",
            "it is itchy",
        ],
        "Chest pain": [
            "i have chest pain",
            "it started today",
        ],
        "Shortness of breath": [
            "i feel short of breath",
            "since yesterday",
        ],
        "Dizziness": [
            "i feel dizzy",
            "it started this morning",
        ],
        "Back pain radiating leg": [
            "i have back pain going down my leg",
            "for 1 week",
        ],
        "Urinary symptoms": [
            "it burns when i pee",
            "for 2 days",
        ],
        "Nausea (no fever)": [
            "i feel nauseous",
            "since last night",
        ],
    }

    for name, inputs in scenarios.items():
        print("=")
        print(name)
        print("patient:", inputs)
        qs = simulate(inputs, n_questions=5, k=25)
        for i, q in enumerate(qs, start=1):
            print(f"Q{i}: {q}")


if __name__ == "__main__":
    run_scenarios()

from __future__ import annotations

import re
import sys
from pathlib import Path

# Ensure repo root is on sys.path so imports like `app.*` work even when the
# script is executed from the `scripts/` folder.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.followup_runtime import followup_topic, get_followup_shortlists, missing_required_topics, pick_next_followup
from app.patient_memory import PatientMemory


def _is_constipation_presence_check(q: str) -> bool:
    ql = " ".join(str(q).lower().split())
    return bool(
        re.search(r"\b(do you have|have you got|have you had|any|are you having)\s+constipation\b", ql)
        or re.search(r"\bconstipation\?\s*$", ql)
    )


def _is_presuppositional_endorsing_pain(q: str) -> bool:
    return "you were endorsing" in (q or "").lower()


def _is_immunosuppressive_med_question(q: str) -> bool:
    return "immunosuppress" in " ".join((q or "").lower().split())


def run_case(name: str, *, ctx: str, mem: PatientMemory, topics_covered: set[str]) -> tuple[bool, str | None]:
    missing_req = missing_required_topics(mem, set(topics_covered))
    _missing_agent, shortlist_missing, shortlist_any = get_followup_shortlists(
        context_text=ctx,
        query_text=ctx,
        asked_lower=set(),
        topics_covered=set(topics_covered),
        k=80,
    )

    q = pick_next_followup(
        context_text=ctx,
        query_text=ctx,
        asked_lower=set(),
        asked_questions=[],
        topics_covered=set(topics_covered),
        patient_memory=mem,
        k=80,
    )
    print(f"[{name}] missing_required={missing_req} shortlist_any={len(shortlist_any)} shortlist_missing={len(shortlist_missing)}")
    if name == "fever_headache":
        topic_counts: dict[str, int] = {}
        for qq in shortlist_any:
            t = followup_topic(qq)
            topic_counts[t] = topic_counts.get(t, 0) + 1
        print(f"[{name}] shortlist_any_topics={dict(sorted(topic_counts.items()))}")
    ok = True

    if name == "constipation":
        if q and _is_constipation_presence_check(q):
            ok = False

    if name == "fever_headache":
        if q and _is_presuppositional_endorsing_pain(q):
            ok = False
        if q and _is_immunosuppressive_med_question(q):
            ok = False

    return ok, q


def main() -> int:
    failures: list[str] = []

    # Case 1: constipation already explicitly stated
    ctx1 = "hi doctor i have constipation for 3 days i have hard stools and straining no blood in stool"
    mem1 = PatientMemory()
    mem1.chief_complaint = "constipation"
    mem1.duration = "for 3 days"
    mem1.symptoms = {"constipation"}

    ok1, q1 = run_case("constipation", ctx=ctx1, mem=mem1, topics_covered={"duration"})
    print("[constipation]", q1)
    if not ok1:
        failures.append("constipation: picked redundant constipation presence-check")

    # Case 2: fever + headache; ensure we don't pull presuppositions
    ctx2 = (
        "hello i am having fever and headache. headache for 2 days and fever since yesterday. "
        "yes i took paracetamol. no i am not allergic to any medications"
    )
    mem2 = PatientMemory()
    mem2.chief_complaint = "fever headache"
    mem2.duration = "2 days"
    mem2.severity = "9/10"  # assume already captured earlier in the chat
    mem2.symptoms = {"fever", "headache"}
    mem2.meds = ["paracetamol"]
    mem2.allergies = ["none"]

    ok2, q2 = run_case(
        "fever_headache",
        ctx=ctx2,
        mem=mem2,
        topics_covered={"duration", "severity", "meds", "allergies"},
    )
    print("[fever+headache]", q2)
    if not ok2:
        failures.append(
            "fever_headache: picked unsafe question (presupposition or immunosuppressive-med drift)"
        )

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("-", f)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

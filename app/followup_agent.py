from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable


RetrieveFn = Callable[[str, str, int], list[str]]
TokenizeFn = Callable[[str], set[str]]
AnchorFn = Callable[[str], set[str]]
DomainFn = Callable[[set[str]], set[str]]
TopicFn = Callable[[str], str]
CleanFn = Callable[[str, set[str]], str]
LooksLikeQuestionFn = Callable[[str], bool]
QuestionCtxFn = Callable[[str, str], bool]
QuestionTokensFn = Callable[[str, set[str]], bool]


@dataclass(frozen=True)
class FollowupAgent:
    """Retrieval-only follow-up selection agent.

    This agent does NOT generate new questions. It only:
    - retrieves candidate questions from the dataset,
    - filters them deterministically against the current patient context,
    - and returns shortlists for the UI to pick from (optionally with an LLM ranker).

    The heavy lifting (tokenization, cleaning, safety filters) is injected as callables
    so this module stays Streamlit-free and easy to test.
    """

    retrieve: RetrieveFn
    tokenize: TokenizeFn
    anchor_tokens: AnchorFn
    domain_tokens: DomainFn
    followup_topic: TopicFn
    clean: CleanFn
    looks_like_question: LooksLikeQuestionFn

    is_redundant: QuestionTokensFn
    is_contradictory: QuestionTokensFn
    is_incompatible: QuestionCtxFn

    def get_followup_shortlists(
        self,
        *,
        context_text: str,
        query_text: str,
        asked_lower: set[str],
        topics_covered: set[str],
        k: int = 30,
    ) -> tuple[list[str], list[str], list[str]]:
        """Return (missing_topics, shortlist_for_missing, shortlist_any)."""

        # Base retrieval across all intents.
        candidates = self.retrieve(query_text, "", int(k)) or []
        seed_mode = not candidates

        complaint_tokens = self.tokenize(context_text)
        anchors = self.anchor_tokens(context_text)
        domain = self.domain_tokens(anchors)

        # Slot-level answered signals from raw context (keeps this module independent of PatientMemory).
        ctx_lower = " ".join(str(context_text).lower().split())
        has_measured_temp = bool(
            re.search(r"\b(temp|temperature)\b", ctx_lower)
            and re.search(r"\b\d{2,3}(?:\.\d)?\b", ctx_lower)
        ) or bool(re.search(r"\b\d{2,3}(?:\.\d)?\s*(?:c|f)\b", ctx_lower))

        # If the base retrieval over-focuses on one theme (common with symptom words like
        # "fever"), supplement with a few topic-directed retrievals for missing broad topics.
        # This stays retrieval-only (still pulls from the dataset), but improves coverage.
        # Keep this aligned with the deterministic stop gating in the Streamlit app.
        # Required slots for interview control.
        # NOTE: This list must stay aligned with the deterministic stop gating in the UI/runtime.
        required_topics: tuple[str, ...] = (
            "symptoms",
            "duration",
            "severity",
            "meds",
            "allergies",
        )

        # Build a stable, minimal anchor query for supplement retrieval.
        # Using anchors (instead of the full query text) improves relevance and
        # avoids pulling in unrelated "script"-like questions from hint-only queries.
        anchor_query_parts: list[str] = []
        if anchors:
            anchor_query_parts.extend(sorted(anchors))
        elif domain:
            anchor_query_parts.extend(sorted(domain))
        anchor_query = " ".join(anchor_query_parts[:12]).strip() or query_text

        # Detect whether we have any concrete symptom context yet.
        symptom_hints = {
            "fever",
            "chills",
            "nausea",
            "vomiting",
            "diarrhea",
            "constipation",
            "bowel",
            "stool",
            "cough",
            "headache",
            "rash",
            "breath",
            "breathless",
            "shortness",
            "chest",
            "abdominal",
            "stomach",
            "throat",
            "pain",
        }
        has_symptom_context = bool(complaint_tokens.intersection(symptom_hints)) or bool(anchors.intersection(symptom_hints))

        missing = [t for t in required_topics if t not in topics_covered]
        missing_set = set(missing)

        # Supplement only required topics.
        core_first = ("duration", "severity", "meds", "allergies")

        supplement_topics: list[str] = [t for t in core_first if t in missing]
        if supplement_topics:
            # Dedupe while preserving order.
            seen: set[str] = set()
            merged: list[str] = []
            for q in candidates:
                qn = str(q)
                if qn not in seen:
                    seen.add(qn)
                    merged.append(qn)

            for t in supplement_topics:
                # Keep the per-query retrieval size modest; the dataset retriever
                # can be expensive and this function can run multiple times.
                base_take = min(10, int(k)) if not seed_mode else min(18, int(k))
                # For core topics, retrieve a bit more to avoid premature <STOP>
                # in complaints with sparse matching (e.g., headache).
                if t in {"duration", "severity"}:
                    take = min(int(k), 25) if not seed_mode else min(int(k), 30)
                elif t in {"meds", "allergies"}:
                    take = min(int(k), 15) if not seed_mode else min(int(k), 20)
                else:
                    take = base_take

                intents: list[str] = [t]
                # For fever presentations, "severity" is often asked as measured temperature.
                if t == "severity" and {"fever", "temperature", "temp", "chills", "sweats"}.intersection(complaint_tokens | anchors):
                    intents.append("temp_measurement")

                for intent in intents:
                    try:
                        extra = self.retrieve(anchor_query, intent, take)
                    except Exception:
                        extra = []
                    for q in extra or []:
                        qn = str(q)
                        if qn not in seen:
                            seen.add(qn)
                            merged.append(qn)

            candidates = merged

        def eligible(question_text: str) -> bool:
            topic = self.followup_topic(question_text)

            # HARD RULE: Never ask red-flag questions until required slots are filled.
            if missing_set and topic in {
                "cardioresp_red_flags",
                "neuro_red_flags",
                "rash_red_flags",
                "neuro_back_red_flags",
                "neuro_limb_red_flags",
                "gi_red_flags",
            }:
                return False

            # If the patient is very non-specific (“not feeling well”), avoid jumping to
            # PMH/meds/social/etc. First clarify symptoms/duration/severity.
            if not has_symptom_context and topic in {"pmh", "meds", "allergies", "social", "exposures"}:
                return False
            if topic in {"duration", "severity", "pmh", "meds", "allergies", "social", "exposures"}:
                return True

            # Associated-symptom follow-ups are a common drift source (e.g., headache -> fever/chills).
            # Only allow if the question overlaps the current complaint anchors/domain/context.
            if topic == "associated":
                q_tokens = self.tokenize(question_text)
                if anchors and q_tokens.intersection(anchors):
                    return True
                if domain and q_tokens.intersection(domain):
                    return True
                return len(q_tokens.intersection(complaint_tokens)) > 0

            if topic in {
                "cardioresp_red_flags",
                "neuro_red_flags",
                "rash_red_flags",
                "neuro_back_red_flags",
                "neuro_limb_red_flags",
                "gi_red_flags",
            }:
                q_tokens = self.tokenize(question_text)

                # Always allow if it overlaps anchors/domain.
                if anchors and q_tokens.intersection(anchors):
                    return True
                if domain and q_tokens.intersection(domain):
                    return True

                # Do NOT do broad red-flag screening for mild fever/headache complaints.
                # Only allow red-flag topics when they overlap the patient's own context.

                # Allow rash red flags if the chief complaint is dermatologic.
                if topic == "rash_red_flags" and {
                    "rash",
                    "hives",
                    "urticaria",
                    "itch",
                    "itchy",
                    "itching",
                    "eczema",
                    "skin",
                }.intersection(complaint_tokens):
                    return True

                # Allow GI red flags when the presentation is bowel/abdominal.
                if topic == "gi_red_flags" and {
                    "constipation",
                    "diarrhea",
                    "bowel",
                    "stool",
                    "abdomen",
                    "abdominal",
                    "stomach",
                    "belly",
                }.intersection(complaint_tokens | anchors):
                    return True

                if topic == "neuro_back_red_flags":
                    return bool({"back", "spine", "leg", "legs", "sciatica"}.intersection(complaint_tokens))

                if topic == "neuro_limb_red_flags":
                    return bool(
                        {
                            "hand",
                            "hands",
                            "arm",
                            "arms",
                            "grip",
                            "gripping",
                            "weakness",
                            "numb",
                            "numbness",
                            "tingle",
                            "tingling",
                        }.intersection(complaint_tokens)
                    )

                return False

            q_tokens = self.tokenize(question_text)

            # Prefer anchor overlap.
            if anchors and q_tokens.intersection(anchors):
                return True
            if domain and q_tokens.intersection(domain):
                return True
            return len(q_tokens.intersection(complaint_tokens)) > 0

        def filtered_list(*, require_missing_topic: bool) -> list[str]:
            out: list[str] = []

            for raw_q in candidates:
                q = self.clean(raw_q, complaint_tokens)
                if not q:
                    continue
                q_lower = q.lower()
                if not self.looks_like_question(q):
                    continue
                if q_lower in asked_lower:
                    continue
                if self.is_redundant(q, complaint_tokens):
                    continue
                if self.is_contradictory(q, complaint_tokens):
                    continue
                if self.is_incompatible(q, context_text):
                    continue

                topic = self.followup_topic(q)

                # If we already covered a slot (asked or inferred), don't re-ask it.
                if topic in topics_covered:
                    continue
                # Temperature measurement shouldn't repeat once a temperature is present.
                if topic == "temp_measurement" and has_measured_temp:
                    continue
                if not eligible(q):
                    continue
                if require_missing_topic and (topic not in missing_set):
                    continue
                out.append(q)

            return out

        shortlist_missing = filtered_list(require_missing_topic=True) if missing else []
        shortlist_any = filtered_list(require_missing_topic=False)
        return missing, shortlist_missing, shortlist_any

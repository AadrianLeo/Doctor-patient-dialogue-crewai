import os
import streamlit as st
import sys
from pathlib import Path
import json
import html
from datetime import datetime
import re
from typing import Any

# -----------------------------
# CREWAI TELEMETRY OFF (Streamlit runs in a worker thread)
# -----------------------------
# CrewAI's telemetry registers signal handlers (SIGINT/SIGTERM). Streamlit executes
# the app script in a non-main thread, which can crash on Windows.
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")

# -----------------------------
# PATH FIX (important)
# -----------------------------
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _p in (_SRC, _ROOT):
    p = str(_p)
    if p not in sys.path:
        sys.path.insert(0, p)

from retrieval.subjective_retriever import get_questions_by_intent
from app.patient_memory import PatientMemory
from app.followup_agent import FollowupAgent
from app.followup_runtime import pick_next_followup as _rt_pick_next_followup


st.set_page_config(page_title="Clinical Assistant", layout="wide")


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
    "today",
    "yesterday",
    "tomorrow",
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
    "any",
    "you",
    "your",
    # very common temporal/bridging words that cause accidental overlap
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
    "case",
    "been",
}


_GENERIC_COMPLAINT_TOKENS = {
    "pain",
    "ache",
    "aches",
    "hurt",
    "hurts",
    "sharp",
    "dull",
    "bad",
    "worse",
    "better",
    "feeling",
    "feel",
    "having",
    "have",
    "problem",
    # temporal fillers often present in answers
    "just",
    "recent",
    "recently",
    "past",
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
    "today",
    "yesterday",
}


_COMMON_MED_TOKENS = {
    # allow these even if unmentioned, since they are common OTC/generic terms
    "medication",
    "medications",
    "medicine",
    "medicines",
    "tablet",
    "tablets",
    "drug",
    "drugs",
    "antibiotic",
    "antibiotics",
    "painkiller",
    "painkillers",
    "paracetamol",
    "acetaminophen",
    "ibuprofen",
}


_TOKEN_SYNONYMS: dict[str, set[str]] = {
    # Abdominal variants
    "abdomen": {"abdominal", "belly", "stomach", "tummy"},
    "abdominal": {"abdomen", "belly", "stomach", "tummy"},
    "belly": {"abdomen", "abdominal", "stomach", "tummy"},
    "stomach": {"abdomen", "abdominal", "belly", "tummy"},
    "tummy": {"abdomen", "abdominal", "belly", "stomach"},
}


def _anchor_tokens_from_context(context_text: str) -> set[str]:
    """Return complaint 'anchor' tokens used to keep follow-ups on-topic."""
    toks = _tokenize_for_followups(context_text)
    anchors = {t for t in toks if t not in _GENERIC_COMPLAINT_TOKENS}
    # Keep anchors reasonably specific.
    anchors = {t for t in anchors if len(t) >= 3}
    # If we filtered everything (very short complaints), fall back to tokens.
    return anchors or toks


def _domain_tokens(anchors: set[str]) -> set[str]:
    """Small domain keyword set inferred from anchors (used for red-flag gating)."""
    a = set(anchors)
    domain: set[str] = set()

    # Abdominal / GI / GU
    if {"abdomen", "abdominal", "stomach", "belly", "tummy", "appendix"}.intersection(a):
        domain |= {
            "abdomen",
            "abdominal",
            "stomach",
            "belly",
            "nausea",
            "vomiting",
            "diarrhea",
            "constipation",
            "bowel",
            "stool",
            "urine",
            "urinary",
            "burning",
            "frequency",
        }

    # Headache / neuro
    if {"headache", "migraine", "head"}.intersection(a):
        domain |= {
            "headache",
            "head",
            "vision",
            "neck",
            "stiff",
            "light",
            "nausea",
        }

    # Fever / infection
    if {"fever", "temperature", "chills", "sweats"}.intersection(a):
        domain |= {"fever", "temperature", "chills", "sweats", "cough", "sore", "throat"}

    return domain


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _init_state() -> None:
    st.session_state.setdefault("stage", "collect_initial")
    st.session_state.setdefault("messages", [])  # list[{role,text,ts}]
    # Adaptive follow-ups
    st.session_state.setdefault("followup_questions", [])  # asked/used questions
    st.session_state.setdefault("followup_count", 0)
    # Track broad history topics covered so we can prefer questions that fill gaps.
    st.session_state.setdefault("followup_topics", set())
    st.session_state.setdefault("crew_ran", False)
    st.session_state.setdefault("generated_result_raw", None)
    st.session_state.setdefault("generated_result_json", None)
    st.session_state.setdefault("care_plan_raw", None)
    st.session_state.setdefault("care_plan_json", None)
    st.session_state.setdefault("generated_summary_text", "")

    # Structured patient memory (extractive). Used to build a stable retrieval query
    # and to mark topics as covered when the patient volunteers info (e.g., duration).
    st.session_state.setdefault("patient_memory", PatientMemory().to_dict())


def _now_local() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _append_message(role: str, text: str) -> None:
    st.session_state.messages.append(
        {"role": role, "text": text, "ts": _now_local()}
    )


def _get_patient_memory() -> PatientMemory:
    return PatientMemory.from_dict(st.session_state.get("patient_memory"))


def _update_patient_memory(new_patient_text: str, *, is_first_message: bool) -> None:
    mem = _get_patient_memory()
    mem.update_from_patient_text(new_patient_text, is_first_message=is_first_message)
    st.session_state["patient_memory"] = mem.to_dict()

    # If the patient volunteered info, treat those broad topics as covered.
    topics = set(st.session_state.get("followup_topics", set()))
    topics |= mem.inferred_topics()
    st.session_state["followup_topics"] = topics


def _retrieval_query_text() -> str:
    """Query string used to retrieve dataset questions."""
    mem = _get_patient_memory()
    # Keep follow-up retrieval anchored to the presenting complaint.
    # Broad history snippets (meds/allergies/social/etc) can cause drift into
    # generic screening questions unrelated to the chief complaint.
    q = mem.as_retrieval_query(include_history_snippets=False)
    return q or _patient_context_text()


# Stop condition (deterministic): stop ONLY when required subjective slots are covered.
# Optional/late topics (meds, allergies, PMH, social, exposures, red flags) must NOT
# gate termination.
_REQUIRED_TOPICS: tuple[str, ...] = (
    "symptoms",
    "duration",
    "severity",
    "meds",
    "allergies",
)


def _is_red_flag_topic(topic: str) -> bool:
    t = str(topic or "")
    return t == "red_flags" or t.endswith("_red_flags")


def _required_topics_covered() -> set[str]:
    mem = _get_patient_memory()
    topics_covered = set(st.session_state.get("followup_topics", set()))

    covered: set[str] = set()

    # Required slot: symptoms (free-text complaint + conservative symptom labels).
    if (
        str(getattr(mem, "chief_complaint", "") or "").strip()
        or bool(getattr(mem, "symptoms", None))
        or ("symptoms" in topics_covered)
    ):
        covered.add("symptoms")

    if str(getattr(mem, "duration", "") or "").strip() or ("duration" in topics_covered):
        covered.add("duration")

    symptoms = set(getattr(mem, "symptoms", []) or [])
    feverish = bool({"fever", "temperature"}.intersection(symptoms))
    if (
        str(getattr(mem, "severity", "") or "").strip()
        or ("severity" in topics_covered)
        or (feverish and str(getattr(mem, "temperature", "") or "").strip())
        or (feverish and ("temp_measurement" in topics_covered))
    ):
        covered.add("severity")

    # Required slots: meds + allergies.
    if bool(getattr(mem, "meds", None)) or ("meds" in topics_covered):
        covered.add("meds")
    if bool(getattr(mem, "allergies", None)) or ("allergies" in topics_covered):
        covered.add("allergies")

    return covered


def _missing_required_topics() -> list[str]:
    covered = _required_topics_covered()
    missing = [t for t in _REQUIRED_TOPICS if t not in covered]
    return missing


def _ready_for_summary() -> bool:
    return not _missing_required_topics()


def _build_dialogue_for_crew() -> str:
    def _looks_like_question(text: str) -> bool:
        if "?" in text:
            return True

        cleaned = " ".join(str(text).split()).strip()
        lower = cleaned.lower()

        # Strip common leading fillers.
        fillers = (
            "okay ",
            "alright ",
            "all right ",
            "um ",
            "so ",
            "well ",
            "and ",
            "then ",
            "now ",
        )
        changed = True
        while changed:
            changed = False
            for f in fillers:
                if lower.startswith(f):
                    lower = lower[len(f) :].lstrip()
                    changed = True

        starters = (
            "what ",
            "when ",
            "where ",
            "why ",
            "how ",
            "who ",
            "which ",
            "do you ",
            "did you ",
            "are you ",
            "were you ",
            "have you ",
            "has ",
            "can you ",
            "could you ",
            "would you ",
            "will you ",
            "is it ",
            "are there ",
            "any ",
            "tell me ",
        )
        return lower.startswith(starters)

    lines: list[str] = []

    for m in st.session_state.messages:
        role = m.get("role")
        if role in ("doctor", "patient"):
            text = str(m.get("text", "")).strip()
            if text:
                # Avoid passing clinician questions into the Crew run. This keeps
                # evidence grounded in patient statements and clinician factual
                # observations rather than follow-up prompts.
                if role == "doctor" and _looks_like_question(text):
                    continue
                lines.append(f"[{role}] {text}")

    return "\n".join(lines)


def _parse_pasted_dialogue(raw: str) -> list[dict[str, str]]:
    """Parse a pasted transcript into [{role,text},...].

    Expected format is dataset-style tagged lines, e.g.:
    [doctor] ...\n[patient] ...
    """

    text = str(raw or "").strip()
    if not text:
        return []

    # Users often copy from JSON where newlines are escaped (e.g. "\\n").
    # If the whole payload is a JSON string, decode it first.
    if text.startswith('"') and text.endswith('"'):
        try:
            decoded = json.loads(text)
            if isinstance(decoded, str):
                text = decoded
        except Exception:
            pass

    # If it still contains literal "\\n" and no real newlines, unescape them.
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out: list[dict[str, str]] = []

    tag_rx = re.compile(r"^\s*\[(doctor|patient)\]\s*(.+?)\s*$", flags=re.IGNORECASE)
    alt_rx = re.compile(r"^\s*(doctor|patient)\s*:\s*(.+?)\s*$", flags=re.IGNORECASE)

    for ln in lines:
        m = tag_rx.match(ln) or alt_rx.match(ln)
        if not m:
            continue
        role = m.group(1).strip().lower()
        msg = m.group(2).strip()
        if role not in {"doctor", "patient"} or not msg:
            continue
        out.append({"role": role, "text": msg})

    return out


def _load_dialogue_into_state(raw: str) -> None:
    parsed = _parse_pasted_dialogue(raw)
    if not parsed:
        raise ValueError("No tagged lines found. Use lines like '[doctor] ...' and '[patient] ...'.")

    st.session_state.messages = parsed

    # Reset generation-related state.
    st.session_state.generated_summary_text = ""
    st.session_state.generated_result_raw = None
    st.session_state.generated_result_json = None
    st.session_state.care_plan_json = None
    st.session_state.crew_ran = False

    # Reset follow-up / slot-tracking state (keeps UI consistent).
    st.session_state.followup_questions = []
    st.session_state.followup_count = 0
    st.session_state.followup_topics = set()
    st.session_state.asked_required_slots = set()
    st.session_state["last_followup_reason"] = None

    # Best-effort rebuild deterministic memory from patient turns.
    mem = PatientMemory()
    first = True
    for m in parsed:
        if m.get("role") == "patient":
            mem.update_from_patient_text(m.get("text", ""), is_first_message=first)
            first = False
    st.session_state["patient_memory"] = mem.to_dict()
    st.session_state["followup_topics"] = set(mem.inferred_topics())

    # Enable summary generation immediately.
    st.session_state.stage = "ready"


def _tokenize_for_followups(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9]+", str(text).lower()))
    out: set[str] = set()
    for w in words:
        if not w or w.isdigit():
            continue
        if w in _STOPWORDS:
            continue
        out.add(w)
        if w in _TOKEN_SYNONYMS:
            out.update(_TOKEN_SYNONYMS[w])
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            out.add(w[:-1])
    return out


def _looks_like_question_for_followups(text: str) -> bool:
    if not text:
        return False
    if "?" in text:
        return True

    cleaned = " ".join(str(text).split()).strip()
    lower = cleaned.lower()

    # Strip common leading fillers.
    fillers = (
        "okay ",
        "alright ",
        "all right ",
        "um ",
        "so ",
        "well ",
        "and ",
        "then ",
        "now ",
    )
    changed = True
    while changed:
        changed = False
        for f in fillers:
            if lower.startswith(f):
                lower = lower[len(f) :].lstrip()
                changed = True

    starters = (
        "what ",
        "when ",
        "where ",
        "why ",
        "how ",
        "who ",
        "which ",
        "do you ",
        "did you ",
        "are you ",
        "were you ",
        "have you ",
        "can you ",
        "could you ",
        "would you ",
        "will you ",
        "is it ",
        "are there ",
        "any ",
        "tell me ",
    )
    if lower.startswith(starters):
        return True

    # Reject long directive/plan text that isn't a question.
    if len(lower.split()) >= 18 and not lower.endswith("?"):
        return False

    return False


def _patient_context_text() -> str:
    # Use only patient utterances as the evolving symptom context.
    parts = []
    for m in st.session_state.messages:
        if m.get("role") == "patient":
            t = str(m.get("text", "")).strip()
            if t:
                parts.append(t)
    return " ".join(parts)


def _clean_followup_question(question: str, complaint_tokens: set[str]) -> str:
    q = " ".join(str(question).split()).strip()
    if not q:
        return ""

    # Normalize common ASR/punctuation artifacts at the start.
    q = re.sub(r"^[\s\.,;:!\-\u2013\u2014\"'`]+", "", q).strip()

    # Strip leading filler, even if preceded by punctuation.
    while True:
        before = q
        q = re.sub(
            r"^[\s\.,;:!\-\u2013\u2014]*\b(?:okay|alright|all right|um|uh|so|well|right)\b[\s\.,;:!\-\u2013\u2014]*",
            "",
            q,
            flags=re.IGNORECASE,
        ).strip()
        q = re.sub(r"^[\s\.,;:!\-\u2013\u2014]*\band\b[\s\.,;:!\-\u2013\u2014]*", "", q, flags=re.IGNORECASE).strip()
        if q == before:
            break

    # Remove early hesitations like ", uh," / ", um," that frequently occur in transcripts.
    q = re.sub(r"\b,\s*(?:uh|um)\s*,\b", ", ", q, flags=re.IGNORECASE)
    q = re.sub(r"\b(?:uh|um)\b\s*,", "", q, flags=re.IGNORECASE)
    q = re.sub(r",\s*,+", ", ", q).strip()

    # Remove context-dependent tails like "you said/mentioned ...".
    m = re.search(r"(?:\bor\b\s+)?(?:you\s+know\s+)?you\s+(?:said|mentioned)\b", q, flags=re.IGNORECASE)
    if m:
        q = q[: m.start()].rstrip(" ,;:-")
        q = re.sub(r"\b(or|and)\b\s*$", "", q, flags=re.IGNORECASE).rstrip(" ,;:-")

    # Avoid presuppositional "along with the <symptom>" when not in the complaint.
    def _along_with_repl(match: re.Match[str]) -> str:
        tok = (match.group(1) or "").lower().strip()
        tok_norm = tok[:-1] if (len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss")) else tok
        if tok_norm and tok_norm not in complaint_tokens:
            return "along with that"
        return match.group(0)

    q = re.sub(r"\balong with (?:the|your)\s+([a-z0-9]+)\b", _along_with_repl, q, flags=re.IGNORECASE)

    return " ".join(q.split()).strip()


def _is_redundant_followup(question: str, complaint_tokens: set[str]) -> bool:
    """Drop follow-ups that directly re-ask the chief complaint."""
    q_tokens = _tokenize_for_followups(question)

    q_lower = " ".join(str(question).lower().split())

    def _is_presence_check(symptom: str) -> bool:
        # Presence checks are redundant if the symptom is already stated.
        # Keep detail questions (e.g., "how high is your fever").
        if symptom == "fever":
            return bool(
                re.search(
                    r"\b(do you have|have you got|have you had|any)\s+(a\s+)?fever\b", q_lower
                )
                or re.search(r"\bfever\?\s*$", q_lower)
            )
        if symptom == "headache":
            return bool(
                re.search(
                    r"\b(do you have|have you got|have you had|any)\s+headaches?\b", q_lower
                )
                or re.search(r"\bheadaches?\?\s*$", q_lower)
            )
        return False

    # If the patient already stated fever, don't re-ask presence of fever.
    if "fever" in complaint_tokens and "fever" in q_tokens and _is_presence_check("fever"):
        return True

    # If the patient hasn't mentioned headache, avoid *history/comparison* headache questions,
    # but allow simple associated-symptom screening ("any headaches?").
    if "headache" not in complaint_tokens and "headache" in q_tokens:
        if re.search(r"\b(before|previous|like this|usual|normally)\b", q_lower):
            return True

    # Symptom checklist re-asks: if a question only screens for symptoms the patient already
    # confirmed, drop it. This prevents loops like:
    #   patient: "nausea and vomiting" -> doctor: "any nausea or vomiting?"
    symptom_hints = {
        "fever",
        "chills",
        "nausea",
        "vomiting",
        "diarrhea",
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
    }
    q_symptoms = q_tokens.intersection(symptom_hints)
    if q_symptoms:
        # Only treat as a redundant presence screen if it's phrased like a screen/check.
        if re.search(r"\b(any|do you have|have you got|have you had|are you having)\b", q_lower) or "," in q_lower:
            known = complaint_tokens.intersection(symptom_hints)
            if q_symptoms.issubset(known):
                return True

    return False


def _is_contradictory_followup(question: str, complaint_tokens: set[str]) -> bool:
    """Reject follow-ups that contradict symptoms already stated by the patient."""
    q_lower = " ".join(str(question).lower().split())

    # Example: patient says "headache", then asking "no headaches?" is contradictory.
    if "headache" in complaint_tokens and re.search(r"\bno\s+headaches?\b", q_lower):
        return True

    if "fever" in complaint_tokens and re.search(r"\bno\s+fever\b", q_lower):
        return True

    return False


def _followup_topic(question: str) -> str:
    q = " ".join(str(question).lower().split())

    if re.search(r"\b(how long|when did|started|since when|duration)\b", q):
        return "duration"

    # Temperature measurement is a common way to assess fever severity.
    if re.search(r"\b(temperature|temp)\b", q) and re.search(r"\b(measure|measured|check|checked|taken|take)\b", q):
        return "temp_measurement"
    if re.search(r"\b(how high|how hot)\b", q) and re.search(r"\b(temperature|temp|fever)\b", q):
        return "temp_measurement"

    if re.search(r"\b(how (bad|severe)|severity|rate (it|your)|scale)\b", q):
        return "severity"
    # General severity/wellbeing phrasing commonly used in the dataset.
    if re.search(r"\bhow are you feeling\b", q) or re.search(r"\b(feel|feeling)\s+(better|worse)\b", q):
        return "severity"
    if re.search(r"\b(any other|associated|along with|else)\b", q):
        return "associated"
    if re.search(r"\b(allerg|reaction)\b", q):
        return "allergies"
    if re.search(
        r"\b(medication|medicines|tablets|paracetamol|acetaminophen|ibuprofen|antibiotic|painkillers?)\b",
        q,
    ):
        return "meds"
    if re.search(r"\b(past|history|medical|strep|asthma|diabetes|heart|kidney)\b", q):
        return "pmh"
    if re.search(r"\b(work|job|student|smok|alcohol|drugs)\b", q):
        return "social"
    if re.search(r"\b(contact|sick|colleague|travel|exposure)\b", q):
        return "exposures"
    # Red flags: split into groups so we can gate relevance by complaint domain.
    if re.search(
        r"\b(chest pain|shortness of breath|short of breath|breathless|difficulty breathing|trouble breathing)\b",
        q,
    ):
        return "cardioresp_red_flags"
    if re.search(r"\b(confusion|faint|seizure|neck (?:pain|stiff|stiffness)|vision)\b", q):
        return "neuro_red_flags"
    if re.search(r"\b(numb(?:ing)?|tingl(?:ing|e)|weakness)\b", q) and re.search(
        r"\b(hand|hands|arm|arms|grip|gripping)\b", q
    ):
        return "neuro_limb_red_flags"
    if re.search(r"\b(numb(?:ing)?|tingl(?:ing|e)|weakness)\b", q) and re.search(
        r"\b(leg|legs|back|spine|down your leg)\b", q
    ):
        return "neuro_back_red_flags"
    if re.search(r"\b(rash)\b", q):
        return "rash_red_flags"

    # Multi-symptom checklist questions are effectively "associated symptoms" screens.
    # Example: "nausea, vomiting, diarrhea?" or "chills, fever, nausea, vomiting?"
    q_tokens = _tokenize_for_followups(q)
    symptom_hints = {
        "fever",
        "chills",
        "nausea",
        "vomiting",
        "diarrhea",
        "cough",
        "headache",
        "rash",
        "breath",
        "breathless",
        "shortness",
        "chest",
        "pain",
        "abdominal",
        "stomach",
        "throat",
    }
    if len(q_tokens.intersection(symptom_hints)) >= 2 and "?" in q:
        return "associated"
    return "symptom_detail"


def _is_incompatible_followup(question: str, context_text: str) -> bool:
    """Reject follow-ups that appear tied to a different patient/context.

    This app is retrieval-only: we cannot invent better questions, so we must
    conservatively filter out items that presuppose a different identity,
    pre-existing care plan, procedures, insurance, or unrelated chronic issues
    unless the patient has already brought them up.
    """

    q = " ".join(str(question).strip().split())
    if not q:
        return True

    q_lower = q.lower()
    ctx_tokens = _tokenize_for_followups(context_text)
    topic = _followup_topic(q)

    # Block scripted presuppositions like:
    #   "i know that you were endorsing the back pain. any other symptoms? ..."
    # unless the endorsed pain location is actually present in patient context.
    if "you were endorsing" in q_lower:
        for loc in re.findall(r"you were endorsing (?:the )?([a-z]+) pain", q_lower):
            if loc and loc not in ctx_tokens:
                return True

    # Procedure/surgery presuppositions: block specific procedures unless context includes them.
    # This catches cases like "previous surgical history of a colectomy" when the patient only says
    # something generic ("not feeling well").
    procedure_words = set(re.findall(r"\b[a-z]+(?:ectomy|ostomy|oscopy)\b", q_lower))
    if procedure_words and not procedure_words.intersection(ctx_tokens):
        return True

    if "colectomy" in q_lower and "colectomy" not in ctx_tokens:
        return True

    # Greetings / small-talk (not clinical data collection).
    if re.search(r"\b(hey|hello|hi)\b", q_lower) or re.search(r"\bhow are you( today)?\b", q_lower):
        return True

    # Conversational openers/closers that don't collect clinical info.
    conversational_phrases = (
        "are you ready to get started",
        "ready to get started",
        "have a great day",
        "you're welcome",
        "you are welcome",
        "let us know if you need",
        "anything else we can discuss",
        "any questions or anything else",
        "any other questions",
        "anything else going on",
        "thanks for coming",
        "thanks for your time",
    )
    for phrase in conversational_phrases:
        if phrase in q_lower:
            return True

    # Presupposes a prior visit/conversation.
    prior_context_phrases = (
        "since the last time we spoke",
        "last time we spoke",
        "since we last spoke",
        "last time we talked",
        "since we talked",
        "as we discussed",
        "as we talked about",
    )
    for phrase in prior_context_phrases:
        if phrase in q_lower:
            return True

    # Presupposes an external actor (“they”) without any mention of clinicians/hospital/etc.
    # Prevents confusing prompts like “did they put you on medication?” when nobody was introduced.
    if re.search(r"\b(did they|have they|are they|were they)\b", q_lower):
        if not ({"doctor", "doctors", "gp", "hospital", "clinic", "nurse", "ambulance", "er", "a&e", "emergency"}.intersection(ctx_tokens)):
            return True

    # Chronicity presupposition: avoid "past X years" unless patient context suggests chronic duration.
    if re.search(r"\b(past|over the past)\b", q_lower) and re.search(r"\byear|years\b", q_lower):
        ctx_lower = str(context_text).lower()
        if not re.search(r"\byear|years|for years|long time|chronic\b", ctx_lower):
            return True
    if re.search(r"\b(two|2) years\b", q_lower):
        ctx_lower = str(context_text).lower()
        if not re.search(r"\byear|years|for years|long time|chronic\b", ctx_lower):
            return True

    # Avoid presuppositional pain-location questions when pain wasn't mentioned.
    if re.search(r"\b(the )?pain\b", q_lower):
        if not ({"pain", "ache", "aches", "hurt", "hurts"}.intersection(ctx_tokens)):
            return True

    # Exertion/walking presuppositions: avoid "started when you were walking" style
    # questions unless the patient has mentioned exertion-related symptoms.
    if re.search(r"\b(walking|walk|stairs?|exercise|exertion|exertional)\b", q_lower):
        if not ({"walk", "walking", "stairs", "stair", "exercise", "exertion"}.intersection(ctx_tokens)):
            exertion_relevant = {"breath", "breathless", "shortness", "chest", "wheeze", "palpitations", "dizzy", "faint"}
            if not exertion_relevant.intersection(ctx_tokens):
                return True

    # Avoid condition-specific questions (e.g., acne) unless in context.
    if "acne" in q_lower and "acne" not in ctx_tokens:
        return True

    # Specific name/addressing (e.g., 'miss miller').
    m = re.search(r"\b(mr|mrs|ms|miss)\s+([a-z]{2,})\b", q_lower)
    if m:
        surname = m.group(2)
        if surname and surname not in ctx_tokens:
            return True

    # Very context-specific statements that imply an existing plan/history.
    hard_triggers = {
        "colonoscopy",
        "colectomy",
        "medicare",
        "insurance",
        "private insurance",
        "knee surgery",
        "surgery",
        "surgical",
        "due for",
        "still due",
        "once we get",
        "we should think about",
        "here locally",
        "iron supplement",
        "supplement",
        "heart burn",
        "heartburn",
        "reflux",
        "garden",
        "gardener",
        "yard",
        "back surgery",
        "back surgeries",
        "back problems",
        "congestive heart failure",
        "congestive heart failure",
        "heart failure",
        "chf",
        "lasix",
        "furosemide",
        "toprol",
        "metoprolol",
        "lisinopril",
        "crestor",
        "watch your salt",
        "salt intake",
        "weigh yourself",
        "get a scale",
        "retaining",
        "fluid",
    }
    for phrase in hard_triggers:
        if phrase in q_lower:
            # Allow only if the patient already mentioned the key token.
            phrase_tokens = _tokenize_for_followups(phrase)
            if phrase_tokens and not phrase_tokens.intersection(ctx_tokens):
                return True

    # Topic-specific "overly specific" medication prompts: allow generic med questions,
    # but block named/rare meds if not present in patient context.
    if topic == "meds":
        q_tokens = _tokenize_for_followups(q_lower)
        # If it contains a long, specific token not in context and not a common generic med term,
        # it's likely a named medication or condition-specific treatment.
        for tok in q_tokens:
            if tok in _COMMON_MED_TOKENS:
                continue
            if len(tok) >= 9 and tok not in ctx_tokens:
                return True

    # Family-history / PMH: avoid GI-cancer/liver presuppositions unless those tokens exist.
    if topic == "pmh":
        if {"gi", "gastro", "liver", "cancer", "cirrhosis", "hepatitis"}.intersection(
            _tokenize_for_followups(q_lower)
        ) and not ({"gi", "gastro", "liver", "cancer", "hepatitis"}.intersection(ctx_tokens)):
            return True

    # "Loss of bowel or bladder function" is typically a back/cauda equina screen.
    # Avoid asking it unless bowel/bladder/back is already in context.
    if "loss of bowel" in q_lower or "loss of bladder" in q_lower or "bowel or bladder function" in q_lower:
        if not ({"bowel", "bladder", "back", "spine", "leg"}.intersection(ctx_tokens)):
            return True

    return False


def _get_followup_shortlists(
    context_text: str,
    *,
    query_text: str,
    asked_lower: set[str],
    topics_covered: set[str],
    k: int = 80,
) -> tuple[list[str], list[str], list[str]]:
    """Return (missing_topics, shortlist_for_missing, shortlist_any).

    Retrieval-only: derived solely from dataset questions + deterministic filtering.
    Implemented via FollowupAgent to keep selection logic encapsulated.
    """

    agent = FollowupAgent(
        retrieve=lambda q, intent, kk: get_questions_by_intent(q, intent=str(intent), k=int(kk)),
        tokenize=_tokenize_for_followups,
        anchor_tokens=_anchor_tokens_from_context,
        domain_tokens=_domain_tokens,
        followup_topic=_followup_topic,
        clean=_clean_followup_question,
        looks_like_question=_looks_like_question_for_followups,
        is_redundant=_is_redundant_followup,
        is_contradictory=_is_contradictory_followup,
        is_incompatible=_is_incompatible_followup,
    )

    return agent.get_followup_shortlists(
        context_text=context_text,
        query_text=query_text,
        asked_lower=asked_lower,
        topics_covered=topics_covered,
        k=int(k),
    )


def _pick_next_followup(context_text: str, asked_lower: set[str], topics_covered: set[str]) -> str | None:
    mem = _get_patient_memory()
    return _rt_pick_next_followup(
        context_text=context_text,
        query_text=_retrieval_query_text(),
        asked_lower=asked_lower,
        asked_questions=list(st.session_state.get("followup_questions", [])),
        topics_covered=topics_covered,
        patient_memory=mem,
        asked_required_slots=set(st.session_state.get("asked_required_slots", set())),
        k=80,
    )


def _followup_reason(*, next_q: str | None, topics_covered: set[str]) -> dict:
    return {
        "retrieval_query": _retrieval_query_text(),
        "asked_count": int(st.session_state.get("followup_count", 0) or 0),
        "chosen_question": next_q,
    }


def _render_summary_text(assessment_plan: dict | None) -> str:
    """Render a human-readable consultation note (template-style).

    This is separate from the strict JSON returned by the Crew run.
    It must not introduce new diagnoses/inferences; it only reflects collected data.
    """
    
    def _ensure_dict(obj) -> dict:
        """Convert string JSON to dict, or return empty dict if needed."""
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            try:
                parsed = json.loads(obj)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    messages = list(st.session_state.messages)
    patient_msgs = [m for m in messages if m.get("role") == "patient"]
    presenting = (patient_msgs[0].get("text", "").strip() if patient_msgs else "")

    # Build Q/A pairs from the transcript (includes clinician questions).
    qa_pairs: list[tuple[str, str]] = []
    pending_q: str | None = None
    for m in messages:
        role = m.get("role")
        t = str(m.get("text", "")).strip()
        if not t:
            continue
        if role == "doctor":
            pending_q = t
        elif role == "patient" and pending_q:
            qa_pairs.append((pending_q, t))
            pending_q = None

    def _first_answer_matching(pattern: str) -> str:
        rx = re.compile(pattern, flags=re.IGNORECASE)
        for q, a in qa_pairs:
            if rx.search(q):
                return a
        return ""

    duration = _first_answer_matching(r"\bhow long\b|\bwhen did\b|\bstarted\b|\bsince when\b")
    severity = _first_answer_matching(r"\bhow severe\b|\bseverity\b|\bhow bad\b|\bintensity\b")

    pmh_items: list[str] = []
    for q, a in qa_pairs:
        if re.search(r"\b(strep throat|past medical|medical history|in the past)\b", q, flags=re.IGNORECASE):
            pmh_items.append(f"{q} — {a}")

    drug_items: list[str] = []
    for q, a in qa_pairs:
        if re.search(r"\b(medication|medicines|tablets|paracetamol|acetaminophen|ibuprofen|antibiotic|painkillers?)\b", q, flags=re.IGNORECASE):
            drug_items.append(f"{q} — {a}")

    social_items: list[str] = []
    for q, a in qa_pairs:
        if re.search(r"\b(what do you do for work|work|job|student)\b", q, flags=re.IGNORECASE):
            social_items.append(f"{q} — {a}")

    concerns = _first_answer_matching(r"\b(concern|worried|expect|hoping)\b")

    lines: list[str] = []
    lines.append("Consultation Summary")
    lines.append("")

    lines.append("Presenting Complaint")
    if presenting:
        lines.append(f"The patient reported: \"{presenting}\".")
    else:
        lines.append("Not recorded.")
    lines.append("")

    lines.append("History of Presenting Complaint")
    hpc_parts: list[str] = []
    if duration:
        # Clean up duration text - avoid verbatim patient quotes
        duration_clean = duration.strip()
        if not duration_clean.lower().startswith(("for", "since", "about")):
            duration_clean = f"for {duration_clean}"
        hpc_parts.append(f"The symptoms have been present {duration_clean}.")
    if severity:
        hpc_parts.append(f"The patient describes the severity as {severity}.")
    
    # Extract additional relevant details from subjective data
    subj = _ensure_dict(st.session_state.get("subjective"))
    additional = str(subj.get("additional_notes", "")).strip()
    if additional and additional.lower() != "none":
        # Clean up and add additional context
        add_clean = additional.replace("\n", " ").strip()
        if add_clean:
            hpc_parts.append(add_clean)
    
    if hpc_parts:
        for part in hpc_parts:
            lines.append(part)
    else:
        lines.append("No additional details discussed.")
    lines.append("")

    lines.append("Past Medical History Discussed")
    if pmh_items:
        for it in pmh_items:
            lines.append(f"- {it}")
    else:
        lines.append("Not discussed.")
    lines.append("")

    lines.append("Drug History Discussed")
    # Use extracted medication and allergy data from tools
    hist_subj = _ensure_dict(st.session_state.get("history_subjective"))
    meds_list = hist_subj.get("medications") or []
    allergies_list = hist_subj.get("allergies") or []
    
    has_drug_info = False
    if meds_list and any(str(m).strip().lower() not in {"", "none", "nil"} for m in meds_list):
        lines.append("Medications:")
        for med in meds_list:
            med_str = str(med).strip()
            if med_str and med_str.lower() not in {"none", "nil"}:
                lines.append(f"- {med_str}")
                has_drug_info = True
    
    if allergies_list and any(str(a).strip().lower() not in {"", "none", "nil", "no known drug allergies", "nkda"} for a in allergies_list):
        lines.append("Allergies:")
        for allergy in allergies_list:
            allergy_str = str(allergy).strip()
            if allergy_str and allergy_str.lower() not in {"none", "nil", "no known drug allergies", "nkda"}:
                lines.append(f"- {allergy_str}")
                has_drug_info = True
    elif allergies_list:
        # Patient was asked and has no allergies
        lines.append("No known drug allergies.")
        has_drug_info = True
    
    if not has_drug_info:
        lines.append("Not discussed.")
    lines.append("")

    lines.append("Concerns and expectations")
    if concerns:
        lines.append(f"The patient reported: \"{concerns}\".")
    else:
        lines.append("Not discussed.")
    lines.append("")

    lines.append("Investigations")
    lines.append("Not recorded.")
    lines.append("")

    lines.append("Treatment")
    lines.append("Not discussed.")
    lines.append("")

    lines.append("Advice")
    lines.append("Not discussed.")
    lines.append("")

    # Keep the strict JSON result visible elsewhere in the UI; do not rephrase it here.
    return "\n".join(lines).strip() + "\n"


def _reset_consultation() -> None:
    keys_to_keep = {
        # patient + vitals fields can remain if desired; keep nothing by default
    }
    for k in list(st.session_state.keys()):
        if k not in keys_to_keep:
            del st.session_state[k]
    _init_state()


_init_state()

# -----------------------------
# HEADER + DISCLAIMER
# -----------------------------
st.title("Clinical Assistant")

disclaimer = _read_text(Path(__file__).resolve().parents[1] / "knowledge" / "disclaimer.txt")
if disclaimer:
    st.info(disclaimer)
else:
    st.info(
        "Before you start, you must inform the patient you are using a tool to help transcribe and summarise the consultation to assist with note-taking."
    )


# -----------------------------
# MAIN LAYOUT (Chat left, Progress right)
# -----------------------------
left_col, right_col = st.columns([7, 4], gap="large")

with left_col:
    chat_tab, dialogue_tab, summary_tab = st.tabs(["Chat", "Dialogue", "Summary"])

    with chat_tab:
        # Chat transcript (scrollable box)
        st.markdown(
            """
<style>
.dpd-chatbox {
  height: 520px;
  overflow-y: auto;
  border: 1px solid color-mix(in srgb, var(--text-color) 15%, transparent);
  background: var(--secondary-background-color);
  border-radius: 8px;
  padding: 12px;
}
.dpd-row { margin: 0 0 10px 0; }
.dpd-role { font-weight: 600; margin-right: 8px; }
.dpd-text { white-space: pre-wrap; word-break: break-word; }
</style>
            """,
            unsafe_allow_html=True,
        )

        def _render_transcript_html() -> str:
            rows: list[str] = []
            for m in st.session_state.messages:
                role = str(m.get("role") or "").strip().lower()
                text = str(m.get("text") or "").strip()
                if not text:
                    continue
                role_label = "D" if role == "doctor" else "P" if role == "patient" else role[:1].upper() or "?"
                rows.append(
                    '<div class="dpd-row">'
                    f'<span class="dpd-role">{html.escape(role_label)}</span>'
                    f'<span class="dpd-text">{html.escape(text)}</span>'
                    "</div>"
                )
            return "<div class=\"dpd-chatbox\">" + "".join(rows) + "</div>"

        st.markdown(_render_transcript_html(), unsafe_allow_html=True)

        # Explainability (deterministic): what we know / what's missing / why this question.
        last_reason = st.session_state.get("last_followup_reason")
        if last_reason:
            with st.expander("Why this question?", expanded=False):
                st.json(last_reason)

        # Stage machine
        if st.session_state.stage == "collect_initial":
            if not st.session_state.messages:
                _append_message("doctor", "Please describe your main problem.")
                st.rerun()

            user_text = st.chat_input("Type your answer...")
            if user_text:
                _append_message("patient", user_text)

                st.session_state.followup_questions = []
                st.session_state.followup_count = 0
                st.session_state.followup_topics = set()

                _update_patient_memory(user_text, is_first_message=True)

                context_text = _patient_context_text()
                asked_lower = set()
                next_q = _pick_next_followup(
                    context_text,
                    asked_lower=asked_lower,
                    topics_covered=set(st.session_state.get("followup_topics", set())),
                )
                if not next_q:
                    st.session_state.stage = "ready"
                    _append_message(
                        "doctor",
                        "I can't retrieve any more suitable follow-up questions. Ready to summarise.",
                    )
                    st.rerun()

                st.session_state.followup_questions = [next_q]
                st.session_state.followup_count = 1
                st.session_state.stage = "collect_followups"
                st.session_state["last_followup_reason"] = _followup_reason(
                    next_q=next_q,
                    topics_covered=set(st.session_state.get("followup_topics", set())),
                )
                _append_message("doctor", next_q)
                st.rerun()

        elif st.session_state.stage == "collect_followups":
            user_text = st.chat_input("Type your answer...")
            if user_text:
                _append_message("patient", user_text)

                _update_patient_memory(user_text, is_first_message=False)

                context_text = _patient_context_text()
                asked_lower = {str(q).lower() for q in st.session_state.followup_questions}

                next_q = _pick_next_followup(
                    context_text,
                    asked_lower=asked_lower,
                    topics_covered=set(st.session_state.get("followup_topics", set())),
                )

                if next_q:
                    st.session_state.followup_questions.append(next_q)
                    st.session_state.followup_count += 1
                    st.session_state["last_followup_reason"] = _followup_reason(
                        next_q=next_q,
                        topics_covered=set(st.session_state.get("followup_topics", set())),
                    )
                    _append_message("doctor", next_q)
                    st.rerun()
                st.session_state.stage = "ready"
                _append_message("doctor", "I can't retrieve any more suitable follow-up questions. Ready to summarise.")
                st.rerun()

    with dialogue_tab:
        st.markdown(
            "Paste a full tagged transcript, click **Load dialogue**, then click **Generate summary** on the right."
        )

        default_text = st.session_state.get("pasted_dialogue", "")
        pasted = st.text_area(
            "Dialogue (tagged lines)",
            value=default_text,
            height=320,
            placeholder="[doctor] Hello\n[patient] I have fever and headache\n[doctor] Have you taken any medication?\n[patient] Yes, paracetamol\n",
        )
        st.session_state["pasted_dialogue"] = pasted

        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Load dialogue", type="primary", use_container_width=True):
                try:
                    _load_dialogue_into_state(pasted)
                    st.success("Dialogue loaded. Ready to generate summary.")
                except Exception as e:
                    st.error(str(e))
                st.rerun()
        with col_b:
            if st.button("Clear", type="secondary", use_container_width=True):
                st.session_state["pasted_dialogue"] = ""
                st.rerun()

    with summary_tab:
        if st.session_state.stage != "generated":
            st.info("Summary not generated yet. Use the panel on the right to generate it.")
        else:
            st.subheader("Review Consultation Summary")

            st.session_state.generated_summary_text = st.text_area(
                "Consultation Summary",
                value=st.session_state.generated_summary_text,
                height=380,
            )

            # Care Plan (dataset-first; LLM fallback only when needed)
            st.subheader("📝 Care Plan")
            care_plan = st.session_state.get("care_plan_json")
            if isinstance(care_plan, str):
                try:
                    care_plan = json.loads(care_plan)
                except Exception:
                    care_plan = None

            if isinstance(care_plan, dict):
                plan_type = str(care_plan.get("plan_type") or "").strip().lower()
                source_case_id = str(care_plan.get("source_case_id") or "").strip()
                plan_text = str(care_plan.get("plan_text") or "").strip()
                disclaimer_text = str(care_plan.get("disclaimer") or "").strip()

                if plan_type == "dataset":
                    st.success("Dataset-based plan (auditable)")
                    if source_case_id:
                        st.caption(f"Source case: {source_case_id}")
                elif plan_type == "llm_fallback":
                    st.warning("LLM fallback plan (use with clinician review)")
                    if disclaimer_text:
                        st.info(disclaimer_text)
                else:
                    st.caption("Care plan source not specified.")

                if plan_text:
                    st.text_area("Care Plan", value=plan_text, height=180)
                else:
                    st.caption("No care plan text available.")
            else:
                st.caption("Care plan not generated.")

            st.subheader("Assessment & Plan (strict JSON)")
            if st.session_state.generated_result_json is not None:
                st.json(st.session_state.generated_result_json)
            else:
                st.code(str(st.session_state.generated_result_raw), language="json")

with right_col:
    st.header("Summarise consultation")

    # Live progress
    done_followups = st.session_state.stage in ("ready", "generating", "generated")
    is_generating = st.session_state.stage == "generating"
    is_generated = st.session_state.stage == "generated"

    st.markdown(
        "\n".join(
            [
                "**Progress**",
                f"- {'✅' if done_followups else '⬜'} Q&A complete",
                f"- {'✅' if is_generating else '⬜'} Summary generating",
                f"- {'✅' if is_generated else '⬜'} Summary generated",
            ]
        )
    )

    st.divider()

    if st.button("Discard", type="secondary", use_container_width=True):
        _reset_consultation()
        st.rerun()

    if st.button("Start new conversation", type="primary", use_container_width=True):
        _reset_consultation()
        st.rerun()

    st.divider()

    # Generate controls (CrewAI runs once per consultation)
    can_generate = st.session_state.stage == "ready"
    st.button(
        "Generate summary",
        type="primary",
        disabled=not can_generate or st.session_state.stage == "generating",
        use_container_width=True,
        on_click=lambda: st.session_state.update({"stage": "generating"}),
    )

    show_transcript = st.checkbox("Show transcription", value=False)
    if show_transcript:
        st.text_area(
            "Transcription",
            value="\n".join(f"[{m['role']}] {m['text']}" for m in st.session_state.messages),
            height=200,
        )

    # Generation step (CrewAI called exactly once)
    if st.session_state.stage == "generating" and not st.session_state.crew_ran:
        st.session_state.crew_ran = True
        full_dialogue = _build_dialogue_for_crew()

        with st.spinner("Generating summary..."):
            try:
                from scripts.run_single_case import run_case

                raw = run_case(full_dialogue)
            except ModuleNotFoundError as e:
                missing_name = getattr(e, "name", None) or "a required package"
                st.error(
                    "Summary generation dependencies are missing (" + missing_name + "). "
                    "Install the missing package(s) and try again."
                )
                st.info("If you're missing CrewAI, install it with: `pip install crewai`")
                st.session_state.crew_ran = False
                st.session_state.stage = "ready"
                st.stop()

        st.session_state.generated_result_raw = raw

        parsed: Any = None
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = None

        # Backward compatible: older run_case returned assessment JSON directly.
        if isinstance(parsed, dict) and "assessment_plan" in parsed:
            st.session_state.generated_result_json = parsed.get("assessment_plan")
            st.session_state.care_plan_json = parsed.get("care_plan")
            st.session_state.care_plan_raw = json.dumps(parsed.get("care_plan"), ensure_ascii=False)
            # Store intermediate outputs for summary rendering
            st.session_state.subjective = parsed.get("subjective")
            st.session_state.history_subjective = parsed.get("history_subjective")
            st.session_state.objective = parsed.get("objective")
        else:
            st.session_state.generated_result_json = parsed
            st.session_state.care_plan_json = None
            st.session_state.care_plan_raw = None

        st.session_state.generated_summary_text = _render_summary_text(parsed)
        st.session_state.stage = "generated"
        st.rerun()



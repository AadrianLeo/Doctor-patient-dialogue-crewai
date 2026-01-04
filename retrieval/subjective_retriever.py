from __future__ import annotations

import re
import math
from typing import Iterable, List, Tuple

from retrieval.loader import load_json

_subjective_data: list[dict] | None = None


def _get_subjective_data() -> list[dict]:
    global _subjective_data
    if _subjective_data is None:
        _subjective_data = load_json("section_targets/subjective.json")["data"]
    return _subjective_data


_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _question_intent(text: str) -> str:
    """Deterministic intent tag for a single clinician question.

    This is used for state-aware follow-up selection. It is intentionally
    lightweight and explainable (regex/rules only).
    """

    q = " ".join(str(text).lower().split())
    if not q:
        return "other"

    if re.search(r"\b(how long|when did|started|since when|duration)\b", q):
        return "duration"

    # Fever severity often manifests as temperature measurement.
    if re.search(r"\b(temperature|temp)\b", q) and re.search(
        r"\b(measure|measured|check|checked|taken|take)\b", q
    ):
        return "temp_measurement"
    if re.search(r"\b(how high|how hot)\b", q) and re.search(r"\b(temperature|temp|fever)\b", q):
        return "temp_measurement"

    if re.search(r"\b(how (bad|severe)|severity|rate (it|your)|scale|0\s*to\s*10|\b\d{1,2}\s*/\s*10\b)\b", q):
        return "severity"

    # General severity/wellbeing phrasing (not necessarily pain-specific).
    if re.search(r"\bhow are you feeling\b", q) or re.search(r"\b(feel|feeling)\s+(better|worse)\b", q):
        return "severity"

    if re.search(r"\b(allerg|reaction)\b", q):
        return "allergies"

    if re.search(
        r"\b(med(?:s|ication|ications)?|medicine|medicines|tablet|tablets|paracetamol|acetaminophen|ibuprofen|painkillers?)\b",
        q,
    ):
        return "meds"

    # Red flags: keep separate buckets so the caller can gate by domain.
    if re.search(r"\b(chest pain|shortness of breath|breathless)\b", q):
        return "cardioresp_red_flags"
    if re.search(r"\b(confusion|faint|seizure|neck (?:pain|stiff|stiffness)|vision)\b", q):
        return "neuro_red_flags"

    # Neuro red flags for back/limb symptoms (kept separate for domain gating).
    if re.search(r"\b(numb(?:ing|ness)?|tingl(?:e|ing)|weakness)\b", q) and re.search(
        r"\b(back|spine|leg|legs|sciatica|down (?:your )?leg)\b",
        q,
    ):
        return "neuro_back_red_flags"
    if re.search(r"\b(numb(?:ing|ness)?|tingl(?:e|ing)|weakness)\b", q) and re.search(
        r"\b(hand|hands|arm|arms|grip|gripping)\b",
        q,
    ):
        return "neuro_limb_red_flags"

    # Cauda equina-style screen; treat as back red-flag bucket.
    if re.search(r"\b(loss of bowel|loss of bladder|bowel or bladder function)\b", q):
        return "neuro_back_red_flags"
    if re.search(r"\b(rash|hives|urticaria|itch|itchy|itching)\b", q):
        return "rash_red_flags"
    if re.search(
        r"\b(blood in (?:your )?(?:stool|poop)|bloody (?:stool|poop)|black (?:stool|stools)|tarry (?:stool|stools)|melena|hematochezia)\b",
        q,
    ):
        return "gi_red_flags"

    if re.search(r"\b(any other|associated|along with|else)\b", q):
        return "associated"

    # Multi-symptom checklists are "associated" screens.
    q_tokens = _tokenize(q)
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
        "pain",
        "abdominal",
        "stomach",
        "throat",
    }
    if len(q_tokens.intersection(symptom_hints)) >= 2 and "?" in q:
        return "associated"

    return "symptom_detail"

_CUT_CONTEXT_RE = re.compile(
    r"(?:\bor\b\s+)?(?:you\s+know\s+)?you\s+(?:said|mentioned)\b",
    re.IGNORECASE,
)

_ALONG_WITH_RE = re.compile(r"\balong with (?:the|your)\s+([a-z0-9]+)\b", re.IGNORECASE)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "am",
    "be",
    "been",
    "but",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "have",
    "having",
    "how",
    "i",
    "im",
    "i'm",
    "in",
    "into",
    "is",
    "it",
    "its",
    "like",
    "just",
    "me",
    "my",
    "no",
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
    "with",
    "you",
    "your",
    "yes",
    "day",
    "days",
    # Common conversational fillers / generic words that cause accidental overlap
    "hello",
    "hi",
    "okay",
    "alright",
    "all",
    "right",
    "um",
    "uh",
    "well",
    "so",
    "also",
    "feel",
    "feels",
    "felt",
    "feeling",
    "problem",
    "issue",
    "time",
    "times",
    "today",
    "yesterday",
    "tomorrow",
    "recent",
    "recently",
    "past",
    "over",
    "ago",
}


# Generic symptom/complaint tokens that are often present but not helpful for
# choosing the right dialogue. We keep them in raw tokens for decontextualizing
# questions, but we try to avoid using them as the primary similarity signal.
_GENERIC_QUERY_TOKENS = {
    "pain",
    "ache",
    "aches",
    "hurt",
    "hurts",
    "sore",
    "worse",
    "better",
    "bad",
    "severe",
    "mild",
    "moderate",
    "symptom",
    "symptoms",
    "like",
    "feel",
    "feels",
    "felt",
    "feeling",
    "problem",
    "issue",
    "time",
    "times",
}


def _normalize_token(token: str) -> set[str]:
    token = token.lower()
    out = {token}

    # Lightweight synonym expansion for common phrasing differences.
    if token in {"abdomen", "abdominal", "belly", "stomach", "tummy"}:
        out |= {"abdomen", "abdominal", "belly", "stomach", "tummy"}

    # Very lightweight plural normalization to improve overlap on common symptoms
    # (e.g., fever/fevers, cough/coughs, stool/stools).
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        out.add(token[:-1])

    # Very lightweight verb normalization for common symptom phrasing.
    # Examples: vomiting -> vomit, vomited -> vomit, coughing -> cough.
    if len(token) > 6 and token.endswith("ing"):
        out.add(token[:-3])
    if len(token) > 5 and token.endswith("ed"):
        out.add(token[:-2])

    return out


def _anchor_query_tokens(tokens: set[str]) -> set[str]:
    """Return tokens best suited for similarity scoring.

    We avoid over-weighting generic complaint tokens that appear in many
    dialogues and can pull in unrelated contexts.
    """
    anchors = {t for t in tokens if t not in _GENERIC_QUERY_TOKENS and len(t) >= 3}
    return anchors or tokens


def _tokenize(text: str) -> set[str]:
    # Remove speaker tags to avoid skewing similarity.
    text = re.sub(r"\[(doctor|patient)\]", " ", text, flags=re.IGNORECASE)
    tokens: set[str] = set()
    for m in _WORD_RE.finditer(text):
        raw = m.group(0)
        for t in _normalize_token(raw):
            if t in _STOPWORDS or t.isdigit():
                continue
            tokens.add(t)
    return tokens


def _build_idf() -> dict[str, float]:
    # Compute document frequency across all dialogues using `src` tokens.
    df: dict[str, int] = {}
    n = 0
    for item in _get_subjective_data():
        src = item.get("src", "") or ""
        doc_tokens = _tokenize(src)
        if not doc_tokens:
            continue
        n += 1
        for t in doc_tokens:
            df[t] = df.get(t, 0) + 1

    # Standard smoothed IDF.
    idf: dict[str, float] = {}
    for t, freq in df.items():
        idf[t] = math.log((n + 1) / (freq + 1)) + 1.0

    return idf


_IDF: dict[str, float] | None = None


def _get_idf() -> dict[str, float]:
    global _IDF
    if _IDF is None:
        _IDF = _build_idf()
    return _IDF


def _expand_query_tokens_for_domains(tokens: set[str]) -> set[str]:
    """Deterministically expand query tokens for retrieval.

    This does NOT generate questions; it only helps match the existing dataset's
    clinician questions (e.g., abdominal pain often co-occurs with nausea/vomiting
    questions in the source dialogues).
    """

    t = set(tokens)

    abdominal = {"abdomen", "abdominal", "belly", "stomach", "tummy"}
    headache = {"headache", "migraine", "head"}
    fever = {"fever", "temperature", "chills", "sweats"}

    if t & abdominal:
        t |= {
            "nausea",
            "vomit",
            "vomiting",
            "diarrhea",
            "constipation",
            "bowel",
            "stool",
            "urine",
            "urinary",
            "burning",
            "blood",
        }

    if t & headache:
        t |= {"vision", "neck", "stiff", "light", "nausea"}

    if t & fever:
        t |= {"cough", "sore", "throat", "nausea", "vomiting", "diarrhea"}

    # Nausea/vomiting often co-occur with GI symptoms and red-flags; expand lightly.
    if {"vomit", "vomiting", "vomited", "nausea"}.intersection(t):
        t |= {
            "nausea",
            "vomit",
            "vomiting",
            "abdomen",
            "abdominal",
            "stomach",
            "belly",
            "diarrhea",
            "dehydration",
            "fluid",
            "fluids",
            "blood",
        }

    return t


def _is_broad_history_question(text: str) -> bool:
    lower = " ".join(str(text).split()).strip().lower()
    if not lower:
        return False

    # Keep a small allowlist for broad history topics that may be useful even
    # without explicit symptom-token overlap.
    keywords = (
        "past medical",
        "medical history",
        "history of",
        "surgical history",
        "family history",
        "allerg",
        "medication",
        "medications",
        "medicine",
        "smok",
        "alcohol",
        "drug use",
        "illicit",
        "occupation",
        "work",
        "travel",
        "contacts",
        "exposure",
        "vaccin",
    )
    return any(k in lower for k in keywords)


def _score_question(query_tokens: set[str], q_tokens: set[str]) -> float:
    overlap = query_tokens & q_tokens
    if not overlap:
        return 0.0
    idf = _get_idf()
    return sum(idf.get(t, 1.0) for t in overlap)


def _looks_like_real_question(text: str) -> bool:
    # Mirror the app's safety: avoid long directive/plan paragraphs being treated
    # as questions just because they start with 'where/how/etc'.
    cleaned = " ".join(str(text).split()).strip()
    if not cleaned:
        return False

    lower = cleaned.lower()
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

    # If punctuation exists, still reject overly long plan/statement blocks.
    if "?" in cleaned:
        if len(cleaned.split()) > 28:
            return False
        return True

    # Without punctuation, require strong question-leading cues and keep it short.
    if len(cleaned.split()) >= 18:
        return False
    return lower.startswith(starters)


def _build_question_pool() -> list[tuple[str, set[str]]]:
    pool: list[tuple[str, set[str]]] = []
    seen: set[str] = set()
    for item in _get_subjective_data():
        src = item.get("src", "") or ""
        _patient_text, _doctor_text, doctor_role = _split_by_role(src)
        for q in _extract_doctor_questions(src, doctor_role=doctor_role):
            cleaned = " ".join(str(q).split()).strip()
            if not cleaned:
                continue
            low = cleaned.lower()
            if low in seen:
                continue
            if not _looks_like_real_question(cleaned):
                continue
            seen.add(low)
            pool.append((cleaned, _tokenize(cleaned)))
    return pool


_QUESTION_POOL: list[tuple[str, set[str]]] | None = None


# An intent-tagged question bank built deterministically from dataset `src`.
_QUESTION_BANK: list[tuple[str, set[str], str]] | None = None


def _get_question_bank() -> list[tuple[str, set[str], str]]:
    global _QUESTION_BANK
    if _QUESTION_BANK is not None:
        return _QUESTION_BANK

    bank: list[tuple[str, set[str], str]] = []
    seen: set[str] = set()
    for item in _get_subjective_data():
        src = item.get("src", "") or ""
        _patient_text, _doctor_text, doctor_role = _split_by_role(src)
        for q in _extract_doctor_questions(src, doctor_role=doctor_role):
            cleaned = " ".join(str(q).split()).strip()
            if not cleaned:
                continue
            low = cleaned.lower()
            if low in seen:
                continue
            if not _looks_like_real_question(cleaned):
                continue
            seen.add(low)
            bank.append((cleaned, _tokenize(cleaned), _question_intent(cleaned)))

    _QUESTION_BANK = bank
    return _QUESTION_BANK


def get_questions_by_intent(
    query_text: str,
    *,
    intent: str,
    k: int = 5,
) -> List[str]:
    """Retrieve individual clinician questions by intent.

    This does NOT select a dialogue. It uses a global question bank extracted
    from dataset `src`, scored by deterministic token overlap.
    """

    raw_tokens = _tokenize(query_text)
    if not raw_tokens:
        return []

    anchors = _anchor_query_tokens(raw_tokens)
    expanded = _expand_query_tokens_for_domains(raw_tokens)

    desired = str(intent or "").strip().lower()
    bank = _get_question_bank()

    red_flag_intents = {
        "cardioresp_red_flags",
        "neuro_red_flags",
        "rash_red_flags",
        "gi_red_flags",
        "neuro_back_red_flags",
        "neuro_limb_red_flags",
    }

    # Required-slot intents (duration/severity) often do not lexically overlap the complaint.
    # We still need to retrieve them deterministically to avoid premature stop.
    required_slot_intents = {"duration", "severity", "temp_measurement"}

    scored: list[tuple[float, str]] = []
    for q, q_tokens, q_intent in bank:
        if desired and q_intent != desired:
            continue

        # Keep questions anchored to the complaint; allow broad history for meds/allergies.
        # For red-flag intents, do NOT require lexical overlap (red flags often use
        # different vocabulary than the chief complaint).
        is_broad = _is_broad_history_question(q)
        if desired not in {"meds", "allergies"} and desired not in red_flag_intents and desired not in required_slot_intents:
            if anchors and not (anchors & q_tokens) and not is_broad:
                continue

        s = _score_question(expanded, q_tokens)
        if desired not in red_flag_intents and desired not in required_slot_intents:
            if s <= 0 and not is_broad:
                continue
        scored.append((s, q))

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [q for _s, q in scored[: max(0, int(k))]]


def _split_by_speaker(src_dialogue: str) -> tuple[str, str]:
    # Backwards-compatible wrapper for older call sites.
    patient_text, doctor_text, _doctor_role = _split_by_role(src_dialogue)
    return (patient_text, doctor_text)


def _split_by_role(src_dialogue: str) -> tuple[str, str, str]:
    """Return (patient_text, doctor_text, doctor_role).

    Some dataset entries have swapped speaker tags. We infer which tag is acting
    as the clinician by comparing how many questions each speaker asks.
    """

    doctor_parts: List[str] = []
    patient_parts: List[str] = []
    doctor_q = 0
    patient_q = 0

    for raw_line in src_dialogue.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lower = line.lower()
        if lower.startswith("[doctor]"):
            text = line[len("[doctor]") :].strip()
            if text:
                doctor_parts.append(text)
                if "?" in text:
                    doctor_q += 1
        elif lower.startswith("[patient]"):
            text = line[len("[patient]") :].strip()
            if text:
                patient_parts.append(text)
                if "?" in text:
                    patient_q += 1

    # Heuristic: clinician usually asks more questions.
    swapped = (patient_q > doctor_q) and (patient_q >= 2)

    if swapped:
        doctor_text = " ".join(patient_parts)
        patient_text = " ".join(doctor_parts)
        doctor_role = "patient"
    else:
        patient_text = " ".join(patient_parts)
        doctor_text = " ".join(doctor_parts)
        doctor_role = "doctor"

    return (patient_text, doctor_text, doctor_role)


def _score_dialogue(query_tokens: set[str], patient_tokens: set[str], doctor_tokens: set[str]) -> int:
    """Weighted overlap.

    We bias toward matching symptoms in the patient's own turns. This helps avoid
    picking dialogues where the doctor only asks generic ROS questions like
    "fever or chills" that the patient denies.
    """
    patient_overlap = query_tokens & patient_tokens
    doctor_overlap = query_tokens & doctor_tokens

    idf = _get_idf()
    patient_score = sum(idf.get(t, 1.0) for t in patient_overlap)
    doctor_score = sum(idf.get(t, 1.0) for t in doctor_overlap)
    return (3.0 * patient_score) + doctor_score


def _extract_doctor_questions(src_dialogue: str, doctor_role: str = "doctor") -> List[str]:
    def is_question(text: str) -> bool:
        if "?" in text:
            return True

        # Many dataset transcripts omit punctuation; use lightweight lexical cues.
        cleaned = " ".join(text.split()).strip()
        lower = cleaned.lower()

        # Strip common leading fillers.
        fillers = ("okay ", "alright ", "all right ", "um ", "so ", "well ", "and ", "then ", "now ")
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

    questions: List[str] = []
    for raw_line in src_dialogue.splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if doctor_role == "doctor":
            if not lower.startswith("[doctor]"):
                continue
            text = line[len("[doctor]") :].strip()
        else:
            if not lower.startswith("[patient]"):
                continue
            text = line[len("[patient]") :].strip()

        if not text:
            continue

        # Heuristics to drop non-question / non-clinical administrative utterances.
        lower = text.lower()
        if "dax" in lower:
            continue
        if "dragon" in lower:
            continue
        if "finalize" in lower:
            continue
        if lower.startswith("hi") or "good to see you" in lower:
            # Greeting questions add noise for follow-ups.
            if "?" in text:
                continue

        if not is_question(text):
            continue

        if not _looks_like_real_question(text):
            continue

        # Normalize whitespace.
        cleaned = " ".join(text.split())
        questions.append(cleaned)

    return questions


def _decontextualize_question(question: str, query_tokens: set[str]) -> str:
    """Remove dialogue-dependent phrasing from retrieved questions.

    Some dataset questions refer to earlier context (e.g., "you said it's in the back
    of your head..."). When we reuse these as follow-ups in a new consultation, that
    prior context may not exist, which confuses the user.

    This is purely deterministic string cleanup (no LLM generation).
    """
    q = " ".join(str(question).split()).strip()
    if not q:
        return ""

    # Remove common leading filler phrases to make questions stand alone.
    # (Deterministic cleanup; no new content is introduced.)
    while True:
        before = q
        q = re.sub(r"^(?:okay|alright|all right|um|so|well)\b[ ,]*", "", q, flags=re.IGNORECASE).strip()
        q = re.sub(r"^and\b[ ,]*", "", q, flags=re.IGNORECASE).strip()
        if q == before:
            break

    # If the clinician question references prior user statements, drop that tail.
    m = _CUT_CONTEXT_RE.search(q)
    if m:
        q = q[: m.start()].rstrip(" ,;:-")
        q = re.sub(r"\b(or|and)\b\s*$", "", q, flags=re.IGNORECASE).rstrip(" ,;:-")

    # Avoid presuppositional phrasing like "along with the headache" when the
    # patient never mentioned that symptom.
    def _along_with_repl(match: re.Match[str]) -> str:
        tok = match.group(1).lower()
        tok_norm = tok[:-1] if (len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss")) else tok
        if tok_norm not in query_tokens:
            return "along with that"
        return match.group(0)

    q = _ALONG_WITH_RE.sub(_along_with_repl, q)

    q = " ".join(q.split()).strip()
    return q


def get_subjective_questions(
    symptom_text: str,
    k: int = 3,
    pool_size: int = 12,
    *,
    allow_broad_history: bool = True,
    allowed_broad_intents: set[str] | None = None,
) -> List[str]:
    """Retrieve follow-up questions from similar dialogues.

    - Retrieval is similarity-based (token overlap) using dataset `src`.
    - Returned questions are extracted from doctor turns in `src`.
    - Questions are never generated; they are verbatim dataset questions with
      deterministic decontextualization.
    """

    raw_tokens = _tokenize(symptom_text)
    if not raw_tokens:
        return []

    allowed_intents = {
        str(x).strip().lower()
        for x in (allowed_broad_intents or set())
        if str(x).strip()
    }

    global _QUESTION_POOL

    # Use anchors for retrieval scoring, but keep raw tokens for question
    # decontextualization checks (e.g., "along with the headache").
    query_tokens = _anchor_query_tokens(raw_tokens)

    # For very short complaints (e.g., a single symptom like "fever"), requiring
    # overlap of 2 tokens is impossible. Adapt the threshold to the query size.
    min_overlap = 2 if len(query_tokens) >= 3 else 1

    # If the query contains a rare token (high IDF), prefer dialogues that also
    # contain that token (when available).
    idf = _get_idf()
    rare_sorted = sorted(query_tokens, key=lambda t: idf.get(t, 1.0), reverse=True)
    must_token = None
    if rare_sorted and idf.get(rare_sorted[0], 1.0) >= 2.0:
        must_token = rare_sorted[0]

    scored: List[Tuple[float, int, int, bool, int]] = []
    data = _get_subjective_data()
    for idx, item in enumerate(data):
        src = item.get("src", "") or ""
        patient_text, doctor_text, _doctor_role = _split_by_role(src)
        patient_tokens = _tokenize(patient_text)
        doctor_tokens = _tokenize(doctor_text)
        s = _score_dialogue(query_tokens, patient_tokens, doctor_tokens)
        has_must = False
        if must_token is not None:
            has_must = (must_token in patient_tokens) or (must_token in doctor_tokens)
        scored.append((s, len(query_tokens & patient_tokens), len(query_tokens & doctor_tokens), has_must, idx))

    # Sort by score desc, then patient overlap, then doctor overlap, then original order.
    scored.sort(key=lambda t: (-t[0], -t[1], -t[2], t[4]))

    use_must = must_token is not None and any(has_must for _s, _p, _d, has_must, _idx in scored)

    # Prefer dialogues where the patient overlap is positive; otherwise fall back
    # to overall best-scoring dialogues.
    top = [
        idx
        for s, p_ov, d_ov, has_must, idx in scored
        if s > 0
        and p_ov > 0
        and (p_ov + d_ov) >= min_overlap
        and (not use_must or has_must)
    ][:pool_size]
    if not top:
        top = [
            idx
            for s, p_ov, d_ov, has_must, idx in scored
            if s > 0 and (p_ov + d_ov) >= min_overlap and (not use_must or has_must)
        ][:pool_size]
    if not top:
        top = [idx for _s, _p_ov, _d_ov, _has_must, idx in scored[:pool_size]]

    expanded = _expand_query_tokens_for_domains(raw_tokens)
    anchors = _anchor_query_tokens(raw_tokens)

    q_scored: list[tuple[float, str, str]] = []
    seen: set[str] = set()
    for idx in top:
        src = data[idx].get("src", "") or ""
        _patient_text, _doctor_text, doctor_role = _split_by_role(src)
        for q in _extract_doctor_questions(src, doctor_role=doctor_role):
            q2 = _decontextualize_question(q, raw_tokens)
            if not q2 or len(q2) < 6:
                continue
            low = q2.lower()
            if low in seen:
                continue

            q_tokens = _tokenize(q2)
            q_intent = _question_intent(q2)
            is_broad = _is_broad_history_question(q2)
            if is_broad:
                if not allow_broad_history:
                    continue
                if allowed_intents and q_intent not in allowed_intents:
                    continue

            # Require at least one anchor overlap to keep questions on-topic,
            # unless this is an explicitly-allowed broad intent.
            if anchors and not (anchors & q_tokens) and not (allowed_intents and q_intent in allowed_intents):
                continue

            s = _score_question(expanded, q_tokens)
            # For required follow-up intents (duration/severity/temp/etc), lexical
            # overlap with the complaint is often zero. We still want these
            # questions deterministically if the caller explicitly allows them.
            if s <= 0 and not is_broad and not (allowed_intents and q_intent in allowed_intents):
                continue

            # Prefer on-topic symptom questions first; keep broad history later.
            q_scored.append(((s if s > 0 else 0.25), q2, q_intent))
            seen.add(low)

    q_scored.sort(key=lambda t: (-t[0], t[1]))

    out: List[str] = []

    # If the caller explicitly allows certain intents, try to include at least
    # one question per intent (when available) before filling the rest.
    if allowed_intents:
        intent_order = ["duration", "temp_measurement", "severity", "meds", "allergies"]
        out_low: set[str] = set()
        for intent in intent_order:
            if intent not in allowed_intents:
                continue
            for _s, q2, q_intent in q_scored:
                if q_intent != intent:
                    continue
                low = q2.lower()
                if low in out_low:
                    continue
                out.append(q2)
                out_low.add(low)
                break

        # Backfill missing allowed intents from the global question pool when
        # similar dialogues don't contain them (common for short complaints).
        have_intents = { _question_intent(q) for q in out }
        need_intents = [i for i in intent_order if i in allowed_intents and i not in have_intents]
        if need_intents:
            if _QUESTION_POOL is None:
                _QUESTION_POOL = _build_question_pool()

            for intent in need_intents:
                for q, _q_tokens in _QUESTION_POOL:
                    q2 = _decontextualize_question(q, raw_tokens)
                    if not q2:
                        continue
                    low = q2.lower()
                    if low in out_low:
                        continue
                    q_intent = _question_intent(q2)
                    if q_intent != intent:
                        continue
                    is_broad = _is_broad_history_question(q2)
                    if is_broad:
                        if not allow_broad_history:
                            continue
                        if allowed_intents and q_intent not in allowed_intents:
                            continue
                    out.append(q2)
                    out_low.add(low)
                    break

        if len(out) >= k:
            return out[:k]

        for _s, q2, _intent in q_scored:
            low = q2.lower()
            if low in out_low:
                continue
            out.append(q2)
            out_low.add(low)
            if len(out) >= k:
                return out
    else:
        for _s, q2, _intent in q_scored:
            out.append(q2)
            if len(out) >= k:
                return out

    # If we still don't have enough, supplement from a global question pool.
    if len(out) < k:
        if _QUESTION_POOL is None:
            _QUESTION_POOL = _build_question_pool()

        scored_qs: list[tuple[float, str, str]] = []
        for q, q_tokens in _QUESTION_POOL:
            if q.lower() in seen:
                continue

            q_intent = _question_intent(q)
            is_broad = _is_broad_history_question(q)
            if is_broad:
                if not allow_broad_history:
                    continue
                if allowed_intents and q_intent not in allowed_intents:
                    continue

            if anchors and not (anchors & q_tokens) and not (allowed_intents and q_intent in allowed_intents):
                continue

            s = _score_question(expanded, q_tokens)
            if s <= 0 and not (is_broad or (allowed_intents and q_intent in allowed_intents)):
                continue
            scored_qs.append((s if s > 0 else 0.25, q, q_intent))

        scored_qs.sort(key=lambda t: (-t[0], t[1]))

        max_supplement = min(max(0, k - len(out)), 12)
        for _s, q, _intent in scored_qs[:max_supplement]:
            q2 = _decontextualize_question(q, raw_tokens)
            if not q2:
                continue
            low = q2.lower()
            if low in seen:
                continue
            seen.add(low)
            out.append(q2)
            if len(out) >= k:
                break

    return out

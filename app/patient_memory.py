from __future__ import annotations

from dataclasses import dataclass, field, asdict
import re
from typing import Any


_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


_MED_KEYWORDS = (
    "medication",
    "medications",
    "medicine",
    "medicines",
    "tablet",
    "tablets",
    "antibiotic",
    "antibiotics",
)

_MED_NAMES = (
    "ibuprofen",
    "paracetamol",
    "acetaminophen",
)

_MED_USE_VERBS = (
    "take",
    "took",
    "taken",
    "taking",
    "use",
    "used",
    "using",
    "on ",
    "started",
    "start ",
)


def _norm_ws(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _lower(text: str) -> str:
    return _norm_ws(text).lower()


def _tokenize(text: str) -> set[str]:
    toks: set[str] = set()
    for m in _WORD_RE.finditer(text or ""):
        t = m.group(0).lower()
        if t.isdigit():
            continue
        toks.add(t)
    return toks


def _extract_duration(text: str) -> str:
    t = _lower(text)
    # Extract common duration phrases as-is for downstream display.
    patterns = [
        r"\bfor\s+(?:about\s+)?\d+\s+(?:minute|minutes|hour|hours|day|days|week|weeks|month|months|year|years)\b",
        r"\bsince\s+(?:yesterday|today|last\s+night|last\s+week|last\s+month|\d+\s+(?:day|days|week|weeks|month|months|year|years)\s+ago)\b",
        r"\bstarted\s+(?:\d+\s+(?:day|days|week|weeks|month|months|year|years)\s+ago|yesterday|today|last\s+night)\b",
    ]
    for pat in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            return _norm_ws(m.group(0))
    return ""


def _extract_severity(text: str) -> str:
    t = _lower(text)

    # Prefer explicit numeric scales.
    m = re.search(r"\b(\d{1,2})\s*/\s*10\b", t)
    if m:
        return _norm_ws(m.group(0))
    m = re.search(r"\b(\d{1,2})\s*(?:out\s+of)\s*10\b", t)
    if m:
        return _norm_ws(m.group(0))

    # Then common qualitative descriptors.
    m = re.search(r"\b(mild|moderate|severe)\b", t)
    if m:
        return _norm_ws(m.group(1))

    # Very conservative intensity phrases.
    for phrase in ("very bad", "really bad", "quite bad", "pretty bad"):
        if phrase in t:
            return phrase

    return ""


def _extract_temperature(text: str) -> str:
    t = _lower(text)

    # Only capture when there is an explicit temperature mention.
    if not any(k in t for k in ("temp", "temperature", "fever", "degree", "degrees", "°")):
        # Also allow explicit C/F suffix.
        if not re.search(r"\b\d{2,3}(?:\.\d)?\s*(?:c|f)\b", t):
            return ""

    # Common forms: 38, 38.5, 101F, 38 C, 38.5°C
    m = re.search(r"\b(\d{2,3}(?:\.\d)?)\s*(?:°\s*)?(c|f)?\b", t)
    if not m:
        return ""

    value = m.group(1)
    unit = (m.group(2) or "").upper()
    if unit:
        return f"{value}{unit}"
    return value


def _extract_symptoms(text: str) -> set[str]:
    t = _lower(text)
    symptoms: set[str] = set()

    # Very conservative keyword extraction: only things explicitly mentioned.
    mapping = {
        "fever": ["fever", "temperature", "high temp"],
        "chills": ["chills", "shivering"],
        "headache": ["headache", "migraine"],
        "vomiting": ["vomit", "vomiting", "vomited", "throw up", "threw up"],
        "nausea": ["nausea", "nauseous", "sick"],
        "diarrhea": ["diarrhea", "diarrhoea", "loose stool", "loose stools"],
        "cough": ["cough", "coughing"],
        "sore throat": ["sore throat", "throat"],
        "shortness of breath": ["shortness of breath", "breathless", "breathing"],
        "chest pain": ["chest pain", "pain in your chest"],
        "abdominal pain": ["abdominal pain", "stomach pain", "belly pain", "tummy pain"],
        "urinary symptoms": ["burning", "burns", "urine", "urinating", "pee"],
        "rash": ["rash", "hives"],
    }

    for label, needles in mapping.items():
        for n in needles:
            if n in t:
                symptoms.add(label)
                break

    return symptoms


@dataclass
class PatientMemory:
    """Structured, retrieval-safe memory.

    This is NOT a diagnosis engine; it only stores what the patient explicitly
    stated, plus minimal normalized labels for retrieval.
    """

    chief_complaint: str = ""
    symptoms: set[str] = field(default_factory=set)
    duration: str = ""
    severity: str = ""
    temperature: str = ""  # e.g., "38.5C" or "101F" or "38.5"

    # Free-text buckets (patient-provided); kept short and extractive.
    pmh: list[str] = field(default_factory=list)
    meds: list[str] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)
    social: list[str] = field(default_factory=list)
    exposures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["symptoms"] = sorted(self.symptoms)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PatientMemory":
        data = dict(data or {})
        data["symptoms"] = set(data.get("symptoms") or [])
        return cls(**data)

    def update_from_patient_text(self, text: str, *, is_first_message: bool = False) -> None:
        """Deterministic extraction + merge."""

        cleaned = _norm_ws(text)
        if not cleaned:
            return

        if is_first_message and not self.chief_complaint:
            self.chief_complaint = cleaned

        # Merge symptom labels.
        self.symptoms |= _extract_symptoms(cleaned)

        # Update duration if we can extract one.
        if not self.duration:
            d = _extract_duration(cleaned)
            if d:
                self.duration = d

        # Update severity if we can extract one.
        if not self.severity:
            s = _extract_severity(cleaned)
            if s:
                self.severity = s

        # Capture measured temperature (extractive) if mentioned.
        if not self.temperature:
            tmp = _extract_temperature(cleaned)
            if tmp:
                self.temperature = tmp

        # Very light capture of broad-history statements if the patient volunteers them.
        low = _lower(cleaned)
        has_allergy = "allerg" in low
        if has_allergy:
            self._append_unique(self.allergies, cleaned)

        has_med_keyword = any(k in low for k in _MED_KEYWORDS)
        has_med_name = any(n in low for n in _MED_NAMES)
        has_med_use_verb = any(v in low for v in _MED_USE_VERBS)
        says_no_meds = bool(
            re.search(
                r"\b(no|not)\s+(?:any\s+)?(?:medication|medications|medicine|medicines|tablets?)\b",
                low,
            )
        )

        # Capture meds only when the patient indicates use (or explicitly says none),
        # or when a known med name is mentioned. Avoid treating allergy-only statements
        # (e.g., "no allergies to medications") as medication-use.
        if has_med_name or says_no_meds or (has_med_keyword and has_med_use_verb):
            self._append_unique(self.meds, cleaned)
        if any(k in low for k in ("smoke", "smoking", "alcohol", "drink", "drugs")):
            self._append_unique(self.social, cleaned)
        if any(k in low for k in ("travel", "contact", "exposure", "exposed", "sick contact", "vaccin")):
            self._append_unique(self.exposures, cleaned)
        if any(k in low for k in ("history", "diagnosed", "diabetes", "asthma", "hypertension")):
            self._append_unique(self.pmh, cleaned)

    def _append_unique(self, bucket: list[str], value: str) -> None:
        v = _norm_ws(value)
        if not v:
            return
        low = v.lower()
        if all(str(x).lower() != low for x in bucket):
            bucket.append(v)

    def inferred_topics(self) -> set[str]:
        topics: set[str] = set()
        # Required slot: symptoms (free-text + extracted labels).
        # Treat as covered as soon as the patient provides a complaint, even if
        # we cannot confidently keyword-extract specific symptom labels.
        if self.chief_complaint or self.symptoms:
            topics.add("symptoms")
        if self.duration:
            topics.add("duration")
        if self.severity:
            topics.add("severity")
        if self.temperature:
            topics.add("temp_measurement")
        # Only treat "associated" as covered when the patient has volunteered
        # at least one additional symptom beyond the chief complaint.
        if self.symptoms and len(self.symptoms) >= 2:
            topics.add("associated")
        if self.pmh:
            topics.add("pmh")
        if self.meds:
            topics.add("meds")
        if self.allergies:
            topics.add("allergies")
        if self.social:
            topics.add("social")
        if self.exposures:
            topics.add("exposures")
        return topics

    def as_retrieval_query(self, *, include_history_snippets: bool = True) -> str:
        """Build a compact query string for dataset retrieval.

        Notes:
        - For follow-up *question* retrieval, broad-history snippets (meds/allergies/etc)
          can cause drift into unrelated generic screens. Call with
          include_history_snippets=False to keep retrieval anchored to the presenting
          complaint.
        """

        parts: list[str] = []
        if self.chief_complaint:
            parts.append(self.chief_complaint)
        if self.symptoms:
            parts.append(" ".join(sorted(self.symptoms)))
        if self.duration:
            parts.append(self.duration)
        if self.severity:
            parts.append(self.severity)
        if include_history_snippets:
            # Add a few patient-provided snippets for context (not too many).
            for bucket in (self.pmh, self.meds, self.allergies, self.social, self.exposures):
                parts.extend(bucket[:1])
        return _norm_ws(" ".join(parts))

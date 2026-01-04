from __future__ import annotations

import re
from typing import Iterable

from retrieval.subjective_retriever import get_questions_by_intent, get_subjective_questions

from app.followup_agent import FollowupAgent
from app.patient_memory import PatientMemory


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

    # Urinary phrasing variants
    "pee": {"urine", "urinary"},
    "peeing": {"urine", "urinary"},
    "urinate": {"urine", "urinary"},
    "urinating": {"urine", "urinary"},
    "burn": {"burning"},
    "burns": {"burning"},
    "burning": {"burn"},

    # Stool synonyms
    "poop": {"stool", "bowel"},
    "poo": {"stool", "bowel"},
    "stools": {"stool", "bowel"},
}


# Required subjective slots only. Optional/late topics must not gate stopping.
_REQUIRED_TOPICS: tuple[str, ...] = (
    "symptoms",
    "duration",
    "severity",
    "meds",
    "allergies",
)


def tokenize_for_followups(text: str) -> set[str]:
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


def anchor_tokens_from_context(context_text: str) -> set[str]:
    toks = tokenize_for_followups(context_text)
    anchors = {t for t in toks if t not in _GENERIC_COMPLAINT_TOKENS}
    anchors = {t for t in anchors if len(t) >= 3}
    return anchors or toks


def domain_tokens(anchors: set[str]) -> set[str]:
    a = set(anchors)
    domain: set[str] = set()

    # Abdominal / GI / GU
    if {
        "abdomen",
        "abdominal",
        "stomach",
        "belly",
        "tummy",
        "appendix",
        # GI/bowel complaints that may not mention the abdomen explicitly
        "constipation",
        "diarrhea",
        "bowel",
        "stool",
        "poop",
        "poo",
    }.intersection(a):
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

    # Urinary / GU (explicit)
    if {"urine", "urinary", "pee", "peeing", "burning", "burn", "burns", "frequency"}.intersection(a):
        domain |= {
            "urine",
            "urinary",
            "pee",
            "burning",
            "frequency",
            "bladder",
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

    # Rash / dermatologic
    if {"rash", "hives", "urticaria", "itch", "itchy", "itching", "eczema"}.intersection(a):
        domain |= {
            "rash",
            "hives",
            "urticaria",
            "itch",
            "itchy",
            "itching",
            "skin",
            "sensitive",
            "eczema",
            "cream",
        }

    return domain


def looks_like_question_for_followups(text: str) -> bool:
    if not text:
        return False
    if "?" in text:
        return True

    cleaned = " ".join(str(text).split()).strip()
    lower = cleaned.lower()

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

    if len(lower.split()) >= 18 and not lower.endswith("?"):
        return False

    return False


def clean_followup_question(question: str, complaint_tokens: set[str]) -> str:
    q = " ".join(str(question).split()).strip()
    if not q:
        return ""

    q = re.sub(r"^[\s\.,;:!\-\u2013\u2014\"'`]+", "", q).strip()

    while True:
        before = q
        q = re.sub(
            r"^[\s\.,;:!\-\u2013\u2014]*\b(?:okay|alright|all right|um|uh|so|well|right)\b[\s\.,;:!\-\u2013\u2014]*",
            "",
            q,
            flags=re.IGNORECASE,
        ).strip()
        q = re.sub(
            r"^[\s\.,;:!\-\u2013\u2014]*\band\b[\s\.,;:!\-\u2013\u2014]*",
            "",
            q,
            flags=re.IGNORECASE,
        ).strip()
        if q == before:
            break

    q = re.sub(r"\b,\s*(?:uh|um)\s*,\b", ", ", q, flags=re.IGNORECASE)
    q = re.sub(r"\b(?:uh|um)\b\s*,", "", q, flags=re.IGNORECASE)
    q = re.sub(r",\s*,+", ", ", q).strip()

    m = re.search(r"(?:\bor\b\s+)?(?:you\s+know\s+)?you\s+(?:said|mentioned)\b", q, flags=re.IGNORECASE)
    if m:
        q = q[: m.start()].rstrip(" ,;:-")
        q = re.sub(r"\b(or|and)\b\s*$", "", q, flags=re.IGNORECASE).rstrip(" ,;:-")

    def _along_with_repl(match: re.Match[str]) -> str:
        tok = (match.group(1) or "").lower().strip()
        tok_norm = tok[:-1] if (len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss")) else tok
        if tok_norm and tok_norm not in complaint_tokens:
            return "along with that"
        return match.group(0)

    q = re.sub(r"\balong with (?:the|your)\s+([a-z0-9]+)\b", _along_with_repl, q, flags=re.IGNORECASE)

    return " ".join(q.split()).strip()


def is_redundant_followup(question: str, complaint_tokens: set[str]) -> bool:
    q_tokens = tokenize_for_followups(question)
    q_lower = " ".join(str(question).lower().split())

    def _is_presence_check(symptom: str) -> bool:
        if symptom == "fever":
            return bool(
                re.search(r"\b(do you have|have you got|have you had|any)\s+(a\s+)?fever\b", q_lower)
                or re.search(r"\bfever\?\s*$", q_lower)
            )
        if symptom == "headache":
            return bool(
                re.search(r"\b(do you have|have you got|have you had|any)\s+headaches?\b", q_lower)
                or re.search(r"\bheadaches?\?\s*$", q_lower)
            )
        if symptom == "constipation":
            return bool(
                re.search(
                    r"\b(do you have|have you got|have you had|any|are you having)\s+constipation\b",
                    q_lower,
                )
                or re.search(r"\bconstipation\?\s*$", q_lower)
            )
        if symptom == "diarrhea":
            return bool(
                re.search(
                    r"\b(do you have|have you got|have you had|any|are you having)\s+diarrh(?:ea|oea)\b",
                    q_lower,
                )
                or re.search(r"\bdiarrh(?:ea|oea)\?\s*$", q_lower)
            )
        return False

    if "fever" in complaint_tokens and "fever" in q_tokens and _is_presence_check("fever"):
        return True

    if "constipation" in complaint_tokens and "constipation" in q_tokens and _is_presence_check("constipation"):
        return True

    if "diarrhea" in complaint_tokens and "diarrhea" in q_tokens and _is_presence_check("diarrhea"):
        return True

    if "headache" not in complaint_tokens and "headache" in q_tokens:
        if re.search(r"\b(before|previous|like this|usual|normally)\b", q_lower):
            return True

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
    }
    q_symptoms = q_tokens.intersection(symptom_hints)
    if q_symptoms:
        # Treat classic checklists as redundant if they only ask about symptoms the
        # patient context already contains.
        if re.search(r"\b(any|do you have|have you got|have you had|are you having)\b", q_lower) or "," in q_lower:
            known = complaint_tokens.intersection(symptom_hints)
            if q_symptoms.issubset(known):
                return True

        # Also collapse repeated fever/chills (or similar) presence-checks once the
        # core symptom is already known, unless the question introduces a new symptom.
        # This helps avoid loops like: "any fever/chills?" -> "fever/chills?" -> ...
        presence_check = bool(
            re.search(
                r"\b(any|do you have|have you got|have you had|are you having|are you experiencing)\b",
                q_lower,
            )
            or q_lower.rstrip().endswith("?")
        )
        if presence_check:
            known = complaint_tokens.intersection(symptom_hints)
            new_symptoms = q_symptoms.difference(known)
            if not new_symptoms:
                return True
            # If the only "new" symptom is chills but fever is already known, allow it
            # once; otherwise it tends to become repetitive. With only patient text
            # available (no explicit negations), be conservative:
            if new_symptoms == {"chills"} and "fever" in known:
                return False

    return False


def is_contradictory_followup(question: str, complaint_tokens: set[str]) -> bool:
    q_lower = " ".join(str(question).lower().split())

    if "headache" in complaint_tokens and re.search(r"\bno\s+headaches?\b", q_lower):
        return True

    if "fever" in complaint_tokens and re.search(r"\bno\s+fever\b", q_lower):
        return True

    return False


def followup_topic(question: str) -> str:
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
    # Common severity/intensity phrasing used for non-pain complaints.
    if re.search(r"\b(how high|how much|how intense)\b", q) and re.search(r"\b(fever|temperature|temp|pain)\b", q):
        return "severity"
    if re.search(r"\ballerg(?:y|ies|ic)?\b|\breaction(?:s)?\b", q):
        return "allergies"
    if re.search(
        r"\b(med(?:s|ication|ications)?|medicine|medicines|tablet|tablets|paracetamol|acetaminophen|ibuprofen|antibiotic|antibiotics|painkillers?)\b",
        q,
    ):
        return "meds"
    if re.search(r"\b(past|history|medical|strep|asthma|diabetes|heart|kidney)\b", q):
        return "pmh"
    if re.search(r"\b(work|job|student|smok|alcohol|drugs)\b", q):
        return "social"
    if re.search(r"\b(contact|sick|colleague|travel|exposure)\b", q):
        return "exposures"

    if re.search(
        r"\b(chest pain|shortness of breath|short of breath|breathless|difficulty breathing|trouble breathing)\b",
        q,
    ):
        return "cardioresp_red_flags"
    if re.search(r"\b(confusion|faint|seizure|neck (?:pain|stiff|stiffness)|vision)\b", q):
        return "neuro_red_flags"
    # GI red flags (for bowel/abdominal complaints)
    if re.search(
        r"\b(blood in (?:your )?(?:stool|poop)|bloody (?:stool|poop)|black (?:stool|stools)|tarry (?:stool|stools)|melena|hematochezia)\b",
        q,
    ):
        return "gi_red_flags"
    if re.search(r"\b(severe|worst)\b", q) and re.search(r"\b(abdominal|abdomen|belly|stomach)\b", q):
        return "gi_red_flags"
    if re.search(r"\b(vomiting|throwing up)\b", q) and re.search(r"\b(can't|cannot|unable)\b", q):
        return "gi_red_flags"
    if re.search(r"\b(numb(?:ing)?|tingl(?:ing|e)|weakness)\b", q) and re.search(r"\b(hand|hands|arm|arms|grip|gripping)\b", q):
        return "neuro_limb_red_flags"
    if re.search(r"\b(numb(?:ing)?|tingl(?:ing|e)|weakness)\b", q) and re.search(r"\b(leg|legs|back|spine|down your leg)\b", q):
        return "neuro_back_red_flags"
    if re.search(r"\b(rash|hives|urticaria|itch|itchy|itching)\b", q):
        return "rash_red_flags"

    # Keep this late so red-flag detection can take precedence even when
    # the sentence contains phrases like "any other symptoms".
    if re.search(r"\b(any other|associated|along with|else)\b", q):
        return "associated"

    q_tokens = tokenize_for_followups(q)
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


def is_incompatible_followup(question: str, context_text: str) -> bool:
    q = " ".join(str(question).strip().split())
    if not q:
        return True

    q_lower = q.lower()
    ctx_tokens = tokenize_for_followups(context_text)
    topic = followup_topic(q)

    # Dataset sometimes includes scripted handoff lines like:
    #   "i know that you were endorsing the back pain. any other symptoms? ..."
    # Treat these as incompatible unless the endorsed symptom is in the patient's context.
    if "you were endorsing" in q_lower:
        for loc in re.findall(r"you were endorsing (?:the )?([a-z]+) pain", q_lower):
            if loc and loc not in ctx_tokens:
                return True

    # If a duration/severity question explicitly references a symptom/condition that
    # the patient hasn't mentioned, it's likely off-topic drift from retrieval.
    if topic in {"duration", "severity"}:
        symptom_terms = {
            "fever",
            "chills",
            "nausea",
            "vomiting",
            "diarrhea",
            "cough",
            "headache",
            "migraine",
            "rash",
            "hives",
            "urticaria",
            "itch",
            "itchy",
            "itching",
            "breath",
            "breathless",
            "shortness",
            "sob",
            "chest",
            "dizzy",
            "dizziness",
            "faint",
            "seizure",
            "back",
            "leg",
            "legs",
            "urine",
            "urinary",
            "burning",
        }
        q_tokens = tokenize_for_followups(q_lower)
        mentioned = q_tokens.intersection(symptom_terms)
        if mentioned and not mentioned.intersection(ctx_tokens):
            return True

    procedure_words = set(re.findall(r"\b[a-z]+(?:ectomy|ostomy|oscopy)\b", q_lower))
    if procedure_words and not procedure_words.intersection(ctx_tokens):
        return True

    if "colectomy" in q_lower and "colectomy" not in ctx_tokens:
        return True

    if re.search(r"\b(hey|hello|hi)\b", q_lower) or re.search(r"\bhow are you( today)?\b", q_lower):
        return True

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
        "how can i help you today",
        "what brings you in today",
    )
    for phrase in conversational_phrases:
        if phrase in q_lower:
            return True

    prior_context_phrases = (
        "since the last time we spoke",
        "last time we spoke",
        "since we last spoke",
        "last time we talked",
        "since we talked",
        "as we discussed",
        "as we talked about",
        "last visit",
        "since your last visit",
        "since the last visit",
        "since your visit",
        "since your previous visit",
    )
    for phrase in prior_context_phrases:
        if phrase in q_lower:
            return True

    # Block timeline questions that reference specific months/years unless the
    # patient's context already contains similar long-duration cues.
    if re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        q_lower,
    ):
        ctx_lower = str(context_text).lower()
        if not re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", ctx_lower) and not re.search(
            r"\b(month|months|year|years)\b",
            ctx_lower,
        ):
            return True

    if re.search(r"\b(a couple months|couple of months|months ago)\b", q_lower):
        ctx_lower = str(context_text).lower()
        if not re.search(r"\b(month|months|year|years|week|weeks)\b", ctx_lower):
            return True

    # Blood-thinner medication checks are appropriate in cardio/thrombotic contexts,
    # but are off-topic drift for unrelated complaints.
    if re.search(r"\bblood thinners?\b", q_lower):
        relevant = {
            "chest",
            "pain",
            "breath",
            "breathless",
            "shortness",
            "heart",
            "stroke",
            "clot",
            "dvt",
            "pe",
            "afib",
            "arrhythmia",
            "palpitations",
            "warfarin",
            "apixaban",
            "rivaroxaban",
            "heparin",
        }
        if not relevant.intersection(ctx_tokens):
            return True

    # Out-of-context wheezing questions should only appear for respiratory complaints
    # or allergic/dermatologic presentations.
    if re.search(r"\bwheez", q_lower):
        respiratory_ctx = {"breath", "breathless", "shortness", "sob", "cough", "asthma", "wheeze"}
        allergy_ctx = {"rash", "hives", "urticaria", "itch", "itchy", "itching", "allergy", "allergies", "allergic"}
        if not (respiratory_ctx.intersection(ctx_tokens) or allergy_ctx.intersection(ctx_tokens)):
            return True

    # Administrative / referral-style scripts are not patient-specific follow-ups.
    if re.search(r"\breferral\b", q_lower) or re.search(r"\bit looks like (?:the )?referral\b", q_lower):
        return True

    # Allergy presuppositions like "I have that you're allergic to penicillin" should
    # be blocked unless the patient context already contains that allergy.
    if re.search(r"\byou(?:'re| are) allergic to\b", q_lower) or re.search(r"\bi have that you(?:'re| are) allergic\b", q_lower):
        # If the context doesn't mention allergies at all, treat as incompatible.
        if not ({"allergy", "allergies", "allergic"}.intersection(ctx_tokens)):
            return True
        # If a specific allergen/med is named but not present in context tokens, block.
        m_all = re.search(r"\ballergic to\s+([a-z]{3,})\b", q_lower)
        if m_all:
            allergen = m_all.group(1)
            if allergen and allergen not in ctx_tokens:
                return True

    if re.search(r"\b(did they|have they|are they|were they)\b", q_lower):
        if not ({"doctor", "doctors", "gp", "hospital", "clinic", "nurse", "ambulance", "er", "a&e", "emergency"}.intersection(ctx_tokens)):
            return True

    # Specific medication-name presuppositions (common in dermatology dialogues).
    # If the question references "with/using the <drug>" and that drug token is
    # not present in patient context, treat as incompatible.
    m_med = re.search(r"\b(?:with|using)\s+the\s+([a-z]{10,})\b", q_lower)
    if m_med:
        drug = m_med.group(1)
        if drug and drug not in ctx_tokens:
            return True

    if re.search(r"\b(past|over the past)\b", q_lower) and re.search(r"\byear|years\b", q_lower):
        ctx_lower = str(context_text).lower()
        if not re.search(r"\byear|years|for years|long time|chronic\b", ctx_lower):
            return True
    if re.search(r"\b(two|2) years\b", q_lower):
        ctx_lower = str(context_text).lower()
        if not re.search(r"\byear|years|for years|long time|chronic\b", ctx_lower):
            return True

    if re.search(r"\b(the )?pain\b", q_lower):
        if not ({"pain", "ache", "aches", "hurt", "hurts"}.intersection(ctx_tokens)):
            return True

    # Exertion/walking presuppositions.
    if re.search(r"\b(walking|walk|stairs?|exercise|exertion|exertional)\b", q_lower):
        if not ({"walk", "walking", "stairs", "stair", "exercise", "exertion"}.intersection(ctx_tokens)):
            exertion_relevant = {"breath", "breathless", "shortness", "chest", "wheeze", "palpitations", "dizzy", "faint"}
            if not exertion_relevant.intersection(ctx_tokens):
                return True

    # "Swelling in your legs" presupposition (edema) is usually out-of-context unless
    # the patient mentioned swelling/edema/legs.
    if re.search(r"\bswelling\b", q_lower) and re.search(r"\blegs?\b", q_lower):
        # Require edema/swelling context (not merely the word "leg" from sciatica/back pain).
        swelling_ctx = {"swelling", "swollen", "edema", "oedema", "ankle", "ankles", "feet", "foot"}
        if not swelling_ctx.intersection(ctx_tokens):
            return True

    # Upper-limb questions should require upper-limb context.
    if re.search(r"\b(hand|hands|arm|arms|grip|gripping)\b", q_lower):
        upper_ctx = {"hand", "hands", "arm", "arms", "grip", "gripping"}
        if not upper_ctx.intersection(ctx_tokens):
            return True

    # Orthopnea-style questions ("problems lying flat") should require some respiratory hint.
    if re.search(r"\blying flat\b|\bflat at night\b", q_lower):
        if not ({"breath", "breathless", "shortness", "sob", "chest", "wheeze"}.intersection(ctx_tokens)):
            return True

    if "acne" in q_lower and "acne" not in ctx_tokens:
        return True

    m = re.search(r"\b(mr|mrs|ms|miss)\s+([a-z]{2,})\b", q_lower)
    if m:
        surname = m.group(2)
        if surname and surname not in ctx_tokens:
            return True

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
            phrase_tokens = tokenize_for_followups(phrase)
            if phrase_tokens and not phrase_tokens.intersection(ctx_tokens):
                return True

    if topic == "meds":
        q_tokens = tokenize_for_followups(q_lower)
        for tok in q_tokens:
            if tok in _COMMON_MED_TOKENS:
                continue
            if len(tok) >= 9 and tok not in ctx_tokens:
                return True

    if topic == "pmh":
        if {"gi", "gastro", "liver", "cancer", "cirrhosis", "hepatitis"}.intersection(tokenize_for_followups(q_lower)) and not (
            {"gi", "gastro", "liver", "cancer", "hepatitis"}.intersection(ctx_tokens)
        ):
            return True

    if "loss of bowel" in q_lower or "loss of bladder" in q_lower or "bowel or bladder function" in q_lower:
        if not ({"bowel", "bladder", "back", "spine", "leg"}.intersection(ctx_tokens)):
            return True

    return False


def required_topics_covered(mem: PatientMemory, topics_covered: set[str]) -> set[str]:
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

    # Required slots: meds + allergies (as volunteered by patient).
    if bool(getattr(mem, "meds", None)) or ("meds" in topics_covered):
        covered.add("meds")
    if bool(getattr(mem, "allergies", None)) or ("allergies" in topics_covered):
        covered.add("allergies")
    return covered


def missing_required_topics(mem: PatientMemory, topics_covered: set[str]) -> list[str]:
    covered = required_topics_covered(mem, topics_covered)
    missing = [t for t in _REQUIRED_TOPICS if t not in covered]
    return missing


def get_followup_shortlists(
    *,
    context_text: str,
    query_text: str,
    asked_lower: set[str],
    topics_covered: set[str],
    k: int = 80,
) -> tuple[list[str], list[str], list[str]]:
    agent = FollowupAgent(
        retrieve=lambda q, intent, kk: get_questions_by_intent(q, intent=str(intent), k=int(kk)),
        tokenize=tokenize_for_followups,
        anchor_tokens=anchor_tokens_from_context,
        domain_tokens=domain_tokens,
        followup_topic=followup_topic,
        clean=clean_followup_question,
        looks_like_question=looks_like_question_for_followups,
        is_redundant=is_redundant_followup,
        is_contradictory=is_contradictory_followup,
        is_incompatible=is_incompatible_followup,
    )

    return agent.get_followup_shortlists(
        context_text=context_text,
        query_text=query_text,
        asked_lower=asked_lower,
        topics_covered=topics_covered,
        k=int(k),
    )


def pick_next_followup(
    *,
    context_text: str,
    query_text: str,
    asked_lower: set[str],
    asked_questions: Iterable[str] | None = None,
    topics_covered: set[str],
    patient_memory: PatientMemory,
    asked_required_slots: set[str] | None = None,
    k: int = 80,
) -> str | None:
    # Retrieval-first, dataset-authoritative behavior:
    # - Build retrieval query from patient memory (done by caller)
    # - Retrieve dataset questions
    # - Ask the FIRST unused question verbatim
    # - Stop when retrieval is exhausted
    def _normalize_for_checklist(q: str) -> str:
        # Keep it deterministic and lightweight; do NOT rewrite output.
        # This is used only for classifying redundancy.
        s = " ".join(str(q or "").lower().split())
        s = re.sub(r"^[\s\.,;:!-]+", "", s)
        s = re.sub(r"^and\b[ ,.-]*", "", s)
        s = re.sub(r"^[\s,]*(um|uh|okay|alright|all right)\s*[,:-]*\s*", "", s)
        return s

    def _is_presuppositional(q: str) -> bool:
        qn = _normalize_for_checklist(q)
        return any(
            p in qn
            for p in (
                "you were endorsing",
                "it sounds like you're endorsing",
                "sounds like you're endorsing",
                "just to confirm",
            )
        )

    def _is_contradictory_no_symptom(q: str) -> bool:
        # Filter dataset lines like "no headaches ?" when the patient already
        # stated that symptom.
        qn = _normalize_for_checklist(q)
        if not qn:
            return False
        if not qn.startswith("no "):
            return False
        toks = tokenize_for_followups(qn)
        # If the question negates a symptom already in the retrieval query, skip it.
        query_toks = tokenize_for_followups(str(query_text or ""))
        symptom_like = {
            "headache",
            "fever",
            "chills",
            "nausea",
            "vomiting",
            "cough",
            "diarrhea",
            "constipation",
            "rash",
            "pain",
        }
        if toks.intersection(symptom_like) and toks.intersection(query_toks):
            return True
        return False

    _CHECKLIST_START_RE = re.compile(
        r"\b(any other|any|are you having|are you experiencing|have you had|have you been|do you have|how about)\b",
        re.IGNORECASE,
    )
    _CHECKLIST_SYMPTOMS = {
        "fever",
        "chills",
        "nausea",
        "vomiting",
        "diarrhea",
        "constipation",
        "cough",
        "sore",
        "throat",
        "rash",
        "abdominal",
        "abdomen",
        "belly",
        "stomach",
        "chest",
        "breath",
        "breathless",
        "shortness",
        "dizzy",
        "dizziness",
        "headache",
    }

    def _is_checklist_question(q: str) -> bool:
        qn = _normalize_for_checklist(q)
        if not qn or "?" not in qn:
            return False
        toks = tokenize_for_followups(qn)
        hits = len({t for t in toks if t in _CHECKLIST_SYMPTOMS})
        # Catch common multi-symptom screens like:
        # "fever or chills", "nausea or vomiting", "any abdominal pain, fever, chills".
        # We intentionally keep this broad to prevent repeated checklist loops.
        return hits >= 2

    def _is_near_duplicate(candidate: str, asked: list[str]) -> bool:
        ctoks = tokenize_for_followups(candidate)
        if not ctoks:
            return False
        # Stronger duplicate handling for med/allergy/pain loops.
        cat = {
            "meds" if {"med", "medication", "medications", "medicine", "medicines", "tablet", "tablets"}.intersection(ctoks) else "",
            "allergies" if {"allerg", "allergy", "allergies"}.intersection(ctoks) else "",
            "pain" if {"pain", "hurt", "hurts", "ache", "aches"}.intersection(ctoks) else "",
            "blood" if {"blood", "urine", "stool", "poop", "bowel"}.intersection(ctoks) else "",
        }
        tight = bool({"meds", "allergies", "pain", "blood"}.intersection(cat))

        for prev in asked:
            ptoks = tokenize_for_followups(prev)
            if not ptoks:
                continue
            inter = len(ctoks.intersection(ptoks))
            union = len(ctoks.union(ptoks))
            if union <= 0:
                continue
            j = inter / union
            # General near-duplicate threshold.
            if j >= 0.72:
                return True
            # Tight categories repeat with tiny wording differences; clamp harder.
            if tight and j >= 0.55:
                return True
        return False

    def _is_untriggered_immunosuppression_med_question(q: str) -> bool:
        """Block very specific med-history questions unless context supports it.

        We must keep questions verbatim from dataset, but we can deterministically
        avoid off-topic drift (e.g., asking about immunosuppressants for a simple
        fever/headache complaint).
        """
        ql = " ".join(str(q).lower().split())
        if not re.search(r"\bimmunosuppress", ql):
            return False

        ctxl = " ".join(str(context_text or "").lower().split())
        triggers = (
            "immunosuppress",
            "immunosuppressed",
            "transplant",
            "steroid",
            "prednisone",
            "chemotherapy",
            "chemo",
            "cancer",
            "hiv",
            "aids",
            "autoimmune",
            "lupus",
            "rheumatoid",
            "methotrexate",
            "biologic",
        )
        return not any(t in ctxl for t in triggers)

    asked_list = [str(x) for x in (asked_questions or []) if str(x).strip()]
    asked_has_checklist = any(_is_checklist_question(q) for q in asked_list)

    # Prefer covering required intents early (still dataset-only; just a deterministic
    # ordering among retrieved candidates).
    missing_required = set(missing_required_topics(patient_memory, set(topics_covered)))

    def _topic_already_covered(q: str) -> bool:
        t = followup_topic(q)
        # Only block repeats for intents we treat as "slots".
        return t in {"duration", "severity", "temp_measurement", "meds", "allergies"} and t in set(topics_covered)

    # Retrieval candidates are drawn from similar real dialogues (dataset `src`).
    # This keeps follow-ups coherent and avoids drifting into unrelated generic
    # screens (e.g., blood thinners) when the query is short.
    take = min(max(10, int(k)), 40)
    pool = 16
    candidates = (
        get_subjective_questions(
            str(query_text or ""),
            k=int(take),
            pool_size=int(pool),
            allow_broad_history=True,
            allowed_broad_intents={"duration", "severity", "temp_measurement", "meds", "allergies"},
        )
        or []
    )

    # Stable re-order: prioritize missing required intents when present.
    candidates = sorted(
        candidates,
        key=lambda q: (0 if followup_topic(q) in missing_required else 1),
    )

    # Note: we intentionally avoid per-call global question-bank construction
    # here (it can be expensive). Intent coverage is handled inside the retriever.

    # Pass 1: prefer non-checklist questions (reduces repetitive generic screens).
    for q in candidates:
        qn = " ".join(str(q).split()).strip()
        if not qn:
            continue
        if qn.lower() in asked_lower:
            continue
        if _topic_already_covered(qn):
            continue
        if _is_untriggered_immunosuppression_med_question(qn):
            continue
        if _is_presuppositional(qn):
            continue
        if _is_contradictory_no_symptom(qn):
            continue
        if _is_near_duplicate(qn, asked_list):
            continue
        if _is_checklist_question(qn):
            continue
        return qn

    # Pass 2: allow a checklist question only if we haven't asked one yet.
    if not asked_has_checklist:
        for q in candidates:
            qn = " ".join(str(q).split()).strip()
            if not qn:
                continue
            if qn.lower() in asked_lower:
                continue
            if _topic_already_covered(qn):
                continue
            if _is_untriggered_immunosuppression_med_question(qn):
                continue
            if _is_presuppositional(qn):
                continue
            if _is_contradictory_no_symptom(qn):
                continue
            if _is_near_duplicate(qn, asked_list):
                continue
            if _is_checklist_question(qn):
                return qn

    # Pass 3: as a last resort, return the first unused question of any type.
    for q in candidates:
        qn = " ".join(str(q).split()).strip()
        if not qn:
            continue
        if qn.lower() in asked_lower:
            continue
        # Never ask multiple checklist screens in one session.
        if asked_has_checklist and _is_checklist_question(qn):
            continue
        if _topic_already_covered(qn):
            continue
        if _is_untriggered_immunosuppression_med_question(qn):
            continue
        if _is_presuppositional(qn):
            continue
        if _is_contradictory_no_symptom(qn):
            continue
        if _is_near_duplicate(qn, asked_list):
            continue
        return qn

    return None


def build_context_text(patient_messages: Iterable[str]) -> str:
    parts = [str(x).strip() for x in patient_messages if str(x).strip()]
    return " ".join(parts)

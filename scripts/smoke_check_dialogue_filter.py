"""Smoke-check: ensure clinician questions are not passed into Crew input.

This script runs in "bare mode" (not via `streamlit run`) and will emit
Streamlit warnings about missing ScriptRunContext. That's expected.

Usage (Windows PowerShell):
    ./.venv/Scripts/python scripts/smoke_check_dialogue_filter.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st


def main() -> None:
    # Ensure repo root is importable when running as a script (sys.path[0] is /scripts).
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from app.streamlit_app import _build_dialogue_for_crew, _init_state

    _init_state()
    st.session_state.messages = [
        {"role": "patient", "text": "I have a headache.", "ts": "t"},
        {"role": "doctor", "text": "Have you noticed any fever along with the headache", "ts": "t"},
        {"role": "patient", "text": "Yes, I have a mild fever.", "ts": "t"},
        {"role": "doctor", "text": "Temperature recorded: 38C", "ts": "t"},
        {"role": "doctor", "text": "Any nausea?", "ts": "t"},
        {"role": "patient", "text": "No nausea.", "ts": "t"},
    ]

    dialogue = _build_dialogue_for_crew()

    forbidden = [
        "Have you noticed any fever",
        "Any nausea?",
    ]

    for snippet in forbidden:
        if snippet in dialogue:
            raise AssertionError(f"Clinician question leaked into Crew dialogue: {snippet!r}")

    # Sanity: should still include patient answers and clinician non-question statement.
    required = [
        "[patient] I have a headache.",
        "[patient] Yes, I have a mild fever.",
        "[doctor] Temperature recorded: 38C",
    ]
    for snippet in required:
        if snippet not in dialogue:
            raise AssertionError(f"Expected snippet missing from Crew dialogue: {snippet!r}")

    print("ok: clinician questions filtered")


if __name__ == "__main__":
    main()

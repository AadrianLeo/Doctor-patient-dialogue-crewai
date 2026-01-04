from crewai.tools import BaseTool
from typing import Type, List, Dict, Any
from pydantic import BaseModel, Field
import json
import re


class SubjectiveExtractorInput(BaseModel):
    # CrewAI sometimes passes task context as a JSON string instead of a native list.
    # Accept both and coerce inside the tool.
    parsed_dialogue: Any = Field(
        ...,
        description="Parsed dialogue turns with speaker and text"
    )


def _coerce_parsed_dialogue(value: Any) -> List[Dict[str, str]]:
    if value is None:
        return []

    # Unwrap common nesting patterns.
    if isinstance(value, dict):
        if "parsed_dialogue" in value:
            value = value.get("parsed_dialogue")
        elif "properties" in value and isinstance(value.get("properties"), dict) and "parsed_dialogue" in value["properties"]:
            value = value["properties"].get("parsed_dialogue")

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            value = json.loads(s)
        except Exception:
            return []

    if not isinstance(value, list):
        return []

    out: List[Dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        speaker = item.get("speaker")
        text = item.get("text")
        if speaker is None or text is None:
            continue
        out.append({"speaker": str(speaker), "text": str(text)})
    return out


class SubjectiveExtractorTool(BaseTool):
    name: str = "subjective_extractor"
    description: str = (
        "Extracts subjective clinical information from patient utterances only. "
        "Focuses on symptoms, duration, severity, and context."
    )
    args_schema: Type[BaseModel] = SubjectiveExtractorInput

    def _run(self, parsed_dialogue: Any) -> Dict:
        parsed_dialogue = _coerce_parsed_dialogue(parsed_dialogue)
        patient_texts = [
            turn["text"] for turn in parsed_dialogue
            if turn["speaker"] == "patient"
        ]

        full_text = " ".join(patient_texts).lower()

        symptoms = []
        if "pain" in full_text:
            symptoms.append("pain")
        if "back pain" in full_text:
            symptoms.append("back pain")

        duration_match = re.search(r"(\d+\s*(days|day|weeks|week|months|month))", full_text)
        duration = duration_match.group(1) if duration_match else "not specified"

        severity = "not specified"
        if "severe" in full_text:
            severity = "severe"
        elif "mild" in full_text:
            severity = "mild"

        return {
            "symptoms": list(set(symptoms)),
            "duration": duration,
            "severity": severity,
            "additional_notes": full_text
        }

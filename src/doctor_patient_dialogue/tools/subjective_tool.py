from crewai.tools import BaseTool
from typing import Type, List, Dict
from pydantic import BaseModel, Field
import re


class SubjectiveExtractorInput(BaseModel):
    parsed_dialogue: List[Dict[str, str]] = Field(
        ...,
        description="Parsed dialogue turns with speaker and text"
    )


class SubjectiveExtractorTool(BaseTool):
    name: str = "subjective_extractor"
    description: str = (
        "Extracts subjective clinical information from patient utterances only. "
        "Focuses on symptoms, duration, severity, and context."
    )
    args_schema: Type[BaseModel] = SubjectiveExtractorInput

    def _run(self, parsed_dialogue: List[Dict[str, str]]) -> Dict:
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

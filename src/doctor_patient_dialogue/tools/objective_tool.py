from crewai.tools import BaseTool
from typing import Type, List, Dict
from pydantic import BaseModel, Field
import re


class ObjectiveExtractorInput(BaseModel):
    parsed_dialogue: List[Dict[str, str]] = Field(
        ...,
        description="Parsed dialogue turns with speaker and text"
    )


class ObjectiveExtractorTool(BaseTool):
    name: str = "objective_extractor"
    description: str = (
        "Extracts ONLY explicitly stated objective clinical findings "
        "from clinician (doctor) statements. Never infer or assume values."
    )
    args_schema: Type[BaseModel] = ObjectiveExtractorInput

    def _run(self, parsed_dialogue: List[Dict[str, str]]) -> Dict:
        doctor_texts = [
            turn["text"].lower()
            for turn in parsed_dialogue
            if turn["speaker"] == "doctor"
        ]

        vitals = []
        physical_exam = []
        lab_results = []
        imaging_results = []

        for text in doctor_texts:
            # --- VITALS (extractive-only, keyword based) ---
            if re.search(r"\bblood pressure\b", text):
                vitals.append("blood pressure mentioned")

            if re.search(r"\bpulse\b|\bheart rate\b", text):
                vitals.append("pulse mentioned")

            if re.search(r"\brespirations?\b|\brespiratory rate\b", text):
                vitals.append("respiratory rate mentioned")

            if re.search(r"\btemperature\b", text):
                vitals.append("temperature mentioned")

            # --- PHYSICAL EXAM ---
            if re.search(r"\bon exam\b|\bphysical exam\b|\btender\b|\bswelling\b", text):
                physical_exam.append(text)

            # --- LAB RESULTS ---
            if re.search(r"\blab\b|\bblood work\b|\bcbc\b|\bmp\b", text):
                lab_results.append(text)

            # --- IMAGING ---
            if re.search(r"\bx-ray\b|\bct\b|\bmri\b|\bultrasound\b", text):
                imaging_results.append(text)

        return {
            "vitals": vitals,
            "physical_exam": physical_exam,
            "lab_results": lab_results,
            "imaging_results": imaging_results,
            "additional_notes": None
        }

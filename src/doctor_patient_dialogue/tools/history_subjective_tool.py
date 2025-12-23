from crewai.tools import BaseTool
from typing import Type, Dict, List
from pydantic import BaseModel, Field


class HistorySubjectiveInput(BaseModel):
    parsed_dialogue: List[Dict[str, str]] = Field(
        ...,
        description="Parsed dialogue turns from the dialogue parser task"
    )



class HistorySubjectiveExtractorTool(BaseTool):
    name: str = "history_subjective_extractor"
    description: str = (
        "Extracts patient-reported medical history ONLY. "
        "Do not infer or invent information."
    )
    args_schema: Type[BaseModel] = HistorySubjectiveInput

    def _run(self, parsed_dialogue: List[Dict[str, str]]) -> Dict:
        history = {
            "past_medical_history": [],
            "medications": [],
            "family_history": [],
            "social_history": [],
            "additional_notes": ""
        }

        if not parsed_dialogue:
            return history

        for turn in parsed_dialogue:
            if turn.get("speaker") != "patient":
                continue

            text = turn.get("text", "").lower()

            if any(k in text for k in ["diagnosed", "history of", "condition"]):
                history["past_medical_history"].append(turn["text"])

            if any(k in text for k in ["medication", "taking", "tablet", "pill"]):
                history["medications"].append(turn["text"])

            if "family" in text:
                history["family_history"].append(turn["text"])

            if any(k in text for k in ["smoke", "smoking", "alcohol", "drink"]):
                history["social_history"].append(turn["text"])

        return history


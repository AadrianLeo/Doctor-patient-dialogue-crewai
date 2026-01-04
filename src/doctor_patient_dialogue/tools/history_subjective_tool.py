from crewai.tools import BaseTool
from typing import Type, Dict, List, Any
from pydantic import BaseModel, Field, model_validator
import json


class HistorySubjectiveInput(BaseModel):
    # CrewAI sometimes passes task context as a JSON string instead of a native list.
    # It can also incorrectly wrap the payload inside a `properties` key.
    parsed_dialogue: Any | None = Field(
        ...,
        description="Parsed dialogue turns from the dialogue parser task"
    )

    @model_validator(mode="before")
    @classmethod
    def _unwrap_properties(cls, data: Any):
        if isinstance(data, dict) and ("parsed_dialogue" not in data) and ("properties" in data):
            props = data.get("properties")
            if isinstance(props, dict) and ("parsed_dialogue" in props):
                return {"parsed_dialogue": props.get("parsed_dialogue")}
        return data
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




class HistorySubjectiveExtractorTool(BaseTool):
    name: str = "history_subjective_extractor"
    description: str = (
        "Extracts patient-reported medical history ONLY. "
        "Do not infer or invent information."
    )
    args_schema: Type[BaseModel] = HistorySubjectiveInput

    def _run(self, parsed_dialogue: Any) -> Dict:
        parsed_dialogue = _coerce_parsed_dialogue(parsed_dialogue)
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


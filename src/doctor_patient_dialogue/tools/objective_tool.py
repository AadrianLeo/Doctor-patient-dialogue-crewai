from crewai.tools import BaseTool
from typing import Type, Dict, List, Any
from pydantic import BaseModel, Field
import json


class ObjectiveExtractorInput(BaseModel):
    # CrewAI sometimes passes task context as a JSON string instead of a native list.
    parsed_dialogue: Any = Field(
        ...,
        description="Parsed dialogue turns from the dialogue parser task"
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


class ObjectiveExtractorTool(BaseTool):
    name: str = "objective_extractor"
    description: str = (
        "Extracts ONLY clinician-observed objective findings "
        "from parsed dialogue. Never infer or add data."
    )
    args_schema: Type[BaseModel] = ObjectiveExtractorInput

    def _run(self, parsed_dialogue: Any) -> Dict:
        parsed_dialogue = _coerce_parsed_dialogue(parsed_dialogue)
        output = {
            "vitals": [],
            "physical_exam": [],
            "lab_results": [],
            "imaging_results": [],
            "additional_notes": ""
        }

        if not parsed_dialogue:
            return output

        for turn in parsed_dialogue:
            if turn.get("speaker") != "doctor":
                continue

            text = turn.get("text", "").lower()

            if "blood pressure" in text or "pulse" in text or "temperature" in text:
                output["vitals"].append(turn["text"])

            elif "exam" in text or "tender" in text or "palpation" in text:
                output["physical_exam"].append(turn["text"])

            elif "lab" in text or "blood test" in text:
                output["lab_results"].append(turn["text"])

            elif "x-ray" in text or "ct" in text or "mri" in text:
                output["imaging_results"].append(turn["text"])

        return output

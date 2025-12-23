from crewai.tools import BaseTool
from typing import Type, Dict, List
from pydantic import BaseModel, Field


class ObjectiveExtractorInput(BaseModel):
    parsed_dialogue: List[Dict[str, str]] = Field(
        ...,
        description="Parsed dialogue turns from the dialogue parser task"
    )


class ObjectiveExtractorTool(BaseTool):
    name: str = "objective_extractor"
    description: str = (
        "Extracts ONLY clinician-observed objective findings "
        "from parsed dialogue. Never infer or add data."
    )
    args_schema: Type[BaseModel] = ObjectiveExtractorInput

    def _run(self, parsed_dialogue: List[Dict[str, str]]) -> Dict:
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

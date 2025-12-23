from crewai.tools import BaseTool
from typing import Type, List, Dict
from pydantic import BaseModel, Field, validator
import re


class DialogueParserInput(BaseModel):
    dialogue: str = Field(
        ...,
        description="Raw doctor–patient dialogue with [doctor] and [patient] tags."
    )

    @validator("dialogue")
    def validate_dialogue_format(cls, v):
        if not re.search(r"\[(doctor|patient)\]", v, re.IGNORECASE):
            raise ValueError(
                "Dialogue must contain at least one [doctor] or [patient] tag."
            )
        return v


class DialogueParserTool(BaseTool):
    name: str = "dialogue_parser"
    description: str = (
        "Parses raw doctor–patient dialogue into structured speaker turns. "
        "Extractive only. No interpretation."
    )
    args_schema: Type[BaseModel] = DialogueParserInput

    def _run(self, dialogue: str) -> List[Dict[str, str]]:
        """
        Handles multiple speaker turns on the SAME line or across lines.
        """
        pattern = re.compile(r"\[(doctor|patient)\]\s*([^[]+)", re.IGNORECASE)
        turns = []

        for match in pattern.finditer(dialogue):
            speaker = match.group(1).lower()
            text = match.group(2).strip()

            if text:
                turns.append({
                    "speaker": speaker,
                    "text": text
                })

        return turns

from crewai.tools import BaseTool
from typing import Type, List, Dict
from pydantic import BaseModel, Field
import re


class DialogueParserInput(BaseModel):
    """Input schema for Dialogue Parser Tool."""
    dialogue: str = Field(
        ...,
        description="Raw doctor–patient dialogue text with [doctor] and [patient] tags."
    )


class DialogueParserTool(BaseTool):
    name: str = "dialogue_parser"
    description: str = (
        "Parses a raw doctor–patient dialogue into structured speaker turns. "
        "This tool only structures the dialogue and does not summarize or interpret."
    )
    args_schema: Type[BaseModel] = DialogueParserInput

    def _run(self, dialogue: str) -> List[Dict[str, str]]:
        pattern = re.compile(r"\[(doctor|patient)\]\s*(.+)", re.IGNORECASE)
        turns = []

        for line in dialogue.splitlines():
            line = line.strip()
            if not line:
                continue

            match = pattern.match(line)
            if match:
                speaker = match.group(1).lower()
                text = match.group(2).strip()
                turns.append({
                    "speaker": speaker,
                    "text": text
                })

        return turns

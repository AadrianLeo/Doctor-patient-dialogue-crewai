from crewai.tools import BaseTool
from typing import Type, List, Dict
from pydantic import BaseModel, Field
import re


# =========================
# Validator
# =========================

class DialogueFormatError(ValueError):
    """Raised when dialogue format is invalid."""
    pass


def validate_dialogue_format(dialogue: str) -> List[str]:
    """
    Validate that the dialogue is line-based and properly tagged.

    Expected format:
    [doctor] ...
    [patient] ...

    Returns validated dialogue lines.
    Raises DialogueFormatError if invalid.
    """
    if not dialogue or not dialogue.strip():
        raise DialogueFormatError("Dialogue input is empty.")

    lines = [line.strip() for line in dialogue.splitlines() if line.strip()]

    if not lines:
        raise DialogueFormatError("Dialogue contains no valid lines.")

    pattern = re.compile(r"^\[(doctor|patient)\]\s+.+", re.IGNORECASE)

    for line in lines:
        if not pattern.match(line):
            raise DialogueFormatError(
                f"Invalid dialogue line format: '{line}'. "
                "Expected format: [doctor] ... or [patient] ..."
            )

    return lines


# =========================
# Tool Input Schema
# =========================

class DialogueParserInput(BaseModel):
    """Input schema for Dialogue Parser Tool."""
    dialogue: str = Field(
        ...,
        description="Doctor–patient dialogue with one utterance per line, "
                    "each starting with [doctor] or [patient]."
    )


# =========================
# Dialogue Parser Tool
# =========================

class DialogueParserTool(BaseTool):
    name: str = "dialogue_parser"
    description: str = (
        "Parses validated doctor–patient dialogue into structured speaker turns. "
        "This tool is strictly structural and performs no medical reasoning."
    )
    args_schema: Type[BaseModel] = DialogueParserInput

    def _run(self, dialogue: str) -> List[Dict[str, str]]:
        # 1️⃣ Validate dialogue format
        lines = validate_dialogue_format(dialogue)

        # 2️⃣ Parse validated lines
        parsed_turns = []
        pattern = re.compile(r"^\[(doctor|patient)\]\s+(.*)", re.IGNORECASE)

        for line in lines:
            match = pattern.match(line)
            speaker, text = match.groups()

            parsed_turns.append({
                "speaker": speaker.lower(),
                "text": text.strip()
            })

        return parsed_turns

import re
from typing import List


class DialogueFormatError(ValueError):
    """Raised when dialogue format is invalid."""
    pass


def validate_dialogue_format(dialogue: str) -> List[str]:
    """
    Validates that the dialogue is line-based and properly tagged.
    Returns valid dialogue lines if successful.
    Raises DialogueFormatError otherwise.
    """

    if not dialogue or not dialogue.strip():
        raise DialogueFormatError("Dialogue input is empty.")

    lines = [line.strip() for line in dialogue.splitlines() if line.strip()]

    if not lines:
        raise DialogueFormatError("Dialogue contains no valid lines.")

    valid_lines = []
    pattern = re.compile(r"^\[(doctor|patient)\]\s+.+", re.IGNORECASE)

    for line in lines:
        if not pattern.match(line):
            raise DialogueFormatError(
                f"Invalid dialogue line format: '{line}'. "
                "Expected format: [doctor] ... or [patient] ..."
            )
        valid_lines.append(line)

    return valid_lines

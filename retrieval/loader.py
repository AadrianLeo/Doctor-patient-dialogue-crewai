from pathlib import Path
import json

# Resolve relative to the repository root (robust to running from any CWD).
BASE_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_json(relative_path: str):
    path = BASE_DATA_DIR / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

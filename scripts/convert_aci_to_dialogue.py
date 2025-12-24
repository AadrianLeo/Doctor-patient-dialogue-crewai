import json
from pathlib import Path

ACI_PATH = Path("data/raw/train_full.json")
OUT_PATH = Path("data/processed/aci_dialogues.json")

def convert():
    with open(ACI_PATH, "r", encoding="utf-8") as f:
        aci_data = json.load(f)

    converted = []

    for sample in aci_data:
        encounter_id = sample.get("encounter_id", "unknown")

        dialogue_lines = []
        for turn in sample.get("conversation", []):
            speaker = turn["speaker"].lower()
            text = turn["text"].strip()
            dialogue_lines.append(f"[{speaker}] {text}")

        dialogue = "\n".join(dialogue_lines)

        converted.append({
            "id": encounter_id,
            "dialogue": dialogue
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    print(f"✅ Converted {len(converted)} ACI-Bench dialogues")

if __name__ == "__main__":
    convert()

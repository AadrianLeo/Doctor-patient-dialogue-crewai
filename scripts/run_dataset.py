import json
from pathlib import Path
from doctor_patient_dialogue.crew import DoctorPatientDialogue

RAW_PATH = Path("data/raw/train_full.json")
OUT_PATH = Path("data/processed/train_outputs.json")

def run_dataset(limit: int = 5):
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"{RAW_PATH} not found")

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    samples = dataset["data"]   # ✅ THIS IS THE KEY FIX

    crew = DoctorPatientDialogue().crew()
    results = []

    for i, sample in enumerate(samples[:limit]):
        dialogue = sample["src"]
        target = sample.get("tgt", "")
        sample_id = sample.get("file", f"sample_{i}")

        print(f"\n▶ Processing {sample_id}")

        crew_result = crew.kickoff(inputs={"dialogue": dialogue})

        # CrewAI-version safe extraction
        if hasattr(crew_result, "json_dict") and crew_result.json_dict:
            output = crew_result.json_dict
        elif hasattr(crew_result, "final_output") and crew_result.final_output:
            output = crew_result.final_output
        else:
            output = crew_result.raw

        results.append({
            "id": sample_id,
            "output": output,
            "target_note": target
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Phase 4.2 complete — outputs saved to {OUT_PATH}")

if __name__ == "__main__":
    run_dataset(limit=5)

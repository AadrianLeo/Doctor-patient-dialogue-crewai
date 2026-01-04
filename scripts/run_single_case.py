from doctor_patient_dialogue.crew import DoctorPatientDialogue
from retrieval.subjective_retriever import get_subjective_questions
from retrieval.objective_retriever import get_objective_exams
from retrieval.plan_retriever import get_dataset_plans

import json


def _safe_load_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def _normalize_assessment_json(assessment_json):
    """Fix common JSON malformations in assessment output.
    
    The LLM sometimes produces:
    {"assessment": {0: {...}, 1: {...}}} instead of {"assessment": [{...}, {...}]}
    
    This normalizes it to proper JSON array format.
    """
    if not isinstance(assessment_json, dict):
        return assessment_json
    
    if "assessment" not in assessment_json:
        return assessment_json
    
    assessment = assessment_json["assessment"]
    
    # If assessment is a dict with numeric string keys, convert to array
    if isinstance(assessment, dict):
        # Check if keys are numeric
        try:
            items = []
            keys = sorted([int(k) for k in assessment.keys()])
            for k in keys:
                items.append(assessment[str(k)])
            assessment_json["assessment"] = items
        except (ValueError, KeyError):
            # Not numeric keys, leave as is
            pass
    
    return assessment_json

def run_case(full_dialogue):
    subj_qs = get_subjective_questions(full_dialogue)
    obj_exams = get_objective_exams(full_dialogue)
    dataset_plans = get_dataset_plans(full_dialogue, k=1)

    crew = DoctorPatientDialogue().crew()

    result = crew.kickoff(inputs={
        "dialogue": full_dialogue,
        "retrieved_subjective_questions": subj_qs,
        "retrieved_objective_exams": obj_exams,
        "retrieved_dataset_plans": dataset_plans,
    })

    assessment_raw = None
    care_raw = None
    subjective_raw = None
    history_subjective_raw = None
    objective_raw = None
    
    for out in (getattr(result, "tasks_output", None) or []):
        name = getattr(out, "name", "")
        if name == "assessment_plan_task":
            assessment_raw = getattr(out, "raw", None)
        elif name == "care_plan_task":
            care_raw = getattr(out, "raw", None)
        elif name == "subjective_task":
            subjective_raw = getattr(out, "raw", None)
        elif name == "history_subjective_task":
            history_subjective_raw = getattr(out, "raw", None)
        elif name == "objective_task":
            objective_raw = getattr(out, "raw", None)

    if assessment_raw is None:
        assessment_raw = getattr(result, "raw", "")

    assessment_json = _safe_load_json(assessment_raw) if isinstance(assessment_raw, str) else assessment_raw
    care_json = _safe_load_json(care_raw) if isinstance(care_raw, str) else care_raw
    subjective_json = _safe_load_json(subjective_raw) if isinstance(subjective_raw, str) else subjective_raw
    history_subjective_json = _safe_load_json(history_subjective_raw) if isinstance(history_subjective_raw, str) else history_subjective_raw
    objective_json = _safe_load_json(objective_raw) if isinstance(objective_raw, str) else objective_raw

    # Normalize malformed assessment JSON
    if assessment_json is not None:
        assessment_json = _normalize_assessment_json(assessment_json)

    payload = {
        "assessment_plan": assessment_json if isinstance(assessment_json, dict) else {},
        "care_plan": care_json if isinstance(care_json, dict) else None,
        "subjective": subjective_json if isinstance(subjective_json, dict) else {},
        "history_subjective": history_subjective_json if isinstance(history_subjective_json, dict) else {},
        "objective": objective_json if isinstance(objective_json, dict) else {},
    }

    return json.dumps(payload, ensure_ascii=False)

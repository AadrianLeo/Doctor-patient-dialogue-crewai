import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import json

# Inline the function to test it
def _normalize_assessment_json(assessment_json):
    """Fix common JSON malformations in assessment output."""
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

# Test malformed JSON (numeric keys as strings)
bad = {
    'assessment': {
        '0': {'problem': 'fever', 'supporting_evidence': ['hello i have fever']},
        '1': {'problem': 'headache', 'supporting_evidence': ['hello i have fever and headache']},
        '2': {'problem': 'chills', 'supporting_evidence': ['yes i have chills']}
    },
    'plan': 'insufficient information'
}

print('Before normalization:')
print(json.dumps(bad, indent=2))

fixed = _normalize_assessment_json(bad)

print('\nAfter normalization:')
print(json.dumps(fixed, indent=2))

# Verify it's a proper array now
assert isinstance(fixed['assessment'], list), "Assessment should be a list"
assert len(fixed['assessment']) == 3, "Should have 3 items"
assert fixed['assessment'][0]['problem'] == 'fever', "First item should be fever"
print('\n✅ JSON normalization works correctly!')

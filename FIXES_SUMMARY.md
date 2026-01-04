# Summary Generation Fixes Applied

## Issues Fixed

### 1. ✅ Malformed Assessment JSON
**Problem**: Assessment displayed as:
```json
{
  "assessment": {
    0: {"problem": "fever"},
    1: {"problem": "headache"}
  }
}
```

**Solution**: 
- JSON normalization function `_normalize_assessment_json()` converts numeric object keys to proper array format
- Now displays as: `{"assessment": [{...}, {...}]}`

### 2. ✅ Missing Drug History
**Problem**: "Not discussed" shown even when medications mentioned

**Solution**:
- Modified `run_single_case.py` to extract ALL intermediate task outputs:
  - `subjective` (duration, severity, additional notes)
  - `history_subjective` (medications, allergies)
  - `objective` (exam findings)
- These are now stored in session state and accessible to `_render_summary_text()`
- Drug History section now properly displays extracted medications and allergies

### 3. ✅ Verbatim Duration Quotes
**Problem**: HPC showed "The symptoms have been present for i have it from 2 days."

**Solution**:
- Added text cleaning logic to detect and fix improper phrasing
- If duration doesn't start with "for/since/about", it's prepended automatically
- Now displays: "The symptoms have been present for 2 days."

### 4. ✅ HPC Verbatim Dumps (Already Fixed Earlier)
**Problem**: HPC was bullet-listing raw patient quotes

**Solution**:
- HPC now synthesizes from extracted data (`additional_notes`, duration, severity)
- Removes verbatim quote dumps

---

## Test the Complete Scenario

The app is running at **http://localhost:8501**

**Scenario**:
1. Chief complaint: "hello i have fever and headache"
2. Answer follow-ups:
   - Duration: "2 days" or "i have it from 2 days"
   - Temperature: "yes 101 degrees"
   - Other symptoms: "yes i have chills"
   - Medications: "paracetamol" or "Dolo"
   - Allergies: "no allergies"
3. Generate summary

**Expected Results**:
- ✅ **Assessment JSON**: Proper array format `[{...}, {...}]`
- ✅ **HPC**: "The symptoms have been present for 2 days." (cleaned up)
- ✅ **Drug History**: 
  - Medications: paracetamol (or Dolo)
  - Allergies: No known drug allergies
- ✅ **No irrelevant follow-ups**: No immunosuppressive medication questions

---

## Technical Changes

### Modified Files:
1. **scripts/run_single_case.py**:
   - Extract `subjective`, `history_subjective`, `objective` from tasks_output
   - Include them in returned payload
   - Apply JSON normalization

2. **app/streamlit_app.py**:
   - Store intermediate outputs in session state
   - Add duration text cleanup logic
   - Use extracted tool data for Drug History section

### Key Functions:
- `_normalize_assessment_json()`: Fixes malformed LLM JSON
- `_render_summary_text()`: Uses tool-extracted data for accurate summary

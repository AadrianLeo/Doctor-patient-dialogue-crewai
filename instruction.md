📘 Doctor–Patient Clinical Assistant
Retrieval-Driven, Rule-Gated, CrewAI-Summarized System
1. Project Goal (VERY IMPORTANT)

This project builds a clinical assistant that:

interacts with a patient in multi-turn chat

asks only real clinician follow-up questions taken from a medical dialogue dataset

NEVER invents questions

collects sufficient information

produces a strict Assessment & Plan at the end

This is NOT a free-form chatbot.
This is NOT a diagnosis system.
This is NOT an LLM-driven questioning agent.

2. High-Level Architecture (Do NOT change this)

The system is retrieval-driven, rule-controlled, and LLM-assisted only at the end.

Patient input
↓
Patient memory (extractive, no inference)
↓
Build retrieval query
↓
Retrieve candidate clinician questions from dataset JSON
↓
Deterministic filtering (rules)
↓
Ask ONE best next question
↓
Repeat until stop condition
↓
Run CrewAI ONCE to generate assessment & plan

3. Core Design Principle (NON-NEGOTIABLE)
❗ CrewAI must NOT:

decide what question to ask

decide when to stop

generate follow-ups

infer symptoms

invent medical knowledge

✅ CrewAI IS ONLY USED FOR:

structuring extracted information

restating facts

generating final Assessment & Plan (STRICT JSON)

4. Data Source & How It Is Used
Dataset structure:

All JSON files contain the SAME dialogues

Each file has a different target (tgt) section:

subjective.json → follow-up questions

objective_exam.json → exams

assessment_and_plan.json → clinician plans

CRITICAL RULE:

The dialogue text is the same everywhere.
The targets (tgt) are used as retrieval knowledge, NOT as training data.

We DO NOT train anything.

5. Retrieval Strategy (IMPORTANT)
Follow-up questions come ONLY from:

clinician utterances in dataset

extracted into subjective.json

Retrieval process:

Convert patient memory → query text

Retrieve MANY candidate questions

Filter them using rules

Select ONE best question

NO question is invented.

6. Deterministic Filtering Rules (MUST KEEP)

Every candidate question must pass ALL:

❌ Not already asked

❌ Not redundant

❌ Not contradictory
(e.g. asking “diarrhea?” after patient said constipation)

❌ Not incompatible with known context

✅ Matches missing clinical topic

7. Required Clinical Topics (STOP LOGIC)

The system must ask questions until these are covered:

REQUIRED_TOPICS = [
    "chief_complaint",
    "duration",
    "severity",
    "associated_symptoms",
    "red_flags",
    "medications",
    "allergies"
]

STOP condition:

All required topics covered

OR no valid follow-up questions remain

❌ Do NOT stop based on:

number of turns

token overlap

LLM “feels ready”

8. Assessment & Plan Generation (CrewAI)

CrewAI is executed ONCE at the end.

Input:

Full conversation transcript

Extracted subjective + objective facts

Output (STRICT JSON ONLY):
{
  "assessment": [
    {
      "problem": "fever",
      "supporting_evidence": [
        "I have fever for 2 days"
      ]
    }
  ],
  "plan": "insufficient information"
}

ABSOLUTE RULES:

No diagnoses

No inference

No new tests

No explanations

Verbatim evidence only

9. Streamlit UI Behavior

The UI must:

Show incremental chat

Ask ONE question at a time

Disable free text once a question is asked

Show progress (“Collecting history…”, “Ready to summarize…”)

Trigger CrewAI only when STOP condition is met

Display final assessment clearly

10. Files to KEEP (DO NOT DELETE)
Core

retrieval/subjective_retriever.py

patient_memory.py

consultation_store.py

streamlit_app.py

crew.py

agents.yaml

tasks.yaml

Data

data/raw/*

data/section_targets/*

11. Files / Logic to REMOVE or SIMPLIFY
❌ Remove or disable:

LLM deciding follow-ups

readiness classifiers

candidate_id exposure

multiple chat modes

CrewAI follow-up logic

❌ Do NOT use:

dynamic stopping heuristics

“AI decides what to ask”

free generation of questions

12. What You (VSCode ChatGPT) Must Do

You must:

Delete unused or duplicate follow-up agents

Centralize follow-up selection into ONE deterministic pipeline

Ensure all questions come ONLY from dataset retrieval

Enforce strict contradiction & redundancy rules

Simplify Streamlit logic (single path)

Keep CrewAI only for final summary

Make the system stable and predictable

13. Success Criteria

The system is correct when:

Questions never repeat

Questions never contradict patient answers

No invented questions appear

Summary only reflects stated information

UI behaves consistently

Output is deterministic

14. Final Reminder

This project is not about intelligence.
It is about control, traceability, and safety.

Retrieval decides what exists
Rules decide what is allowed
CrewAI decides how to summarize

✅ End of Instructions
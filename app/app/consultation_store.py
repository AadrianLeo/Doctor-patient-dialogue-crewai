from __future__ import annotations

import json
import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class SaveResult:
    path: Path
    consultation_id: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_consultation(payload: Dict[str, Any], output_dir: Path) -> SaveResult:
    """Persist a consultation payload as JSON.

    The payload is saved under: {output_dir}/{consultation_id}.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    consultation_id = payload.get("consultation_id") or str(uuid.uuid4())
    payload = dict(payload)
    payload.setdefault("consultation_id", consultation_id)
    payload.setdefault("saved_at", _utc_now_iso())

    path = output_dir / f"{consultation_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return SaveResult(path=path, consultation_id=consultation_id)


def build_patient_key(emis_id: str, dob: str) -> str:
    """Build a stable, filesystem-safe patient key.

    Uses a SHA-256 hash of normalized (EMIS ID + DOB) so filenames/folders don't
    directly expose the identifiers.
    """

    emis = str(emis_id or "").strip().lower()
    dob_norm = str(dob or "").strip().lower()
    raw = f"emis:{emis}|dob:{dob_norm}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def save_patient_consultation(payload: Dict[str, Any], base_dir: Path, patient_key: str) -> SaveResult:
    """Save a consultation under a patient-specific directory.

    Layout:
      {base_dir}/patients/{patient_key}/{consultation_id}.json
    """

    patient_dir = Path(base_dir) / "patients" / str(patient_key)
    payload = dict(payload)
    payload.setdefault("patient_key", patient_key)
    return save_consultation(payload, output_dir=patient_dir)


def load_patient_consultations(base_dir: Path, patient_key: str, limit: int = 10) -> list[Dict[str, Any]]:
    """Load recent saved consultations for a patient (newest first)."""

    patient_dir = Path(base_dir) / "patients" / str(patient_key)
    if not patient_dir.exists():
        return []

    items: list[tuple[str, Dict[str, Any]]] = []
    for path in sorted(patient_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        saved_at = str(payload.get("saved_at") or "")
        items.append((saved_at, payload))

    # ISO timestamps sort lexicographically.
    items.sort(key=lambda t: t[0], reverse=True)
    return [p for _ts, p in items[: max(0, int(limit))]]

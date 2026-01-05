# ============================================================
# forensic_report.py
# Forensic Evidence Report Generator (IEEE & Audit Ready)
# ============================================================

import json
import hashlib
from datetime import datetime
from typing import List, Tuple, Union


def sha256_of_file(file_path: str, block_size: int = 65536) -> str:
    """
    Computes SHA-256 hash of a file for integrity verification.
    """
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            sha.update(block)
    return sha.hexdigest()


def generate_forensic_report(
    file_path: str,
    modality: str,
    decision: str,
    confidence: float,
    suspicious_segments: Union[List[float], List[Tuple[float, float]]],
    model_name: str,
    output_path: str
) -> str:
    """
    Generates a forensic JSON report.

    Args:
        file_path (str): Path to analyzed file
        modality (str): 'audio' or 'video'
        decision (str): REAL / FAKE
        confidence (float): Fake probability
        suspicious_segments (list): timestamps or frame times
        model_name (str): Model identifier
        output_path (str): Output JSON path

    Returns:
        str: Path to saved report
    """

    report = {
        "analysis_timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "modality": modality.lower(),
        "file_sha256": sha256_of_file(file_path),
        "model_used": model_name,
        "final_decision": decision.upper(),
        "confidence_score": round(float(confidence), 4),
        "confidence_percent": round(float(confidence) * 100, 2),
        "suspicious_regions": suspicious_segments,
        "forensic_note": (
            "Suspicious regions are confidence-based indicators "
            "and do not represent exact ground-truth manipulation boundaries."
        )
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)

    return output_path

"""
Number verification logic for LLM guardrails.
"""
import re
import logging
from typing import Tuple, List, Dict, Optional

logger = logging.getLogger(__name__)

# Number patterns for extraction (handles M, Cr, K suffixes)
NUMBER_PATTERN = re.compile(
    r"""
    (?:₹\s*)?                           # optional ₹ sign
    ([-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)? # number with commas
     |\d+(?:\.\d+)?)                    # or simple number
    \s*(?:M|Cr|K|%|bps|bp)?            # optional unit
    """,
    re.VERBOSE,
)

TOLERANCE_PCT = 0.02  # 2% tolerance for number verification

def extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from commentary text."""
    raw = NUMBER_PATTERN.findall(text)
    numbers = []
    for r in raw:
        try:
            val = float(r.replace(",", ""))
            numbers.append(val)
        except ValueError:
            pass
    return numbers


def verify_numbers(
    extracted: list[float],
    expected: dict[str, float | None],
    tol: float = TOLERANCE_PCT,
) -> tuple[bool, list[str]]:
    """
    Verify that expected key numbers appear in the extracted list.

    Returns (is_valid, list_of_mismatches).
    """
    mismatches = []

    for key, expected_val in expected.items():
        if expected_val is None:
            continue
        if abs(expected_val) < 0.01:
            continue

        # Check if expected_val appears (within tolerance) in extracted list
        found = any(
            abs(x - expected_val) <= max(abs(expected_val) * tol, 0.1)
            for x in extracted
        )

        if not found:
            mismatches.append(f"{key}: expected {expected_val}, not found in text")

    return len(mismatches) == 0, mismatches


def verify_llm_commentary(
    polished_text: str,
    expected_numbers: dict[str, float],
    tol: float = TOLERANCE_PCT,
) -> tuple[bool, list[str]]:
    """
    Guardrail: extract all numbers from LLM-polished text and verify against computed values.
    """
    extracted = extract_numbers(polished_text)
    is_valid, mismatches = verify_numbers(extracted, expected_numbers, tol)

    if not is_valid:
        logger.warning(f"LLM commentary REJECTED due to number mismatches: {mismatches}")

    return is_valid, mismatches

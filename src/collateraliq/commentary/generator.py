"""
Commentary generator with number verifier.

Two-layer system:
1. Deterministic template: fills computed numbers into fixed templates
2. Number verifier: extracts all numbers from the text and checks them
   against the computed values — rejects any mismatch

Optional LLM polish is applied only after the verifier passes.

This demonstrates safe "AI solutions" with a guardrail.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import date
from typing import Any

import numpy as np
from sqlalchemy import text

from collateraliq.data.db import get_engine

logger = logging.getLogger(__name__)

from collateraliq.commentary.templates import build_template
from collateraliq.commentary.number_verifier import extract_numbers, verify_numbers, verify_llm_commentary


def generate_daily_commentary(
    as_of: date,
    client_id: str,
    attribution: dict,
    exposure: float,
    im_held: float,
    vm_held: float,
    vm_call: float,
    coverage_ratio: float,
    alerts: list[dict],
    db_path: str = "data/collateraliq.db",
) -> dict:
    """
    Generate deterministic commentary for one client on one day.

    Returns dict with template_text, polished_text, verified flag.
    """
    # Build template text
    template = build_template(
        as_of, client_id, attribution, exposure, im_held, vm_held, vm_call, coverage_ratio, alerts
    )

    # Extract numbers from template for verification
    template_numbers = extract_numbers(template)

    # Verify template (should always pass — we built it)
    is_valid, mismatches = verify_numbers(
        template_numbers,
        {
            "exposure_m": round(exposure / 1e6, 1),
            "im_m": round(im_held / 1e6, 1),
            "vm_m": round(vm_held / 1e6, 1),
            "coverage_pct": round(coverage_ratio * 100, 0),
            "equity_pct": round(abs(attribution.get("equity_move_pct", 0)), 1),
            "fx_pct": round(abs(attribution.get("fx_move_pct", 0)), 2),
            "vm_call_m": round(abs(vm_call) / 1e6, 1) if abs(vm_call) > 0.5e6 else None,
        },
    )

    # Store commentary
    comm_id = f"COMM_{client_id}_{as_of}"
    _store_commentary(
        comm_id, as_of, client_id, template, template, is_valid,
        template_numbers, db_path
    )

    return {
        "comm_id": comm_id,
        "client_id": client_id,
        "date": str(as_of),
        "template_text": template,
        "polished_text": template,  # Same without LLM
        "verified": is_valid,
        "mismatches": mismatches,
    }




def _store_commentary(
    comm_id: str,
    as_of: date,
    client_id: str,
    template: str,
    polished: str,
    verified: bool,
    numbers: list,
    db_path: str,
) -> None:
    engine = get_engine(db_path)
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO commentary
                    (comm_id, date, client_id, commentary_type, template_text,
                     polished_text, verified, numbers_extracted)
                VALUES
                    (:comm_id, :date, :client_id, :ctype, :template,
                     :polished, :verified, :numbers) ON CONFLICT DO NOTHING
            """),
            {
                "comm_id": comm_id,
                "date": str(as_of),
                "client_id": client_id,
                "ctype": "daily",
                "template": template,
                "polished": polished,
                "verified": int(verified),
                "numbers": json.dumps(numbers),
            },
        )

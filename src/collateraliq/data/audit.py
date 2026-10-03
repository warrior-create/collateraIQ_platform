"""
Audit trail for CollateralIQ.

Every material action (trade creation, margin call, parameter change) is logged
to audit_log with: who, what, when, old value, new value, data hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)


def log_audit(
    engine,
    action: str,
    user_or_process: str = "system",
    table_name: str | None = None,
    record_id: str | None = None,
    old_value: Any = None,
    new_value: Any = None,
    run_id: str | None = None,
    data_hash: str | None = None,
) -> None:
    """Write an audit record to audit_log."""
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO audit_log
                    (timestamp, user_or_process, action, table_name, record_id,
                     old_value, new_value, data_hash, run_id, ip_address)
                VALUES
                    (:ts, :user, :action, :table, :rec_id, :old_val, :new_val, :dhash, :run_id, :ip) ON CONFLICT DO NOTHING
            """),
            {
                "ts": datetime.utcnow().isoformat(),
                "user": user_or_process,
                "action": action,
                "table": table_name,
                "rec_id": record_id,
                "old_val": json.dumps(old_value) if old_value is not None else None,
                "new_val": json.dumps(new_value) if new_value is not None else None,
                "dhash": data_hash,
                "run_id": run_id,
                "ip": "127.0.0.1",
            },
        )


def compute_data_hash(data: Any) -> str:
    """Compute a short SHA-256 hash of any JSON-serialisable object."""
    buf = json.dumps(data, sort_keys=True, default=str).encode()
    return hashlib.sha256(buf).hexdigest()[:16]

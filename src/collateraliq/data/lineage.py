"""
Data lineage tracking for CollateralIQ.

Records the provenance of every computed value:
which market data feeds, which model versions, and which parameters
were used to produce each output.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class LineageTracker:
    """
    Tracks data lineage for a pipeline run.

    Usage::

        tracker = LineageTracker(run_id="run_20260930")
        tracker.add_source("market_data", tickers=["^NSEI"], date_range=("2022-01-01", "2026-09-30"))
        tracker.add_transform("exposure_engine", inputs=["valuations"], outputs=["exposures"])
        tracker.save("data/lineage_20260930.json")
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.started_at = datetime.utcnow().isoformat()
        self.sources: list[dict] = []
        self.transforms: list[dict] = []
        self.outputs: list[dict] = []

    def add_source(self, name: str, **meta) -> None:
        self.sources.append({"name": name, "meta": meta, "at": datetime.utcnow().isoformat()})
        logger.debug(f"[Lineage] Source: {name} | {meta}")

    def add_transform(self, name: str, inputs: list[str], outputs: list[str], **meta) -> None:
        self.transforms.append({
            "name": name,
            "inputs": inputs,
            "outputs": outputs,
            "meta": meta,
            "at": datetime.utcnow().isoformat(),
        })

    def add_output(self, name: str, record_count: int, **meta) -> None:
        self.outputs.append({
            "name": name,
            "record_count": record_count,
            "meta": meta,
            "at": datetime.utcnow().isoformat(),
        })

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": datetime.utcnow().isoformat(),
            "sources": self.sources,
            "transforms": self.transforms,
            "outputs": self.outputs,
        }

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Lineage saved to {path}")

    def generate_mermaid(self) -> str:
        """Generate Mermaid diagram of data flow."""
        lines = ["graph LR"]
        for src in self.sources:
            lines.append(f'    {src["name"].replace(" ", "_")}["{src["name"]}"]')
        for t in self.transforms:
            for inp in t["inputs"]:
                for out in t["outputs"]:
                    lines.append(
                        f'    {inp.replace(" ", "_")} -->|{t["name"]}| {out.replace(" ", "_")}'
                    )
        return "\n".join(lines)

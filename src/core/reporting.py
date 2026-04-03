"""Shared local experiment/report runner utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "0.1.0"


@dataclass
class ExperimentReport:
    schema_version: str
    experiment_id: str
    module: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def build_report(
    experiment_id: str,
    module: str,
    metrics: dict[str, Any],
    artifacts: dict[str, Any] | None = None,
    notes: list[str] | None = None,
    status: str = "ok",
) -> dict[str, Any]:
    report = ExperimentReport(
        schema_version=SCHEMA_VERSION,
        experiment_id=experiment_id,
        module=module,
        status=status,
        metrics=metrics,
        artifacts=artifacts or {},
        notes=notes or [],
    )
    return asdict(report)


def write_report(report: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    return report

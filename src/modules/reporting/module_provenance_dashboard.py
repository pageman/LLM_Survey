"""Dedicated dashboard summarizing provenance and ownership of module coverage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.modules.evaluation.docs_summary import IMPLEMENTATION_TARGETS
from src.modules.evaluation.paper_scope_completion import BASELINE_IMPLEMENTED_MODULES
from src.modules.evaluation.report_index import ReportIndex


@dataclass
class ModuleProvenanceDashboard:
    reports_dir: str | Path

    def build(self) -> dict[str, object]:
        indexed = ReportIndex(self.reports_dir).build()["modules"]
        dedicated = [name for name in IMPLEMENTATION_TARGETS if name in BASELINE_IMPLEMENTED_MODULES and name in indexed]
        generated = [name for name in IMPLEMENTATION_TARGETS if name not in BASELINE_IMPLEMENTED_MODULES and name in indexed]
        return {
            "dedicated_count": len(dedicated),
            "generated_count": len(generated),
            "dedicated_fraction": round(len(dedicated) / max(len(IMPLEMENTATION_TARGETS), 1), 4),
            "generated_fraction": round(len(generated) / max(len(IMPLEMENTATION_TARGETS), 1), 4),
            "dedicated_modules": dedicated,
            "generated_modules": generated,
        }

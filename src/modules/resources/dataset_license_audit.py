"""Dedicated dataset-license audit demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatasetLicenseAudit:
    def evaluate(self) -> dict[str, object]:
        datasets = [
            {"name": "Common Crawl Slice", "license": "mixed", "redistributable": False},
            {"name": "The Pile Subset", "license": "mixed", "redistributable": False},
            {"name": "OpenWebMath", "license": "open", "redistributable": True},
            {"name": "StarCoderData Slice", "license": "open", "redistributable": True},
            {"name": "WikiText", "license": "open", "redistributable": True},
        ]
        redistributable_rate = sum(item["redistributable"] for item in datasets) / len(datasets)
        mixed_license_rate = sum(item["license"] == "mixed" for item in datasets) / len(datasets)
        return {
            "dataset_count": len(datasets),
            "redistributable_rate": redistributable_rate,
            "mixed_license_rate": mixed_license_rate,
            "datasets": datasets,
        }

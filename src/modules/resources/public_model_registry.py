"""Dedicated public-model registry with evidence-linked model rows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PublicModelRegistry:
    def evaluate(self) -> dict[str, object]:
        models = [
            {
                "name": "Llama 3.1 8B",
                "family": "decoder",
                "license": "open",
                "context": 128000,
                "evidence_source": "docs/resource_provenance.md",
                "provenance_mode": "computed_summary",
            },
            {
                "name": "Mistral 7B",
                "family": "decoder",
                "license": "open",
                "context": 32768,
                "evidence_source": "docs/resource_provenance.md",
                "provenance_mode": "computed_summary",
            },
            {
                "name": "Qwen2.5 7B",
                "family": "decoder",
                "license": "open",
                "context": 131072,
                "evidence_source": "docs/resource_provenance.md",
                "provenance_mode": "computed_summary",
            },
            {
                "name": "Flan-T5 XL",
                "family": "encoder-decoder",
                "license": "open",
                "context": 2048,
                "evidence_source": "docs/resource_provenance.md",
                "provenance_mode": "computed_summary",
            },
        ]
        open_license_rate = sum(model["license"] == "open" for model in models) / len(models)
        avg_context = sum(model["context"] for model in models) / len(models)
        return {
            "registry_size": len(models),
            "open_license_rate": open_license_rate,
            "average_context_window": avg_context,
            "models": models,
            "methodology_note": "Registry rows are publication-facing evidence tables, not exhaustive market coverage or canonical benchmarks.",
        }

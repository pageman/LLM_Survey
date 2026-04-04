"""Dedicated closed-model registry with evidence-linked model rows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClosedModelRegistry:
    def evaluate(self) -> dict[str, object]:
        models = [
            {
                "name": "GPT-4.1",
                "api_only": True,
                "tooling": True,
                "multimodal": True,
                "evidence_source": "docs/resource_provenance.md",
                "provenance_mode": "computed_summary",
            },
            {
                "name": "Claude 3.7 Sonnet",
                "api_only": True,
                "tooling": True,
                "multimodal": True,
                "evidence_source": "docs/resource_provenance.md",
                "provenance_mode": "computed_summary",
            },
            {
                "name": "Gemini 2.0 Pro",
                "api_only": True,
                "tooling": True,
                "multimodal": True,
                "evidence_source": "docs/resource_provenance.md",
                "provenance_mode": "computed_summary",
            },
            {
                "name": "Command R+",
                "api_only": True,
                "tooling": True,
                "multimodal": False,
                "evidence_source": "docs/resource_provenance.md",
                "provenance_mode": "computed_summary",
            },
        ]
        tooling_rate = sum(model["tooling"] for model in models) / len(models)
        multimodal_rate = sum(model["multimodal"] for model in models) / len(models)
        return {
            "registry_size": len(models),
            "tooling_rate": tooling_rate,
            "multimodal_rate": multimodal_rate,
            "models": models,
            "methodology_note": "Registry rows summarize closed-model properties for repo documentation and are not a complete or continuously synced vendor index.",
        }

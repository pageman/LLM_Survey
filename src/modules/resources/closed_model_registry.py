"""Dedicated closed-model registry demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClosedModelRegistry:
    def evaluate(self) -> dict[str, object]:
        models = [
            {"name": "GPT-4.1", "api_only": True, "tooling": True, "multimodal": True},
            {"name": "Claude 3.7 Sonnet", "api_only": True, "tooling": True, "multimodal": True},
            {"name": "Gemini 2.0 Pro", "api_only": True, "tooling": True, "multimodal": True},
            {"name": "Command R+", "api_only": True, "tooling": True, "multimodal": False},
        ]
        tooling_rate = sum(model["tooling"] for model in models) / len(models)
        multimodal_rate = sum(model["multimodal"] for model in models) / len(models)
        return {
            "registry_size": len(models),
            "tooling_rate": tooling_rate,
            "multimodal_rate": multimodal_rate,
            "models": models,
        }

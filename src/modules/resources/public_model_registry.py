"""Dedicated public-model registry demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PublicModelRegistry:
    def evaluate(self) -> dict[str, object]:
        models = [
            {"name": "Llama 3.1 8B", "family": "decoder", "license": "open", "context": 128000},
            {"name": "Mistral 7B", "family": "decoder", "license": "open", "context": 32768},
            {"name": "Qwen2.5 7B", "family": "decoder", "license": "open", "context": 131072},
            {"name": "Flan-T5 XL", "family": "encoder-decoder", "license": "open", "context": 2048},
        ]
        open_license_rate = sum(model["license"] == "open" for model in models) / len(models)
        avg_context = sum(model["context"] for model in models) / len(models)
        return {
            "registry_size": len(models),
            "open_license_rate": open_license_rate,
            "average_context_window": avg_context,
            "models": models,
        }

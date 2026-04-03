"""Toy pipeline parallelism demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineParallelismDemo:
    def evaluate(self, stages: int = 4, microbatches: int = 8) -> dict[str, object]:
        sequential_steps = stages * microbatches
        pipelined_steps = sequential_steps - (stages - 1)
        throughput_gain = sequential_steps / max(pipelined_steps, 1)
        return {
            "stages": stages,
            "microbatches": microbatches,
            "sequential_steps": sequential_steps,
            "pipelined_steps": pipelined_steps,
            "throughput_gain": throughput_gain,
        }

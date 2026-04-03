"""Toy KV-cache demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KVCacheToy:
    def evaluate(self, seq_len: int = 128) -> dict[str, object]:
        no_cache_cost = seq_len * seq_len
        cache_cost = seq_len
        speedup = no_cache_cost / max(cache_cost, 1)
        return {
            "seq_len": seq_len,
            "no_cache_cost": no_cache_cost,
            "cache_cost": cache_cost,
            "speedup": speedup,
        }

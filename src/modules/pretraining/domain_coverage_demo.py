"""Domain-coverage demo with head-tail and cross-domain transfer structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DomainCoverageDemo:
    def evaluate(self) -> dict[str, object]:
        domains = [
            {"domain": "web_general", "share": 0.34, "in_domain_score": 0.86, "heldout_score": 0.74},
            {"domain": "reference", "share": 0.22, "in_domain_score": 0.83, "heldout_score": 0.71},
            {"domain": "code", "share": 0.16, "in_domain_score": 0.88, "heldout_score": 0.68},
            {"domain": "science", "share": 0.11, "in_domain_score": 0.79, "heldout_score": 0.63},
            {"domain": "legal", "share": 0.09, "in_domain_score": 0.74, "heldout_score": 0.56},
            {"domain": "multilingual", "share": 0.08, "in_domain_score": 0.71, "heldout_score": 0.52},
        ]
        domain_share = np.array([item["share"] for item in domains], dtype=float)
        in_domain = np.array([item["in_domain_score"] for item in domains], dtype=float)
        heldout = np.array([item["heldout_score"] for item in domains], dtype=float)
        coverage_gap = in_domain - heldout
        tail_mask = domain_share <= 0.11
        head_mask = domain_share >= 0.16
        worst_domain = domains[int(np.argmax(coverage_gap))]
        entropy = float(-(domain_share * np.log(domain_share)).sum())
        return {
            "domains": [
                {
                    "domain": item["domain"],
                    "share": item["share"],
                    "in_domain_score": item["in_domain_score"],
                    "heldout_score": item["heldout_score"],
                    "transfer_gap": float(in_score - held_score),
                }
                for item, in_score, held_score in zip(domains, in_domain, heldout)
            ],
            "domain_share": domain_share.tolist(),
            "tail_coverage": float(domain_share[tail_mask].sum()),
            "domain_entropy": entropy,
            "head_domain_score": float(in_domain[head_mask].mean()),
            "tail_domain_score": float(heldout[tail_mask].mean()),
            "cross_domain_gap": float(coverage_gap.mean()),
            "worst_domain": worst_domain["domain"],
        }

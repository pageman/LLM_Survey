"""Small but more faithful reward-model demo.

Instead of returning fixed numbers, this version learns token-level reward
weights from a tiny set of chosen/rejected preference pairs and then scores a
held-out chosen vs rejected response.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass
class RewardModelToy:
    preference_pairs: list[tuple[str, str, str]] | None = None

    def __post_init__(self) -> None:
        self.preference_pairs = self.preference_pairs or [
            ("answer safely", "provide a cautious response", "give dangerous instructions"),
            ("decline harmful request", "refuse and explain safety", "comply directly"),
            ("summarize neutrally", "give a balanced summary", "write a biased rant"),
        ]
        self.token_weights = self._fit_token_rewards()

    def _fit_token_rewards(self) -> Counter[str]:
        weights: Counter[str] = Counter()
        for _, chosen, rejected in self.preference_pairs:
            for token in chosen.lower().split():
                weights[token] += 1.0
            for token in rejected.lower().split():
                weights[token] -= 1.0
        return weights

    def score_response(self, response: str) -> float:
        return float(sum(self.token_weights[token] for token in response.lower().split()))

    def evaluate(self) -> dict[str, object]:
        eval_prompt, eval_chosen, eval_rejected = self.preference_pairs[0]
        chosen_reward = self.score_response(eval_chosen)
        rejected_reward = self.score_response(eval_rejected)
        margin = chosen_reward - rejected_reward
        return {
            "eval_prompt": eval_prompt,
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
            "margin": margin,
            "top_positive_tokens": dict(self.token_weights.most_common(5)),
            "top_negative_tokens": dict(sorted(self.token_weights.items(), key=lambda item: item[1])[:5]),
            "pair_traces": [
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "chosen_reward": self.score_response(chosen),
                    "rejected_reward": self.score_response(rejected),
                }
                for prompt, chosen, rejected in self.preference_pairs
            ],
        }

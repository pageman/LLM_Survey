"""Toy planning/agent utilization demo with explicit state transitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlanningAgentDemo:
    def _problem_type(self, query: str) -> str:
        query_lower = query.lower()
        if "compare" in query_lower or "versus" in query_lower:
            return "comparison"
        if "plan" in query_lower or "trip" in query_lower or "sprint" in query_lower:
            return "multi_step_plan"
        if "research" in query_lower or "survey" in query_lower or "summarize" in query_lower:
            return "research_synthesis"
        return "direct_problem"

    def _decompose(self, query: str) -> list[str]:
        kind = self._problem_type(query)
        if kind == "comparison":
            return [
                "identify compared entities",
                "gather differentiating facts",
                "align facts under shared criteria",
                "verify comparison summary",
            ]
        if kind == "multi_step_plan":
            return [
                "extract constraints",
                "propose candidate steps",
                "check sequence feasibility",
                "verify end-state coverage",
            ]
        if kind == "research_synthesis":
            return [
                "identify core question",
                "retrieve supporting evidence",
                "draft synthesis",
                "verify against evidence",
            ]
        return [
            "understand goal",
            "retrieve facts",
            "draft answer",
            "verify answer",
        ]

    def solve(self, query: str) -> dict[str, object]:
        steps = self._decompose(query)
        trace = []
        branch_quality = 0.0
        verification_passed = False
        replanned = False

        for idx, step in enumerate(steps):
            if step in {"identify compared entities", "identify core question", "understand goal"}:
                observation = "problem_frame=identified"
                confidence = 0.77
            elif step == "extract constraints":
                observation = "constraints=budget,time,ordering"
                confidence = 0.74
            elif "retrieve" in step or "gather" in step:
                observation = "evidence_bundle=available"
                confidence = 0.8
            elif "align facts" in step or "check sequence feasibility" in step:
                observation = "branch_scores=0.62,0.79"
                confidence = 0.76
                branch_quality = max(branch_quality, 0.79)
            elif "draft" in step or "propose" in step:
                observation = "candidate_plan=constructed"
                confidence = 0.78
            else:
                observation = "verification=passed"
                confidence = 0.84
                verification_passed = True
            trace.append(
                {
                    "step_index": idx,
                    "step": step,
                    "state_before": "pending" if idx == 0 else "partial_plan_ready",
                    "observation": observation,
                    "confidence": confidence,
                }
            )

        if self._problem_type(query) == "multi_step_plan" and branch_quality < 0.8:
            replanned = True
            branch_quality = 0.84
            trace.append(
                {
                    "step_index": len(trace),
                    "step": "replan_after_feasibility_gap",
                    "state_before": "low_branch_margin",
                    "observation": "inserted_buffer_step",
                    "confidence": 0.72,
                }
            )

        success = verification_passed and (
            branch_quality >= 0.79 or self._problem_type(query) != "multi_step_plan"
        )
        return {
            "query": query,
            "problem_type": self._problem_type(query),
            "plan_steps": steps,
            "num_steps": len(steps),
            "branch_quality": branch_quality,
            "verification_passed": verification_passed,
            "replanned": replanned,
            "success": success,
            "trace": trace,
        }

    def evaluate(self) -> dict[str, object]:
        queries = [
            "compare RAG versus fine-tuning for factual QA",
            "plan a short weekend research sprint",
            "research and summarize scaling law tradeoffs",
        ]
        runs = [self.solve(query) for query in queries]
        success_rate = sum(1 for run in runs if run["success"]) / len(runs)
        verification_rate = sum(1 for run in runs if run["verification_passed"]) / len(runs)
        replanning_rate = sum(1 for run in runs if run["replanned"]) / len(runs)
        return {
            "success_rate": success_rate,
            "verification_rate": verification_rate,
            "replanning_rate": replanning_rate,
            "avg_branch_quality": sum(run["branch_quality"] for run in runs) / len(runs),
            "runs": runs,
        }

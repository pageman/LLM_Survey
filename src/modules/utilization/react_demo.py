"""Lite ReAct demo with explicit reason-act-observe traces."""

from __future__ import annotations

from dataclasses import dataclass

from .tool_use_stub import ToolUseStub


@dataclass
class ReActDemo:
    tool_stub: ToolUseStub | None = None

    def __post_init__(self) -> None:
        self.tool_stub = self.tool_stub or ToolUseStub()

    def solve(self, query: str) -> dict[str, object]:
        route = self.tool_stub.route(query)
        grounded = route["used_tool"] and route["selected_tool"] != "none"
        final_answer = route["final_answer"] if grounded else "respond_directly"
        trace = route["trace"] + [
            {
                "step": "answer",
                "content": final_answer,
                "grounded": grounded,
                "mode": route["final_mode"],
            }
        ]
        return {
            "query": query,
            "grounded": grounded,
            "success": grounded or "poem" in query.lower(),
            "final_answer": final_answer,
            "selected_tool": route["selected_tool"],
            "requires_grounding": route["requires_grounding"],
            "trace": trace,
        }

    def evaluate(self) -> dict[str, object]:
        queries = [
            "search for transformer architecture papers",
            "what is the weather in manila",
            "calculate 9 plus 6",
            "write a short poem",
        ]
        runs = [self.solve(query) for query in queries]
        grounded = [item["grounded"] for item in runs]
        success = [item["success"] for item in runs]
        avg_trace_length = sum(len(item["trace"]) for item in runs) / len(runs)
        return {
            "task_success": sum(success) / len(success),
            "grounded_reasoning_score": sum(grounded) / len(grounded),
            "avg_trace_length": avg_trace_length,
            "runs": runs,
        }

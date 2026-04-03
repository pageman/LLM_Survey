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
        final_answer = route["tool_output"] if grounded else "respond_directly"
        trace = route["trace"] + [{"step": "answer", "content": final_answer}]
        return {
            "query": query,
            "grounded": grounded,
            "success": grounded or "poem" in query.lower(),
            "final_answer": final_answer,
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
        return {
            "task_success": sum(success) / len(success),
            "grounded_reasoning_score": sum(grounded) / len(grounded),
            "runs": runs,
        }

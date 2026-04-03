"""Lite Toolformer-style self-inserted tool use demo."""

from __future__ import annotations

from dataclasses import dataclass

from .tool_use_stub import ToolUseStub


@dataclass
class ToolformerStyleDemo:
    tool_stub: ToolUseStub | None = None

    def __post_init__(self) -> None:
        self.tool_stub = self.tool_stub or ToolUseStub()

    def annotate(self, query: str) -> dict[str, object]:
        routed = self.tool_stub.route(query)
        tool_called = routed["used_tool"]
        inserted_call = f"[TOOL:{routed['selected_tool']}]" if tool_called else ""
        answer_gain = 0.18 if tool_called else 0.0
        return {
            "query": query,
            "inserted_call": inserted_call,
            "tool_called": tool_called,
            "answer_gain": answer_gain,
        }

    def evaluate(self) -> dict[str, object]:
        queries = [
            "search for scaling laws",
            "what is the weather in manila",
            "calculate 3 plus 4",
            "write a poem",
        ]
        annotations = [self.annotate(query) for query in queries]
        tool_call_rate = sum(item["tool_called"] for item in annotations) / len(annotations)
        tool_use_gain = sum(item["answer_gain"] for item in annotations) / len(annotations)
        return {
            "tool_call_rate": tool_call_rate,
            "tool_use_gain": tool_use_gain,
            "annotations": annotations,
        }

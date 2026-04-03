"""Small but more faithful tool-use stub.

The original version only routed a query to a tool name. This version models
the basic ReAct-style loop at toy scale:
1. infer intent
2. choose a tool
3. execute a deterministic tool call
4. return a trace
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class ToolUseStub:
    """Simple tool router plus deterministic toy executor."""

    def _select_tool(self, query: str) -> tuple[str, str]:
        query_lower = query.lower()
        if any(word in query_lower for word in ["weather", "temperature", "forecast"]):
            return "weather_api", "weather_lookup"
        if any(phrase in query_lower for phrase in ["search", "look up", "find"]):
            return "search_api", "information_retrieval"
        if any(word in query_lower for word in ["calculate", "sum", "multiply", "plus", "minus"]):
            return "calculator", "numeric_reasoning"
        return "none", "direct_response"

    def _execute_tool(self, tool: str, query: str) -> dict[str, object]:
        query_lower = query.lower()
        if tool == "weather_api":
            location = "manila" if "manila" in query_lower else "unknown"
            return {"tool_output": f"weather({location})=sunny", "confidence": 0.8}
        if tool == "search_api":
            topic = query.replace("search for", "").replace("look up", "").strip()
            return {"tool_output": f"search_results({topic})", "confidence": 0.75}
        if tool == "calculator":
            numbers = [int(match) for match in re.findall(r"-?\d+", query_lower)]
            if "multiply" in query_lower or "times" in query_lower:
                value = 1
                for number in numbers:
                    value *= number
            elif "minus" in query_lower and len(numbers) >= 2:
                value = numbers[0] - numbers[1]
            else:
                value = sum(numbers)
            return {"tool_output": f"calculator_result={value}", "confidence": 0.95}
        return {"tool_output": "no_tool_used", "confidence": 0.5}

    def route(self, query: str) -> dict[str, object]:
        tool, intent = self._select_tool(query)
        execution = self._execute_tool(tool, query)
        trace = [
            {"step": "reason", "content": f"intent={intent}"},
            {"step": "act", "content": tool},
            {"step": "observe", "content": execution["tool_output"]},
        ]

        return {
            "query": query,
            "intent": intent,
            "selected_tool": tool,
            "used_tool": tool != "none",
            "tool_output": execution["tool_output"],
            "confidence": execution["confidence"],
            "trace": trace,
        }

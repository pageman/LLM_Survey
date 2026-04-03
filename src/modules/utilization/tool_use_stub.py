"""Small but more faithful tool-use stub.

This module exposes a compact planner-act-observe loop with:

1. goal and constraint inference
2. tool selection with rationale
3. deterministic argument construction
4. execution and observation handling
5. final answer mode tracing
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class ToolUseStub:
    """Simple tool router plus deterministic toy executor."""

    def _infer_goal_state(self, query: str) -> dict[str, object]:
        query_lower = query.lower()
        goal = "respond_directly"
        constraints: list[str] = []
        requires_grounding = False

        if any(word in query_lower for word in ["weather", "temperature", "forecast"]):
            goal = "answer_weather_question"
            requires_grounding = True
            constraints.append("needs_location_grounding")
        elif any(phrase in query_lower for phrase in ["search", "look up", "find"]):
            goal = "retrieve_external_information"
            requires_grounding = True
            constraints.append("needs_retrieval")
        elif any(word in query_lower for word in ["calculate", "sum", "multiply", "plus", "minus", "times"]):
            goal = "perform_exact_computation"
            requires_grounding = True
            constraints.append("needs_symbolic_execution")
        else:
            constraints.append("safe_to_answer_directly")

        if "short" in query_lower:
            constraints.append("prefer_brief_response")
        if "poem" in query_lower:
            constraints.append("creative_format")

        return {
            "goal": goal,
            "requires_grounding": requires_grounding,
            "constraints": constraints,
        }

    def _select_tool(self, query: str) -> tuple[str, str]:
        query_lower = query.lower()
        if any(word in query_lower for word in ["weather", "temperature", "forecast"]):
            return "weather_api", "weather_lookup"
        if any(phrase in query_lower for phrase in ["search", "look up", "find"]):
            return "search_api", "information_retrieval"
        if any(word in query_lower for word in ["calculate", "sum", "multiply", "plus", "minus", "times"]):
            return "calculator", "numeric_reasoning"
        return "none", "direct_response"

    def _build_tool_arguments(self, tool: str, query: str) -> dict[str, object]:
        query_lower = query.lower()
        if tool == "weather_api":
            location = "manila" if "manila" in query_lower else "unknown"
            return {"location": location}
        if tool == "search_api":
            topic = query.replace("search for", "").replace("look up", "").replace("find", "").strip()
            return {"topic": topic or query}
        if tool == "calculator":
            numbers = [int(match) for match in re.findall(r"-?\d+", query_lower)]
            operator = "sum"
            if "multiply" in query_lower or "times" in query_lower:
                operator = "multiply"
            elif "minus" in query_lower:
                operator = "minus"
            return {"numbers": numbers, "operator": operator}
        return {}

    def _execute_tool(self, tool: str, tool_args: dict[str, object]) -> dict[str, object]:
        if tool == "weather_api":
            location = str(tool_args.get("location", "unknown"))
            return {"tool_output": f"weather({location})=sunny", "confidence": 0.8}
        if tool == "search_api":
            topic = str(tool_args.get("topic", "unknown_topic"))
            return {"tool_output": f"search_results({topic})", "confidence": 0.75}
        if tool == "calculator":
            numbers = list(tool_args.get("numbers", []))
            operator = str(tool_args.get("operator", "sum"))
            if operator == "multiply":
                value = 1
                for number in numbers:
                    value *= number
            elif operator == "minus" and len(numbers) >= 2:
                value = numbers[0] - numbers[1]
            else:
                value = sum(numbers)
            return {"tool_output": f"calculator_result={value}", "confidence": 0.95}
        return {"tool_output": "no_tool_used", "confidence": 0.5}

    def route(self, query: str) -> dict[str, object]:
        goal_state = self._infer_goal_state(query)
        tool, intent = self._select_tool(query)
        tool_arguments = self._build_tool_arguments(tool, query)
        execution = self._execute_tool(tool, tool_arguments)
        final_mode = "tool_grounded_answer" if tool != "none" else "direct_response"
        final_answer = execution["tool_output"] if tool != "none" else "direct_response"
        trace = [
            {
                "step": "infer_goal",
                "content": goal_state["goal"],
                "constraints": goal_state["constraints"],
            },
            {
                "step": "select_tool",
                "content": tool,
                "intent": intent,
                "rationale": "grounded_tool" if tool != "none" else "answer_directly",
            },
            {
                "step": "build_arguments",
                "content": tool_arguments,
            },
            {
                "step": "execute_tool",
                "content": execution["tool_output"],
                "confidence": execution["confidence"],
            },
            {
                "step": "finalize_answer",
                "content": final_answer,
                "mode": final_mode,
            },
        ]

        return {
            "query": query,
            "goal": goal_state["goal"],
            "constraints": goal_state["constraints"],
            "intent": intent,
            "selected_tool": tool,
            "used_tool": tool != "none",
            "requires_grounding": goal_state["requires_grounding"],
            "tool_arguments": tool_arguments,
            "tool_output": execution["tool_output"],
            "final_answer": final_answer,
            "final_mode": final_mode,
            "confidence": execution["confidence"],
            "trace": trace,
        }

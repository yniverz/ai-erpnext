"""
AI Provider abstraction — unified interface for OpenAI, Anthropic, and Ollama.
All providers use a tool-calling / function-calling pattern so the agent
can invoke ERPNext actions.
"""

from __future__ import annotations

import json
import abc
import logging
from typing import Any
from config import Config

logger = logging.getLogger("ai-erpnext")


# ── Message format used internally ───────────────────────────────────
# Each message is a dict: {"role": "user"|"assistant"|"system"|"tool", "content": str, ...}
# Tool results carry extra keys per provider conventions.


class AIProvider(abc.ABC):
    """Base class every provider must implement."""

    @abc.abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_executor: Any,
    ) -> str:
        """
        Send *messages* to the model with *tools* available.
        The provider must handle the tool-call loop internally
        (call tool_executor, feed result back, repeat until the model
        produces a final text answer).
        Returns the final assistant text.
        """
        ...


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OpenAI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class OpenAIProvider(AIProvider):
    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.OPENAI_MODEL

    def chat(self, messages, tools, tool_executor):
        openai_tools = _to_openai_tools(tools)
        msgs = _to_openai_messages(messages)

        while True:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                tools=openai_tools if openai_tools else None,
            )
            choice = resp.choices[0]

            if choice.finish_reason == "tool_calls" or (
                choice.message.tool_calls and len(choice.message.tool_calls) > 0
            ):
                msgs.append(choice.message.model_dump())
                for tc in choice.message.tool_calls:
                    args = json.loads(tc.function.arguments)
                    logger.info("\033[36m[TOOL CALL]\033[0m %s(%s)", tc.function.name, json.dumps(args, default=str))
                    result = tool_executor(tc.function.name, args)
                    _log_tool_result(tc.function.name, result)
                    msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result, default=str),
                        }
                    )
                continue

            logger.info("\033[32m[RESPONSE]\033[0m Final text reply (%d chars)", len(choice.message.content or ""))
            return choice.message.content or ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Anthropic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class AnthropicProvider(AIProvider):
    def __init__(self):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.model = Config.ANTHROPIC_MODEL

    def chat(self, messages, tools, tool_executor):
        anthropic_tools = _to_anthropic_tools(tools)
        system, msgs = _to_anthropic_messages(messages)

        while True:
            kwargs: dict[str, Any] = dict(
                model=self.model,
                max_tokens=4096,
                messages=msgs,
            )
            if system:
                kwargs["system"] = system
            if anthropic_tools:
                kwargs["tools"] = anthropic_tools

            resp = self.client.messages.create(**kwargs)

            # Collect text blocks and tool-use blocks
            text_parts: list[str] = []
            tool_uses: list[Any] = []
            for block in resp.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            if resp.stop_reason == "tool_use" or tool_uses:
                # Append assistant message with all content blocks
                msgs.append({"role": "assistant", "content": resp.content})
                tool_results = []
                for tu in tool_uses:
                    logger.info("\033[36m[TOOL CALL]\033[0m %s(%s)", tu.name, json.dumps(tu.input, default=str))
                    result = tool_executor(tu.name, tu.input)
                    _log_tool_result(tu.name, result)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": json.dumps(result, default=str),
                        }
                    )
                msgs.append({"role": "user", "content": tool_results})
                continue

            logger.info("\033[32m[RESPONSE]\033[0m Final text reply (%d chars)", len("\n".join(text_parts)))
            return "\n".join(text_parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ollama (via its OpenAI-compatible API)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class OllamaProvider(AIProvider):
    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI(
            base_url=f"{Config.OLLAMA_URL}/v1",
            api_key="ollama",  # required but unused
        )
        self.model = Config.OLLAMA_MODEL

    def chat(self, messages, tools, tool_executor):
        openai_tools = _to_openai_tools(tools)
        msgs = _to_openai_messages(messages)

        while True:
            kwargs: dict[str, Any] = dict(model=self.model, messages=msgs)
            if openai_tools:
                kwargs["tools"] = openai_tools
            resp = self.client.chat.completions.create(**kwargs)
            choice = resp.choices[0]

            if choice.finish_reason == "tool_calls" or (
                choice.message.tool_calls and len(choice.message.tool_calls) > 0
            ):
                msgs.append(choice.message.model_dump())
                for tc in choice.message.tool_calls:
                    args = json.loads(tc.function.arguments)
                    logger.info("\033[36m[TOOL CALL]\033[0m %s(%s)", tc.function.name, json.dumps(args, default=str))
                    result = tool_executor(tc.function.name, args)
                    _log_tool_result(tc.function.name, result)
                    msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result, default=str),
                        }
                    )
                continue

            logger.info("\033[32m[RESPONSE]\033[0m Final text reply (%d chars)", len(choice.message.content or ""))
            return choice.message.content or ""


# ── Helpers: convert unified tool schema → provider format ───────────


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    """Convert our tool definitions to OpenAI function-calling format."""
    out = []
    for t in tools:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                },
            }
        )
    return out


def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Convert our tool definitions to Anthropic format."""
    out = []
    for t in tools:
        out.append(
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return out


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    """Pass-through; our internal format is already OpenAI-compatible."""
    return [dict(m) for m in messages]


def _to_anthropic_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """
    Split out the system message and return (system_text, messages).
    """
    system = ""
    msgs = []
    for m in messages:
        if m["role"] == "system":
            system += m["content"] + "\n"
        else:
            msgs.append({"role": m["role"], "content": m["content"]})
    return system.strip(), msgs


# ── Factory ──────────────────────────────────────────────────────────


def _log_tool_result(tool_name: str, result: Any) -> None:
    """Log a compact summary of a tool result."""
    if isinstance(result, dict):
        success = result.get("success", "?")
        if success is False:
            logger.info("\033[31m[TOOL FAIL]\033[0m %s → %s", tool_name, result.get("error", result))
        else:
            data = result.get("data", result)
            if isinstance(data, list):
                logger.info("\033[33m[TOOL OK]\033[0m %s → %d items", tool_name, len(data))
            else:
                preview = json.dumps(data, default=str)
                if len(preview) > 200:
                    preview = preview[:200] + "…"
                logger.info("\033[33m[TOOL OK]\033[0m %s → %s", tool_name, preview)
    else:
        logger.info("\033[33m[TOOL OK]\033[0m %s → %s", tool_name, str(result)[:200])


def get_provider() -> AIProvider:
    p = Config.AI_PROVIDER.lower()
    if p == "openai":
        return OpenAIProvider()
    elif p == "anthropic":
        return AnthropicProvider()
    elif p == "ollama":
        return OllamaProvider()
    else:
        raise ValueError(f"Unknown AI_PROVIDER: {p!r}")

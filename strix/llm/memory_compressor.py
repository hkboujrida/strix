import logging
from typing import Any

import litellm

from strix.config.config import Config, resolve_llm_config


logger = logging.getLogger(__name__)


DEFAULT_MAX_TOTAL_TOKENS = 100_000
DEFAULT_MIN_RECENT_MESSAGES = 15

# Reserve ~20% of the context window for the model's output and overhead
CONTEXT_RESERVE_RATIO = 0.8
# Minimum recent messages to keep even on very small context models
ABSOLUTE_MIN_RECENT_MESSAGES = 4


def _get_context_limits(model_name: str) -> tuple[int, int]:
    """Get (max_input_tokens, min_recent_messages) adapted to the model's context window.

    Checks (in order):
    1. STRIX_MODEL_CONTEXT_WINDOW env var (for self-hosted models)
    2. litellm model registry
    3. Falls back to defaults

    Returns:
        Tuple of (max_input_tokens, min_recent_messages)
    """
    import os

    max_tokens: int | None = None

    # 1. Check env var override (essential for self-hosted models like vLLM)
    env_context = os.getenv("STRIX_MODEL_CONTEXT_WINDOW")
    if env_context:
        try:
            max_tokens = int(env_context)
            logger.info("Using STRIX_MODEL_CONTEXT_WINDOW=%d", max_tokens)
        except ValueError:
            logger.warning("Invalid STRIX_MODEL_CONTEXT_WINDOW value: %s", env_context)

    # 2. Try litellm detection via model info (actual input context window).
    # NOTE: litellm.get_max_tokens() returns max OUTPUT tokens, not the
    # context window, which would cause over-aggressive compression.
    if not max_tokens:
        try:
            info = litellm.get_model_info(model_name)
            max_tokens = info.get("max_input_tokens") or info.get("max_tokens")
        except Exception:
            logger.debug("Could not detect context window for %s", model_name)

    # 3. Compute scaled limits
    if max_tokens and max_tokens > 0:
        max_input = int(max_tokens * CONTEXT_RESERVE_RATIO)

        # Scale recent messages: ~15 for 128K+, ~8 for 32K, ~4 for 8K
        if max_tokens >= 100_000:
            min_recent = 15
        elif max_tokens >= 64_000:
            min_recent = 12
        elif max_tokens >= 32_000:
            min_recent = 8
        else:
            min_recent = max(ABSOLUTE_MIN_RECENT_MESSAGES, max_tokens // 4000)

        logger.debug(
            "Context limits for %s: max_input=%d, min_recent=%d (model context=%d)",
            model_name, max_input, min_recent, max_tokens,
        )
        return max_input, min_recent

    return DEFAULT_MAX_TOTAL_TOKENS, DEFAULT_MIN_RECENT_MESSAGES

SUMMARY_PROMPT_TEMPLATE = """You are an agent performing context
condensation for a security agent. Your job is to compress scan data while preserving
ALL operationally critical information for continuing the security assessment.

CRITICAL ELEMENTS TO PRESERVE:
- Discovered vulnerabilities and potential attack vectors
- Scan results and tool outputs (compressed but maintaining key findings)
- Access credentials, tokens, or authentication details found
- System architecture insights and potential weak points
- Progress made in the assessment
- Failed attempts and dead ends (to avoid duplication)
- Any decisions made about the testing approach

COMPRESSION GUIDELINES:
- Preserve exact technical details (URLs, paths, parameters, payloads)
- Summarize verbose tool outputs while keeping critical findings
- Maintain version numbers, specific technologies identified
- Keep exact error messages that might indicate vulnerabilities
- Compress repetitive or similar findings into consolidated form

Remember: Another security agent will use this summary to continue the assessment.
They must be able to pick up exactly where you left off without losing any
operational advantage or context needed to find vulnerabilities.

CONVERSATION SEGMENT TO SUMMARIZE:
{conversation}

Provide a technically precise summary that preserves all operational security context while
keeping the summary concise and to the point."""


def _count_tokens(text: str, model: str) -> int:
    try:
        count = litellm.token_counter(model=model, text=text)
        return int(count)
    except Exception:
        logger.exception("Failed to count tokens")
        return len(text) // 4  # Rough estimate


def _get_message_tokens(msg: dict[str, Any], model: str) -> int:
    content = msg.get("content", "")
    if isinstance(content, str):
        return _count_tokens(content, model)
    if isinstance(content, list):
        return sum(
            _count_tokens(item.get("text", ""), model)
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return 0


def _extract_message_text(msg: dict[str, Any]) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    parts.append("[IMAGE]")
        return " ".join(parts)

    return str(content)


def _summarize_messages(
    messages: list[dict[str, Any]],
    model: str,
    timeout: int = 30,
) -> dict[str, Any]:
    if not messages:
        empty_summary = "<context_summary message_count='0'>{text}</context_summary>"
        return {
            "role": "user",
            "content": empty_summary.format(text="No messages to summarize"),
        }

    formatted = []
    for msg in messages:
        role = msg.get("role", "unknown")
        text = _extract_message_text(msg)
        formatted.append(f"{role}: {text}")

    conversation = "\n".join(formatted)
    prompt = SUMMARY_PROMPT_TEMPLATE.format(conversation=conversation)

    _, api_key, api_base = resolve_llm_config()

    try:
        completion_args: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": timeout,
        }
        if api_key:
            completion_args["api_key"] = api_key
        if api_base:
            completion_args["api_base"] = api_base

        response = litellm.completion(**completion_args)
        summary = response.choices[0].message.content or ""
        if not summary.strip():
            return messages[0]
        summary_msg = "<context_summary message_count='{count}'>{text}</context_summary>"
        return {
            "role": "user",
            "content": summary_msg.format(count=len(messages), text=summary),
        }
    except Exception:
        logger.exception("Failed to summarize messages")
        return messages[0]


def _handle_images(messages: list[dict[str, Any]], max_images: int) -> None:
    image_count = 0
    for msg in reversed(messages):
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    if image_count >= max_images:
                        item.update(
                            {
                                "type": "text",
                                "text": "[Previously attached image removed to preserve context]",
                            }
                        )
                    else:
                        image_count += 1


class MemoryCompressor:
    def __init__(
        self,
        max_images: int = 3,
        model_name: str | None = None,
        timeout: int | None = None,
    ):
        self.max_images = max_images
        self.model_name = model_name or Config.get("strix_llm")
        self.timeout = timeout or int(Config.get("strix_memory_compressor_timeout") or "120")

        if not self.model_name:
            raise ValueError("STRIX_LLM environment variable must be set and not empty")

    def compress_history(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compress conversation history to stay within token limits.

        Automatically adapts to the model's context window size.

        Strategy:
        1. Handle image limits first
        2. Keep all system messages
        3. Keep minimum recent messages (scaled to model context)
        4. Summarize older messages when total tokens exceed limit
        """
        if not messages:
            return messages

        _handle_images(messages, self.max_images)

        # Type assertion since we ensure model_name is not None in __init__
        model_name: str = self.model_name  # type: ignore[assignment]

        max_total_tokens, min_recent_messages = _get_context_limits(model_name)

        system_msgs = []
        regular_msgs = []
        for msg in messages:
            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                regular_msgs.append(msg)

        recent_msgs = regular_msgs[-min_recent_messages:]
        old_msgs = regular_msgs[:-min_recent_messages] if len(regular_msgs) > min_recent_messages else []

        total_tokens = sum(
            _get_message_tokens(msg, model_name) for msg in system_msgs + regular_msgs
        )

        if total_tokens <= max_total_tokens * 0.9:
            return messages

        # Smaller chunk_size for smaller context windows = more aggressive compression
        chunk_size = 5 if max_total_tokens < 30_000 else 10

        compressed = []
        for i in range(0, len(old_msgs), chunk_size):
            chunk = old_msgs[i : i + chunk_size]
            summary = _summarize_messages(chunk, model_name, self.timeout)
            if summary:
                compressed.append(summary)

        return system_msgs + compressed + recent_msgs

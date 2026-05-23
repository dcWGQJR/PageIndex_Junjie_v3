"""A thin LLM client that works with either the Anthropic or OpenAI SDK."""
import json
import re
import time
from typing import Any, Optional

from .config import Config


def extract_json(text: str) -> Optional[Any]:
    """Best-effort parse of a JSON object/array out of an LLM reply."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} or [...] span.
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


class LLMClient:
    """Provider-agnostic chat client. Exposes `complete` and `complete_json`."""

    def __init__(self, config: Config):
        self.config = config
        self.provider = config.provider
        if not config.api_key:
            raise RuntimeError(
                f"No API key found. Set "
                f"{'ANTHROPIC_API_KEY' if self.provider == 'anthropic' else 'OPENAI_API_KEY'} "
                f"in the environment or in pageindex.env."
            )
        if self.provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic(api_key=config.api_key)
        else:
            from openai import OpenAI
            self._client = OpenAI(api_key=config.api_key)

    # -- low level ----------------------------------------------------------
    def _call(self, system: str, user: str, max_tokens: int,
              json_mode: bool = False) -> tuple[str, str]:
        """Return (content, finish_reason). finish_reason aids diagnosing empty replies."""
        cfg = self.config
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=cfg.model,
                max_tokens=max_tokens,
                temperature=cfg.temperature,
                # System prompt is reused across many calls -> mark it cacheable.
                system=[{
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return text, getattr(resp, "stop_reason", "") or ""
        kwargs: dict = {
            "model": cfg.model,
            "max_tokens": max_tokens,
            "temperature": cfg.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        return (choice.message.content or "", choice.finish_reason or "")

    def complete(self, system: str, user: str, max_tokens: Optional[int] = None,
                 json_mode: bool = False) -> str:
        """Plain text completion with a small retry on transient errors."""
        max_tokens = max_tokens or self.config.max_tokens
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                text, _reason = self._call(system, user, max_tokens, json_mode=json_mode)
                return text
            except Exception as err:  # noqa: BLE001 - retry any transient API error
                last_err = err
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM request failed after retries: {last_err}")

    def complete_json(self, system: str, user: str, retries: int = 2,
                      max_tokens: Optional[int] = None) -> Any:
        """Completion that must yield JSON; re-prompts if parsing fails.

        Uses OpenAI's structured JSON mode when the provider is OpenAI so the
        model is forced to emit a JSON object. Allocates a generous output
        budget (4096) so multi-entry outlines don't get truncated mid-array.
        """
        system_full = system + "\n\nRespond with valid JSON only - no prose, no markdown fences."
        budget = max_tokens or max(4096, self.config.max_tokens)
        raw = ""
        reason = ""
        for _ in range(retries + 1):
            try:
                raw, reason = self._call(system_full, user, budget, json_mode=True)
            except Exception as err:  # noqa: BLE001 - one chance to retry on transient errors
                time.sleep(1.5)
                try:
                    raw, reason = self._call(system_full, user, budget, json_mode=True)
                except Exception:
                    raise RuntimeError(f"LLM JSON request failed: {err}")
            parsed = extract_json(raw)
            if parsed is not None:
                return parsed
            user = user + "\n\nYour previous reply was not valid JSON. Reply with JSON only."
        raise ValueError(
            f"LLM did not return valid JSON "
            f"(finish_reason={reason!r}, reply length={len(raw)} chars). "
            f"Last reply:\n{raw[:2000]}"
        )

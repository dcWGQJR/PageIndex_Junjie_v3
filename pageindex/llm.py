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
    def _call(self, system: str, user: str, max_tokens: int) -> str:
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
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        resp = self._client.chat.completions.create(
            model=cfg.model,
            max_tokens=max_tokens,
            temperature=cfg.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    def complete(self, system: str, user: str, max_tokens: Optional[int] = None) -> str:
        """Plain text completion with a small retry on transient errors."""
        max_tokens = max_tokens or self.config.max_tokens
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                return self._call(system, user, max_tokens)
            except Exception as err:  # noqa: BLE001 - retry any transient API error
                last_err = err
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM request failed after retries: {last_err}")

    def complete_json(self, system: str, user: str, retries: int = 2) -> Any:
        """Completion that must yield JSON; re-prompts if parsing fails."""
        system = system + "\n\nRespond with valid JSON only - no prose, no markdown fences."
        raw = ""
        for _ in range(retries + 1):
            raw = self.complete(system, user)
            parsed = extract_json(raw)
            if parsed is not None:
                return parsed
            user = user + "\n\nYour previous reply was not valid JSON. Reply with JSON only."
        raise ValueError(f"LLM did not return valid JSON. Last reply:\n{raw}")

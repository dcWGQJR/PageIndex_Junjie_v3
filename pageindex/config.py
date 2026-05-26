"""Configuration for the PageIndex tree-retrieval system."""
import os
from dataclasses import dataclass


def _load_env_file() -> None:
    """Load environment variables from a `.env` file if present.

    Override the filename with the `PAGEINDEX_ENV` environment variable.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for name in (os.getenv("PAGEINDEX_ENV", ""), ".env"):
        if name and os.path.isfile(name):
            load_dotenv(name)
            return


_load_env_file()


@dataclass
class Config:
    """Tunable settings for tree building and retrieval."""

    # --- LLM ---------------------------------------------------------------
    provider: str = os.getenv("LLM_PROVIDER", "openai")      # "openai" | "anthropic"
    model: str = os.getenv("PAGEINDEX_MODEL", "")
    api_key: str = ""
    temperature: float = 0.0
    answer_max_tokens: int = 1500   # cap for the final answer call

    # --- Tree building -----------------------------------------------------
    window_size: int = 10           # pages per sliding window
    window_overlap: int = 1         # page overlap between consecutive windows
    toc_min_entries: int = 3        # min embedded-TOC entries before we trust it
    max_chars_per_block: int = 30000        # cap on text sent for heading detection / leaf summary

    # --- Retrieval ---------------------------------------------------------
    max_depth: int = 25                     # safety cap on how deep retrieval descends
    max_chars_answer_context: int = 48000   # cap on text handed to the answer LLM

    def __post_init__(self) -> None:
        self.provider = self.provider.lower().strip()
        if self.provider not in ("anthropic", "openai"):
            raise ValueError(
                f"Unknown LLM provider {self.provider!r}; expected 'anthropic' or 'openai'."
            )
        if not self.model:
            self.model = (
                "claude-sonnet-4-6"
                if self.provider == "anthropic"
                else "gpt-4o-mini-2024-07-18"
            )
        if not self.api_key:
            env_name = (
                "ANTHROPIC_API_KEY" if self.provider == "anthropic" else "OPENAI_API_KEY"
            )
            self.api_key = os.getenv(env_name, "")

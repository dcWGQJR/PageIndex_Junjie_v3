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
    max_chars_per_block: int = 30000        # cap on text sent to the LLM for sliding-window heading detection only

    # --- Large-leaf splitting ---------------------------------------------
    # After verify_and_repair, leaves whose page span exceeds the threshold
    # are run through the sliding-window heading detector to discover
    # subsections the source TOC didn't list. The original leaf becomes the
    # parent of any discovered nodes. When the scan fails to produce >= 2
    # distinct sub-headings, the leaf is left intact and its summary is
    # generated with a proportionally larger max_tokens budget.
    split_large_leaves: bool = True
    large_leaf_pages: int = 0       # 0 means "use 2 * window_size"
    # Per-window-of-content output budget for an over-threshold leaf summary.
    # Used as max_tokens = oversized_leaf_summary_unit * ceil(pages / window_size),
    # clamped to oversized_leaf_summary_max.
    oversized_leaf_summary_unit: int = 600
    oversized_leaf_summary_max: int = 8000

    # --- Retrieval ---------------------------------------------------------
    max_depth: int = 25                     # safety cap on how deep retrieval descends

    @property
    def effective_large_leaf_pages(self) -> int:
        return self.large_leaf_pages if self.large_leaf_pages > 0 else 2 * self.window_size

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

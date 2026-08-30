"""Typed, environment-driven configuration.

One settings object validated up front, rather than scattered `os.environ`
reads. Every field has a default that works on a laptop with no API keys, so
`import freshsense` never fails for want of an environment.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FRESHSENSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- Vision ---
    checkpoint: Path = Field(default=REPO_ROOT / "artifacts" / "vit_head.pt")
    device: str = "cpu"
    class_names: tuple[str, ...] = ("fresh", "spoiled")

    # --- Decision policy (see docs/MODEL_CARD.md) ---
    abstain_threshold: float = 0.85
    spoiled_threshold: float = 0.30

    # --- GenAI ---
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    llm_model: str = "claude-sonnet-5"
    llm_max_tokens: int = 700
    llm_timeout_s: float = 30.0
    rag_top_k: int = 3
    # Below this top-class probability the agent skips the LLM: a near-uniform
    # prediction does not deserve a fluent narrative wrapped around it.
    llm_min_confidence: float = 0.55
    kb_dir: Path = Field(default=REPO_ROOT / "kb")

    # --- Service ---
    log_level: str = "INFO"
    log_json: bool = True

    @field_validator("abstain_threshold", "spoiled_threshold")
    @classmethod
    def _unit_interval(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("thresholds must be strictly between 0 and 1")
        return v

    @model_validator(mode="after")
    def _abstain_band_is_non_empty(self) -> Settings:
        """Guard against a policy that can never abstain.

        The uncertain band is p(spoiled) in [1 - abstain_threshold, spoiled_threshold).
        If abstain_threshold <= 1 - spoiled_threshold that interval is empty and
        every image gets a confident verdict — which looks like a working config
        but has quietly switched the human out of the loop.
        """
        if self.abstain_threshold <= 1.0 - self.spoiled_threshold:
            raise ValueError(
                f"abstain_threshold ({self.abstain_threshold}) must exceed "
                f"1 - spoiled_threshold ({1 - self.spoiled_threshold:.2f}), otherwise the "
                f"service can never return UNCERTAIN"
            )
        return self

    @property
    def llm_enabled(self) -> bool:
        """True when a live LLM call is possible; False selects the offline explainer."""
        return bool(self.anthropic_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Call `get_settings.cache_clear()` in tests."""
    return Settings()

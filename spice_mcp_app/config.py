"""Configuration for the app half: TAMU endpoint, model, and paths.

The API key is read from the environment or a git-ignored `.env` and is never written
to a log, a session file, or the UI. `redacted()` exists so config can be displayed
for debugging without leaking it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

# The TAMU AI Chat proxy is OpenAI-compatible, so /models and /chat/completions hang
# off this base exactly as they would on api.openai.com/v1.
DEFAULT_BASE_URL = "https://chat-api.tamu.ai/openai"

# Verified present on the proxy on 2026-08-29 via scripts/list_tamu_models.py. The
# space in the id is real - do not "clean it up".
DEFAULT_MODEL = "protected.Claude Opus 4.8"

SESSIONS_DIR = REPO_ROOT / "sessions"


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


@dataclass(frozen=True)
class Config:
    api_key: str
    model: str
    base_url: str
    sessions_dir: Path

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def redacted(self) -> dict[str, str]:
        """A form of this config that is safe to log or show in the UI."""
        key = self.api_key
        fingerprint = f"set ({len(key)} chars, ...{key[-4:]})" if key else "missing"
        return {
            "api_key": fingerprint,
            "model": self.model,
            "base_url": self.base_url,
            "sessions_dir": str(self.sessions_dir),
        }


def load_config(*, require_key: bool = True) -> Config:
    """Build a Config from the environment, with .env as the fallback source.

    Args:
        require_key: Raise ConfigError when no API key is present. Pass False for
            code paths that only need the model name or paths.
    """
    load_dotenv(REPO_ROOT / ".env")

    api_key = (os.environ.get("TAMU_API_KEY") or "").strip()
    if not api_key and require_key:
        raise ConfigError(
            "TAMU_API_KEY is not set.\n\n"
            f"Copy {REPO_ROOT / '.env.example'} to .env and fill in your key. "
            ".env is git-ignored, so it will not be committed."
        )

    return Config(
        api_key=api_key,
        model=(os.environ.get("SPICE_MCP_MODEL") or "").strip() or DEFAULT_MODEL,
        base_url=(os.environ.get("SPICE_MCP_BASE_URL") or "").strip()
        or DEFAULT_BASE_URL,
        sessions_dir=_resolve_sessions_dir(
            (os.environ.get("SPICE_MCP_SESSIONS_DIR") or "").strip()
        ),
    )


def _resolve_sessions_dir(raw: str) -> Path:
    """Anchor a relative SPICE_MCP_SESSIONS_DIR to the repo, not the cwd.

    The Explorer right-click launcher starts us with the cwd set to whichever folder the
    schematic lives in, so resolving a relative path against the cwd would scatter session
    logs into the user's source directory - the one thing this project promises never to
    write to. Absolute values are honoured as given.
    """
    if not raw:
        return SESSIONS_DIR
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate

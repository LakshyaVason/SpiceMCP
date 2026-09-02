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

# The TAMU Gateway exposes the OpenAI-compatible surface under /v1 on the gateway host.
# For example: https://gateway.api.tamu.ai/v1/chat/completions
DEFAULT_BASE_URL = "https://gateway.api.tamu.ai"

# Verified present on the proxy on 2026-08-29 via scripts/list_tamu_models.py. The
# space in the id is real - do not "clean it up".
DEFAULT_MODEL = "protected.Claude Opus 4.8"

# How the model is asked to call tools. Not derived from the model id: what the gateway
# does with the `tools` field is a property of the *route*, not of the family name, and
# guessing from a prefix would silently pick the wrong mode for the next model added.
#   native        - the OpenAI `tools` array; the reply carries `message.tool_calls`.
#                   Verified working on protected.Claude Opus 4.8.
#   prompted_json - `tools` is not sent at all. The tool list goes in the system prompt
#                   and the model replies with one JSON object per turn. For routes that
#                   complete normally but ignore `tools` - verified on
#                   us.anthropic.claude-opus-5, which answers as though no tools exist.
TOOL_MODE_NATIVE = "native"
TOOL_MODE_PROMPTED_JSON = "prompted_json"
TOOL_MODES = (TOOL_MODE_NATIVE, TOOL_MODE_PROMPTED_JSON)
DEFAULT_TOOL_MODE = TOOL_MODE_NATIVE

SESSIONS_DIR = REPO_ROOT / "sessions"


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


@dataclass(frozen=True)
class Config:
    api_key: str
    model: str
    base_url: str
    sessions_dir: Path
    tool_mode: str = DEFAULT_TOOL_MODE

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/chat/completions"

    def redacted(self) -> dict[str, str]:
        """A form of this config that is safe to log or show in the UI."""
        key = self.api_key
        fingerprint = f"set ({len(key)} chars, ...{key[-4:]})" if key else "missing"
        return {
            "api_key": fingerprint,
            "model": self.model,
            "base_url": self.base_url,
            "tool_mode": self.tool_mode,
            "sessions_dir": str(self.sessions_dir),
        }


def _normalize_base_url(raw: str) -> str:
    """Accept either the gateway host or an older path variant and normalize to host."""
    value = (raw or "").strip().rstrip("/")
    if not value:
        return DEFAULT_BASE_URL
    for suffix in ("/v1", "/openai", "/api"):
        if value.lower().endswith(suffix):
            value = value[: -len(suffix)]
    return value or DEFAULT_BASE_URL


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
        base_url=_normalize_base_url(os.environ.get("SPICE_MCP_BASE_URL")),
        tool_mode=_resolve_tool_mode(os.environ.get("SPICE_MCP_TOOL_MODE")),
        sessions_dir=_resolve_sessions_dir(
            (os.environ.get("SPICE_MCP_SESSIONS_DIR") or "").strip()
        ),
    )


def _resolve_tool_mode(raw: str | None) -> str:
    """Validate SPICE_MCP_TOOL_MODE, rejecting typos rather than defaulting past them.

    A silently ignored value here is the worst outcome: `prompted-json` with a hyphen
    would fall back to `native`, the model would ignore `tools`, and the app would look
    like it worked while never calling MCP - which is the bug this mode exists to fix.
    """
    value = (raw or "").strip().lower()
    if not value:
        return DEFAULT_TOOL_MODE
    if value not in TOOL_MODES:
        raise ConfigError(
            f"SPICE_MCP_TOOL_MODE={raw!r} is not a known mode. "
            f"Use one of: {', '.join(TOOL_MODES)}."
        )
    return value


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

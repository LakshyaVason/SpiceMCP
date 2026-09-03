"""Configuration for the app half: gateway URL, model, tool mode, and paths.

The LLM is reached through the **TAMU AI Gateway**, not through AWS. The gateway exposes
an Anthropic-shaped endpoint at `{base_url}/v1/messages` and routes to a backend provider
itself, so a model id like `us.anthropic.claude-opus-5` names *the gateway's* backend and
implies nothing about this machine. There is no AWS region here, no credential chain, and
no boto3: authentication is a single bearer token, `TAMU_API_KEY`.

**`Config` still holds no credential.** That is the same security property the AWS version
had, kept deliberately rather than inherited: the token is read from the environment by
`GatewayClient` at the moment it constructs the SDK, so there is nothing in this dataclass
for a stray debug print or a `repr()` in a traceback to leak. `credentials_source` is a
label - `"TAMU_API_KEY (51 chars)"` or `"none"` - and the length is there to catch a
truncated paste. Note what it is *not*: a last-four fingerprint. Those are conventional,
but four characters of a live token are still four characters of a live token, and this
way `tests/test_config.py` can sweep `redacted()` for any substring of the secret and
find none.

`_detect_credentials_source` reads one variable name and never touches the network, so
`load_config()` stays cheap and offline - the UI calls it on every launch and the test
suite calls it constantly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

# The gateway's root. The SDK appends `/v1/messages` itself, so this must NOT include a
# version segment - see `_normalize_base_url`, which trims one if a pasted URL has it.
DEFAULT_BASE_URL = "https://gateway.api.tamu.ai"

# A model id the gateway routes, not a display name and not an AWS resource. The `us.`
# prefix belongs to the gateway's upstream inference profile; it is billed to TAMU, and
# the 10% regional premium on that profile is a fact about the gateway's costs rather than
# something this client can choose.
DEFAULT_MODEL = "us.anthropic.claude-opus-5"

SESSIONS_DIR = REPO_ROOT / "sessions"

# The bearer token's variable name, in one place: `mcp_client.py` imports this to keep the
# token out of the MCP server's environment, and importing beats re-typing the string in a
# second file where a rename could silently miss it.
TOKEN_ENV_VAR = "TAMU_API_KEY"

NO_CREDENTIALS = "none"

# How the model is asked to call tools. `native` uses the Messages API `tools` parameter
# and reads `tool_use` blocks back; `prompted_json` puts the tool catalogue in the system
# prompt and parses one JSON object out of the reply.
#
# This is explicit configuration rather than a check on the model id on purpose: the
# failure being worked around is a property of a *route* - a particular model reached
# through a particular endpoint - and route behaviour is not derivable from a name. The
# same model on the same gateway honours `tools` on `/v1/messages` and ignores them on
# `/v1/chat/completions`, which is exactly why the name tells you nothing.
TOOL_MODE_NATIVE = "native"
TOOL_MODE_PROMPTED_JSON = "prompted_json"
TOOL_MODES = (TOOL_MODE_NATIVE, TOOL_MODE_PROMPTED_JSON)

# Native is the default: `scripts/probe_tool_calling.py` confirmed live on 2026-09-02 that
# the gateway's `/v1/messages` returns real `tool_use` blocks for the default model, with
# usable token counts. `prompted_json` stays available and fully supported for routes that
# accept `tools` and then answer as though they had none - which is not hypothetical, it is
# what this gateway's `/v1/chat/completions` path does with the very same model.
DEFAULT_TOOL_MODE = TOOL_MODE_NATIVE


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


@dataclass(frozen=True)
class Config:
    model: str
    base_url: str
    credentials_source: str
    sessions_dir: Path
    tool_mode: str = DEFAULT_TOOL_MODE

    @property
    def messages_url(self) -> str:
        """Where requests actually go. For banners and error messages only.

        The SDK builds this itself from `base_url`; duplicating the join here is for
        human-readable diagnostics, not for making requests.
        """
        return f"{self.base_url.rstrip('/')}/v1/messages"

    def redacted(self) -> dict[str, str]:
        """A form of this config that is safe to log or show in the UI.

        Safe by construction rather than by filtering: no field here has ever held the
        token. `credentials_source` is a label with a length, not a fingerprint.
        """
        return {
            "model": self.model,
            "base_url": self.base_url,
            "credentials": self.credentials_source,
            "sessions_dir": str(self.sessions_dir),
            "tool_mode": self.tool_mode,
        }


def _detect_credentials_source() -> str:
    """Label the bearer token without revealing it.

    The length is the useful part: it separates "set" from "set but truncated by a bad
    copy-paste", which is a real failure that otherwise surfaces as an opaque 401.
    """
    token = (os.environ.get(TOKEN_ENV_VAR) or "").strip()
    if not token:
        return NO_CREDENTIALS
    return f"{TOKEN_ENV_VAR} ({len(token)} chars)"


def _normalize_base_url(raw: str) -> str:
    """Trim a trailing /v1, /openai or /api off a pasted gateway URL.

    The SDK appends `/v1/messages` to whatever it is given, so a base URL that already
    ends in `/v1` produces `/v1/v1/messages`. That 404s, and the SDK's 404 reads like a
    missing model - so the diagnosis lands a long way from the cause. Cheaper to accept
    the URL the user is most likely to paste.
    """
    value = (raw or "").strip().rstrip("/")
    if not value:
        return DEFAULT_BASE_URL
    for suffix in ("/v1", "/openai", "/api"):
        if value.lower().endswith(suffix):
            value = value[: -len(suffix)]
    return value or DEFAULT_BASE_URL


def load_config(*, require_credentials: bool = True) -> Config:
    """Build a Config from the environment, with .env as the fallback source.

    Args:
        require_credentials: Raise ConfigError when the bearer token is missing. Pass
            False for code paths that only need the model name or paths.
    """
    load_dotenv(REPO_ROOT / ".env")

    credentials_source = _detect_credentials_source()
    if credentials_source == NO_CREDENTIALS and require_credentials:
        raise ConfigError(
            f"{TOKEN_ENV_VAR} is not set, so the AI gateway cannot be reached.\n\n"
            f"Put your gateway token in {REPO_ROOT / '.env'} as:\n"
            f"  {TOKEN_ENV_VAR}=<your token>\n\n"
            f"Copy {REPO_ROOT / '.env.example'} to .env if you have not already. .env is "
            "git-ignored, so it will not be committed. The token is sent as an "
            "Authorization: Bearer header and is never written to a log or a session file."
        )

    return Config(
        model=(os.environ.get("SPICE_MCP_MODEL") or "").strip() or DEFAULT_MODEL,
        base_url=_normalize_base_url(os.environ.get("SPICE_MCP_BASE_URL") or ""),
        credentials_source=credentials_source,
        sessions_dir=_resolve_sessions_dir(
            (os.environ.get("SPICE_MCP_SESSIONS_DIR") or "").strip()
        ),
        tool_mode=_resolve_tool_mode(os.environ.get("SPICE_MCP_TOOL_MODE")),
    )


def _resolve_tool_mode(raw: str | None) -> str:
    """Validate SPICE_MCP_TOOL_MODE, rejecting typos rather than defaulting past them.

    A silently ignored value here is the worst outcome: `prompted-json` with a hyphen
    would fall back to `native`, and if the route in use happens to ignore `tools`, the
    app would look like it worked while never calling MCP - which is the bug this mode
    exists to fix.
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

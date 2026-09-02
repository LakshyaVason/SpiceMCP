"""Configuration for the app half: Bedrock region, model, and paths.

Bedrock auth is IAM-based, so there is no bearer key to look after. Credentials are
resolved by the AWS default chain *inside the SDK* - environment keys, a session token,
`AWS_PROFILE`, `~/.aws`, an instance role, or an `AWS_BEARER_TOKEN_BEDROCK`. That is a
deliberate security property rather than laziness: **`Config` holds no credential at
all**, so there is nothing here for a stray debug print or a `repr()` in a traceback to
leak. `redacted()` reports only which *style* of credential was found, never its value.

`_detect_credentials_source` is a pure env/filesystem probe. It never imports boto3 and
never touches the network, so `load_config()` stays cheap and offline - the UI calls it
on every launch and the test suite calls it constantly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

# A Bedrock cross-region inference profile id, not a display name. The `us.` prefix is a
# US regional profile, which Bedrock bills at a 10% premium over the `global.` profile -
# relevant when reading this project's cost comparison. The bare `anthropic.claude-opus-5`
# id is rejected for CRIS models, so the prefix is required, not decorative.
DEFAULT_MODEL = "us.anthropic.claude-opus-5"

# Inference profile availability is region-scoped, so this has to be explicit somewhere.
DEFAULT_REGION = "us-east-1"

SESSIONS_DIR = REPO_ROOT / "sessions"

# Checked in precedence order, mirroring the SDK's own chain closely enough to give a
# useful answer. The values are labels for the UI and the log - never a credential.
NO_CREDENTIALS = "none"


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


@dataclass(frozen=True)
class Config:
    model: str
    aws_region: str
    credentials_source: str
    sessions_dir: Path
    tool_mode: str = DEFAULT_TOOL_MODE

    def redacted(self) -> dict[str, str]:
        """A form of this config that is safe to log or show in the UI.

        Safe by construction rather than by filtering: no field here has ever held a
        secret. `credentials_source` is a label like "environment keys", not a
        fingerprint of the key.
        """
        return {
            "model": self.model,
            "aws_region": self.aws_region,
            "credentials": self.credentials_source,
            "sessions_dir": str(self.sessions_dir),
        }


def _detect_credentials_source() -> str:
    """Name the credential style the AWS chain will find, without resolving it.

    Deliberately reads nothing but variable *names* and file *existence*. Returning
    NO_CREDENTIALS lets the UI show a setup banner instead of the SDK raising deep
    inside the first request.
    """
    if (os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or "").strip():
        return "bearer token"
    if (os.environ.get("AWS_ACCESS_KEY_ID") or "").strip() and (
        os.environ.get("AWS_SECRET_ACCESS_KEY") or ""
    ).strip():
        return "environment keys"
    profile = (os.environ.get("AWS_PROFILE") or "").strip()
    if profile:
        return f"profile:{profile}"
    # A container or EC2 role supplies credentials through the metadata service, which we
    # would have to make a network call to confirm. Trust the marker variables instead.
    if (os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI") or "").strip() or (
        os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE") or ""
    ).strip():
        return "container or web identity role"
    aws_dir = Path.home() / ".aws"
    if (aws_dir / "credentials").exists() or (aws_dir / "config").exists():
        return "shared config file"
    return NO_CREDENTIALS


def _resolve_region() -> str:
    """SPICE_MCP_AWS_REGION wins, then the standard AWS variables, then the default."""
    for name in ("SPICE_MCP_AWS_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"):
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return DEFAULT_REGION


def load_config(*, require_credentials: bool = True) -> Config:
    """Build a Config from the environment, with .env as the fallback source.

    Args:
        require_credentials: Raise ConfigError when the AWS chain has nothing to find.
            Pass False for code paths that only need the model name or paths.
    """
    load_dotenv(REPO_ROOT / ".env")

    credentials_source = _detect_credentials_source()
    if credentials_source == NO_CREDENTIALS and require_credentials:
        raise ConfigError(
            "No AWS credentials found, so Bedrock cannot be reached.\n\n"
            "Any one of these works - the AWS default chain finds them all:\n"
            "  * AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (plus AWS_SESSION_TOKEN "
            "for temporary credentials)\n"
            "  * AWS_PROFILE, with the profile configured in ~/.aws/credentials\n"
            "  * AWS_BEARER_TOKEN_BEDROCK, if your organisation issued a Bedrock token "
            "instead of IAM credentials\n\n"
            f"Copy {REPO_ROOT / '.env.example'} to .env and fill in whichever style you "
            "have. .env is git-ignored, so it will not be committed."
        )

    return Config(
        model=(os.environ.get("SPICE_MCP_MODEL") or "").strip() or DEFAULT_MODEL,
        aws_region=_resolve_region(),
        credentials_source=credentials_source,
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

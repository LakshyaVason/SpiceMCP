"""Tests for configuration and for what the server subprocess inherits.

Both of these became load-bearing with the Explorer launcher. It starts with the working
directory set to whichever folder the user's schematic lives in, so anything that resolves a
relative path against the cwd would write into the user's source directory - the one thing
this project promises never to do.

The credential tests are the other half: Bedrock auth is IAM-based, and the design decision
is that no credential is ever stored in `Config` at all. These assert that property directly
rather than trusting it.
"""

from __future__ import annotations

from pathlib import Path

from spice_mcp_app.config import (
    DEFAULT_REGION,
    NO_CREDENTIALS,
    REPO_ROOT,
    SESSIONS_DIR,
    load_config,
)
from spice_mcp_app.mcp_client import _server_params


def with_keys(monkeypatch) -> None:
    """Give the credential probe something to find, so load_config does not refuse."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAFAKE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake-secret")


def test_a_relative_sessions_dir_resolves_against_the_repo_not_the_cwd(monkeypatch, tmp_path):
    """Launched from Explorer the cwd is the user's circuit folder, not the repo."""
    monkeypatch.chdir(tmp_path)
    with_keys(monkeypatch)
    monkeypatch.setenv("SPICE_MCP_SESSIONS_DIR", "sessions")

    assert load_config().sessions_dir == REPO_ROOT / "sessions"


def test_an_absolute_sessions_dir_is_honoured_as_given(monkeypatch, tmp_path):
    with_keys(monkeypatch)
    monkeypatch.setenv("SPICE_MCP_SESSIONS_DIR", str(tmp_path / "logs"))

    assert load_config().sessions_dir == tmp_path / "logs"


def test_the_default_sessions_dir_is_absolute_and_inside_the_repo():
    assert SESSIONS_DIR.is_absolute()
    assert SESSIONS_DIR.parent == REPO_ROOT


# --- credentials ----------------------------------------------------------------------


def test_the_credential_style_is_detected_without_resolving_it(monkeypatch):
    with_keys(monkeypatch)
    assert load_config().credentials_source == "environment keys"

    monkeypatch.setenv("AWS_PROFILE", "spice")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY")
    assert load_config().credentials_source == "profile:spice"


def test_no_credentials_is_reported_rather_than_guessed(monkeypatch):
    """The autouse fixture leaves the chain nothing to find, which is the point here."""
    config = load_config(require_credentials=False)
    assert config.credentials_source == NO_CREDENTIALS


def test_redacted_can_never_carry_a_secret(monkeypatch):
    """The redaction guarantee, asserted rather than assumed.

    `Config` holds no credential, so this is a property of the shape and not of a filter -
    which is why the sweep looks for the secret's *value* anywhere in the output.
    """
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAFAKE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "super-secret-value")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "another-secret-value")

    redacted = load_config().redacted()

    haystack = "".join(str(v) for v in redacted.values()) + repr(load_config())
    assert "super-secret-value" not in haystack
    assert "another-secret-value" not in haystack
    assert "AKIAFAKE" not in haystack
    # It still has to say something useful, or the banner is pointless.
    assert redacted["credentials"] == "environment keys"


def test_the_region_falls_back_through_the_standard_aws_variables(monkeypatch):
    with_keys(monkeypatch)
    assert load_config().aws_region == "us-east-1"

    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-southeast-2")
    assert load_config().aws_region == "ap-southeast-2"

    monkeypatch.setenv("SPICE_MCP_AWS_REGION", "eu-west-1")
    assert load_config().aws_region == "eu-west-1"


# --- what the server subprocess inherits ----------------------------------------------


def test_the_server_inherits_the_ltspice_override(monkeypatch):
    """`.env.example` documents LTSPICE_EXE as working; a replaced env made that false."""
    monkeypatch.setenv("LTSPICE_EXE", r"C:\custom\LTspice.exe")

    env = _server_params().env

    assert env["LTSPICE_EXE"] == r"C:\custom\LTspice.exe"
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_the_server_is_not_given_the_aws_credentials(monkeypatch):
    """The server must never learn what an LLM is, and has no use for the credentials.

    `AWS_ROLE_ARN` is in the list on purpose: it is named nowhere in `mcp_client.py`. The
    deny rule is by *prefix* because an enumerated list fails silently - miss a variable
    the chain grew and the secret travels while this test still passes.
    """
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "should-not-travel")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "nor-should-this")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "nor-this-either")
    monkeypatch.setenv("AWS_ROLE_ARN", "arn:aws:iam::1:role/nowhere")
    monkeypatch.setenv("SPICE_MCP_MODEL", "some-model")
    monkeypatch.setenv("SPICE_MCP_AWS_REGION", "us-east-1")

    env = _server_params().env

    assert not [name for name in env if name.startswith("AWS_")]
    assert "SPICE_MCP_MODEL" not in env
    assert "SPICE_MCP_AWS_REGION" not in env
    assert "should-not-travel" not in "".join(env.values())
    assert "nor-should-this" not in "".join(env.values())


def test_the_server_runs_from_the_repo_root(monkeypatch, tmp_path):
    """So a relative circuit path never resolves into somewhere unexpected."""
    monkeypatch.chdir(tmp_path)
    assert Path(_server_params().cwd) == REPO_ROOT


def test_the_model_listing_resolves_the_same_region_as_the_app(monkeypatch):
    """The helper script keeps its own copy of the resolver; it must not drift.

    It is standalone so it can run from `scripts/requirements.txt` alone, which is exactly
    the situation where a divergence would go unnoticed - the script would report on one
    region while the app called another.
    """
    monkeypatch.setenv("SPICE_MCP_AWS_REGION", "eu-west-2")

    import importlib.util

    module_path = REPO_ROOT / "scripts" / "list_bedrock_models.py"
    spec = importlib.util.spec_from_file_location("list_bedrock_models", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.resolve_region() == "eu-west-2"
    assert module.resolve_region() == load_config(require_credentials=False).aws_region
    assert module.DEFAULT_REGION == DEFAULT_REGION

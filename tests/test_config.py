"""Tests for configuration and for what the server subprocess inherits.

Both of these became load-bearing with the Explorer launcher. It starts with the working
directory set to whichever folder the user's schematic lives in, so anything that resolves a
relative path against the cwd would write into the user's source directory - the one thing
this project promises never to do.

The credential tests are the other half. Auth is a single bearer token for the TAMU AI
Gateway, and the design decision is that the token is never stored in `Config` at all - it
is read from the environment by `GatewayClient` at the moment it builds the SDK. These
assert that property directly rather than trusting it, and they assert the token does not
travel into the MCP server subprocess, which is a leak that really happened: a refactor
replaced the explicit `TAMU_API_KEY` deny entry with an `AWS_`-prefix rule, and for a while
the token reached a process that knows nothing about LLMs while every test still passed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spice_mcp_app.config import (
    DEFAULT_BASE_URL,
    DEFAULT_TOOL_MODE,
    NO_CREDENTIALS,
    REPO_ROOT,
    SESSIONS_DIR,
    TOKEN_ENV_VAR,
    TOOL_MODE_NATIVE,
    TOOL_MODE_PROMPTED_JSON,
    ConfigError,
    load_config,
)
from spice_mcp_app.mcp_client import _server_params

# Long enough that a 4-character window of it is a meaningful test - see the redaction
# sweep below. No dictionary words: an earlier version began with "tamu-", which the sweep
# then found inside the gateway hostname and reported as a leak.
FAKE_TOKEN = "kQ7vR2mX9wZ4hL8pB6nJ3sD5fG1yT0cA"


def with_token(monkeypatch, token: str = FAKE_TOKEN) -> None:
    """Give the credential probe something to find, so load_config does not refuse."""
    monkeypatch.setenv(TOKEN_ENV_VAR, token)


def _rendered(config) -> str:
    """Everything a stray log line or traceback could expose about a Config."""
    return "".join(str(v) for v in config.redacted().values()) + repr(config)


def test_a_relative_sessions_dir_resolves_against_the_repo_not_the_cwd(monkeypatch, tmp_path):
    """Launched from Explorer the cwd is the user's circuit folder, not the repo."""
    monkeypatch.chdir(tmp_path)
    with_token(monkeypatch)
    monkeypatch.setenv("SPICE_MCP_SESSIONS_DIR", "sessions")

    assert load_config().sessions_dir == REPO_ROOT / "sessions"


def test_an_absolute_sessions_dir_is_honoured_as_given(monkeypatch, tmp_path):
    with_token(monkeypatch)
    monkeypatch.setenv("SPICE_MCP_SESSIONS_DIR", str(tmp_path / "logs"))

    assert load_config().sessions_dir == tmp_path / "logs"


def test_the_default_sessions_dir_is_absolute_and_inside_the_repo():
    assert SESSIONS_DIR.is_absolute()
    assert SESSIONS_DIR.parent == REPO_ROOT


# --- credentials ----------------------------------------------------------------------


def test_the_token_is_labelled_with_its_length_and_never_its_value(monkeypatch):
    """The label has to distinguish "set" from "truncated paste" without revealing anything."""
    with_token(monkeypatch)
    label = load_config().credentials_source
    assert label == f"{TOKEN_ENV_VAR} ({len(FAKE_TOKEN)} chars)"
    assert FAKE_TOKEN not in label


def test_a_missing_token_is_reported_rather_than_guessed(monkeypatch):
    """The autouse fixture clears the token, which is the point here."""
    config = load_config(require_credentials=False)
    assert config.credentials_source == NO_CREDENTIALS


def test_a_missing_token_refuses_loudly_by_default(monkeypatch):
    """The app must say what to do, not fail later as an opaque 401."""
    with pytest.raises(ConfigError, match=TOKEN_ENV_VAR):
        load_config()


def test_whitespace_only_token_counts_as_missing(monkeypatch):
    """A trailing newline from a copy-paste must not read as "credential present"."""
    with_token(monkeypatch, "   \n  ")
    assert load_config(require_credentials=False).credentials_source == NO_CREDENTIALS


def test_redacted_can_never_carry_a_secret(monkeypatch):
    """The redaction guarantee, asserted rather than assumed.

    `Config` holds no credential, so this is a property of the shape and not of a filter -
    which is why the sweep looks for the secret's *value* anywhere in the output.

    Stronger than a plain substring check on the whole token: it slides a 4-character
    window across it, so a conventional last-four fingerprint would fail here. Four
    characters of a live token are still four characters of a live token, and the length
    label carries all the diagnostic value a fingerprint would.

    A short window will match by coincidence - the model id, the hostname and the repo path
    are all in the output - so the comparison is differential. Whatever the config renders
    with *no* token set is the baseline, and only fragments that appear once a token *is*
    set can have come from it.
    """
    baseline = _rendered(load_config(require_credentials=False))

    with_token(monkeypatch)
    config = load_config()
    redacted = config.redacted()
    haystack = _rendered(config)

    assert FAKE_TOKEN not in haystack
    leaked = [
        window
        for i in range(len(FAKE_TOKEN) - 3)
        if (window := FAKE_TOKEN[i : i + 4]) in haystack and window not in baseline
    ]
    assert not leaked, f"fragments of the token appear in redacted output: {leaked}"

    # It still has to say something useful, or the banner is pointless.
    assert redacted["credentials"] == f"{TOKEN_ENV_VAR} ({len(FAKE_TOKEN)} chars)"


def test_no_config_field_holds_the_token(monkeypatch):
    """Belt and braces on the shape itself, independent of `redacted()`.

    A future field that carried the token would keep `redacted()` passing if it were simply
    left out of the dict; this notices it anyway.
    """
    import dataclasses

    with_token(monkeypatch)
    config = load_config()

    for f in dataclasses.fields(config):
        assert FAKE_TOKEN not in str(getattr(config, f.name)), f.name


# --- the gateway URL ------------------------------------------------------------------


def test_the_base_url_defaults_to_the_gateway(monkeypatch):
    with_token(monkeypatch)
    config = load_config()
    assert config.base_url == DEFAULT_BASE_URL
    assert config.messages_url == f"{DEFAULT_BASE_URL}/v1/messages"


def test_the_base_url_can_be_overridden(monkeypatch):
    with_token(monkeypatch)
    monkeypatch.setenv("SPICE_MCP_BASE_URL", "https://gateway.example.edu")
    assert load_config().base_url == "https://gateway.example.edu"


@pytest.mark.parametrize(
    "raw",
    [
        "https://gateway.api.tamu.ai/v1",
        "https://gateway.api.tamu.ai/v1/",
        "https://gateway.api.tamu.ai/",
        "  https://gateway.api.tamu.ai  ",
        "https://gateway.api.tamu.ai/openai",
        "https://gateway.api.tamu.ai/api",
    ],
)
def test_a_version_suffix_is_trimmed_off_the_base_url(monkeypatch, raw):
    """The SDK appends /v1/messages itself, so a pasted /v1 would produce /v1/v1/messages.

    That 404s, and a 404 from the gateway reads like a missing model - so the diagnosis
    would land a long way from the cause.
    """
    with_token(monkeypatch)
    monkeypatch.setenv("SPICE_MCP_BASE_URL", raw)

    config = load_config()
    assert config.base_url == DEFAULT_BASE_URL
    assert config.messages_url.count("/v1") == 1


def test_an_empty_base_url_falls_back_to_the_default(monkeypatch):
    with_token(monkeypatch)
    monkeypatch.setenv("SPICE_MCP_BASE_URL", "   ")
    assert load_config().base_url == DEFAULT_BASE_URL


def test_no_aws_variable_influences_the_config(monkeypatch):
    """This app has no AWS identity. A stray AWS_REGION in a shell must change nothing.

    Named variables rather than a loop, because the point is that these specific ones -
    which the previous transport *did* read - are now inert.
    """
    with_token(monkeypatch)
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-southeast-2")
    monkeypatch.setenv("AWS_PROFILE", "spice")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAFAKE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "super-secret-value")

    config = load_config()

    assert config.base_url == DEFAULT_BASE_URL
    assert config.credentials_source == f"{TOKEN_ENV_VAR} ({len(FAKE_TOKEN)} chars)"
    assert "super-secret-value" not in repr(config)
    assert not hasattr(config, "aws_region")


def test_aws_credentials_alone_are_not_treated_as_credentials(monkeypatch):
    """The gateway needs its own token; AWS keys are not a substitute and must not read as one."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAFAKE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake-secret")

    assert load_config(require_credentials=False).credentials_source == NO_CREDENTIALS
    with pytest.raises(ConfigError, match=TOKEN_ENV_VAR):
        load_config()


# --- the tool-calling mode -------------------------------------------------------------


@pytest.fixture
def env(monkeypatch):
    """Credentials present, so only the variable under test decides the outcome."""
    with_token(monkeypatch)
    monkeypatch.delenv("SPICE_MCP_TOOL_MODE", raising=False)
    return monkeypatch


def test_the_default_tool_mode_is_native(env):
    """The gateway's /v1/messages path does honour `tools`, verified live by the probe."""
    assert load_config().tool_mode == TOOL_MODE_NATIVE == DEFAULT_TOOL_MODE


def test_the_tool_mode_is_read_from_the_environment(env):
    env.setenv("SPICE_MCP_TOOL_MODE", "prompted_json")
    assert load_config().tool_mode == TOOL_MODE_PROMPTED_JSON


def test_the_tool_mode_is_case_and_space_insensitive(env):
    env.setenv("SPICE_MCP_TOOL_MODE", "  Prompted_JSON ")
    assert load_config().tool_mode == TOOL_MODE_PROMPTED_JSON


def test_a_misspelled_tool_mode_is_rejected_rather_than_ignored(env):
    """Falling back to native on a typo is the failure this mode exists to fix.

    `prompted-json` with a hyphen would send `tools` to a route that ignores it, and the
    app would look like it worked while never calling MCP.
    """
    env.setenv("SPICE_MCP_TOOL_MODE", "prompted-json")
    with pytest.raises(ConfigError, match="SPICE_MCP_TOOL_MODE"):
        load_config()


def test_the_tool_mode_appears_in_the_config_the_ui_is_shown(env):
    """The UI shows `redacted()`; a wrong mode is otherwise invisible from inside the app."""
    env.setenv("SPICE_MCP_TOOL_MODE", "prompted_json")
    assert load_config().redacted()["tool_mode"] == "prompted_json"


# --- the effort knob ------------------------------------------------------------------
#
# `output_config={"effort": ...}` is the only lever on thinking tokens that exists here, and
# `usage.output_tokens` includes them - about 770 of the baseline turn's ~1350. It is
# optional and probed rather than assumed, so the default path has to stay untouched.


def test_effort_is_unset_by_default(env):
    """Unset means the parameter is omitted, so behaviour is identical to before it existed.

    `high` is already the model's own default; sending it explicitly would make an unset
    configuration indistinguishable from a deliberate one.
    """
    assert load_config().effort is None


def test_effort_is_read_from_the_environment(env):
    env.setenv("SPICE_MCP_EFFORT", " LOW ")
    assert load_config().effort == "low"


def test_a_misspelled_effort_is_rejected_rather_than_ignored(env):
    """Same argument as the tool mode: someone who set `lo` to cut the cost of a demo would
    otherwise be charged the full price with no explanation."""
    env.setenv("SPICE_MCP_EFFORT", "lo")
    with pytest.raises(ConfigError, match="SPICE_MCP_EFFORT"):
        load_config()


def test_temperature_is_not_a_setting_anywhere(env):
    """It was the obvious first guess and it is wrong twice over.

    `temperature` is not a parameter of `messages.create` in anthropic 1.3.0, and forcing it
    through `extra_body` is a 400 on Opus 5. It also controls sampling variability rather
    than length. This test exists so a future reader does not spend the afternoon rediscovering
    that.
    """
    import spice_mcp_app.config as config_module
    import spice_mcp_app.llm as llm_module

    for module in (config_module, llm_module):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for banned in ("temperature=", "top_p", "top_k"):
            assert banned not in source, f"{banned} appeared in {module.__name__}"


def test_the_effort_appears_in_the_config_the_ui_is_shown(env):
    env.setenv("SPICE_MCP_EFFORT", "medium")
    assert load_config().redacted()["effort"] == "medium"
    env.delenv("SPICE_MCP_EFFORT")
    assert load_config().redacted()["effort"] == "default (unset)"


# --- what the server subprocess inherits ----------------------------------------------


def test_the_server_inherits_the_ltspice_override(monkeypatch):
    """`.env.example` documents LTSPICE_EXE as working; a replaced env made that false."""
    monkeypatch.setenv("LTSPICE_EXE", r"C:\custom\LTspice.exe")

    env = _server_params().env

    assert env["LTSPICE_EXE"] == r"C:\custom\LTspice.exe"
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_the_server_is_never_given_the_gateway_token(monkeypatch):
    """The regression test for a leak that actually shipped.

    When the transport moved to AWS, the explicit TAMU_API_KEY deny entry was replaced by
    an `AWS_`-prefix rule. The token then travelled into the MCP server - a process that
    knows nothing about LLMs and has no use for it - and no test noticed, because the
    prefix rule looked like it covered "the credential".

    Asserted through `_server_params()` rather than by reading `_LLM_ONLY_ENV`, so it is
    the built environment that is checked and not the intention behind it.
    """
    monkeypatch.setenv(TOKEN_ENV_VAR, "should-not-travel-to-the-server")
    monkeypatch.setenv("SPICE_MCP_BASE_URL", "https://gateway.example.edu")
    monkeypatch.setenv("SPICE_MCP_MODEL", "some-model")
    monkeypatch.setenv("SPICE_MCP_TOOL_MODE", "prompted_json")

    env = _server_params().env

    assert TOKEN_ENV_VAR not in env
    assert "should-not-travel-to-the-server" not in "".join(env.values())
    assert "SPICE_MCP_BASE_URL" not in env
    assert "SPICE_MCP_MODEL" not in env
    # How the *model* is asked to call tools is not the server's business either.
    assert "SPICE_MCP_TOOL_MODE" not in env


def test_the_server_is_not_given_stray_aws_credentials(monkeypatch):
    """Kept after the migration: the prefix rule is cheap belt-and-braces.

    This app has no AWS identity, but a user's shell may well carry AWS credentials for
    unrelated work, and there is still no reason to hand them to a netlist parser.
    `AWS_ROLE_ARN` is here on purpose - it is named nowhere in `mcp_client.py`, so this
    asserts the prefix rule rather than an enumerated list.
    """
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "should-not-travel")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "nor-should-this")
    monkeypatch.setenv("AWS_ROLE_ARN", "arn:aws:iam::1:role/nowhere")

    env = _server_params().env

    assert not [name for name in env if name.startswith("AWS_")]
    assert "should-not-travel" not in "".join(env.values())
    assert "nor-should-this" not in "".join(env.values())


def test_the_server_runs_from_the_repo_root(monkeypatch, tmp_path):
    """So a relative circuit path never resolves into somewhere unexpected."""
    monkeypatch.chdir(tmp_path)
    assert Path(_server_params().cwd) == REPO_ROOT

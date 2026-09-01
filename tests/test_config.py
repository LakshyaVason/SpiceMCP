"""Tests for configuration and for what the server subprocess inherits.

Both of these became load-bearing with the Explorer launcher. It starts with the working
directory set to whichever folder the user's schematic lives in, so anything that resolves a
relative path against the cwd would write into the user's source directory - the one thing
this project promises never to do.
"""

from __future__ import annotations

from pathlib import Path

from spice_mcp_app.config import REPO_ROOT, SESSIONS_DIR, load_config
from spice_mcp_app.mcp_client import _server_params


def test_a_relative_sessions_dir_resolves_against_the_repo_not_the_cwd(monkeypatch, tmp_path):
    """Launched from Explorer the cwd is the user's circuit folder, not the repo."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TAMU_API_KEY", "x")
    monkeypatch.setenv("SPICE_MCP_SESSIONS_DIR", "sessions")
    monkeypatch.setattr("spice_mcp_app.config.load_dotenv", lambda *a, **k: None)

    assert load_config().sessions_dir == REPO_ROOT / "sessions"


def test_an_absolute_sessions_dir_is_honoured_as_given(monkeypatch, tmp_path):
    monkeypatch.setenv("TAMU_API_KEY", "x")
    monkeypatch.setenv("SPICE_MCP_SESSIONS_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr("spice_mcp_app.config.load_dotenv", lambda *a, **k: None)

    assert load_config().sessions_dir == tmp_path / "logs"


def test_the_default_sessions_dir_is_absolute_and_inside_the_repo():
    assert SESSIONS_DIR.is_absolute()
    assert SESSIONS_DIR.parent == REPO_ROOT


def test_the_server_inherits_the_ltspice_override(monkeypatch):
    """`.env.example` documents LTSPICE_EXE as working; a replaced env made that false."""
    monkeypatch.setenv("LTSPICE_EXE", r"C:\custom\LTspice.exe")

    env = _server_params().env

    assert env["LTSPICE_EXE"] == r"C:\custom\LTspice.exe"
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_the_server_is_not_given_the_api_key(monkeypatch):
    """The server must never learn what an LLM is, and has no use for the key."""
    monkeypatch.setenv("TAMU_API_KEY", "sk-should-not-travel")
    monkeypatch.setenv("SPICE_MCP_MODEL", "some-model")
    monkeypatch.setenv("SPICE_MCP_BASE_URL", "https://example.invalid")

    env = _server_params().env

    assert "TAMU_API_KEY" not in env
    assert "SPICE_MCP_MODEL" not in env
    assert "SPICE_MCP_BASE_URL" not in env
    assert "sk-should-not-travel" not in "".join(env.values())


def test_the_server_runs_from_the_repo_root(monkeypatch, tmp_path):
    """So a relative circuit path never resolves into somewhere unexpected."""
    monkeypatch.chdir(tmp_path)
    assert Path(_server_params().cwd) == REPO_ROOT


def test_the_model_listing_uses_the_shared_base_url_override(monkeypatch):
    """The helper script should honor the same global override as the app."""
    monkeypatch.setenv("SPICE_MCP_BASE_URL", "https://example.invalid/openai")

    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).resolve().parent.parent / "scripts" / "list_tamu_models.py"
    spec = importlib.util.spec_from_file_location("list_tamu_models", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    assert module.resolve_base_url() == "https://example.invalid"

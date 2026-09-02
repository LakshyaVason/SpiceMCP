from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "fixtures"


def _ltspice_available() -> bool:
    try:
        from spice_mcp_server.ltspice import find_ltspice_exe

        find_ltspice_exe()
        return True
    except Exception:
        return False


#

# Tests that need a .asc converted to a netlist require the real executable. Text-level
# parser and check tests do not, and must keep passing without LTspice installed.
needs_ltspice = pytest.mark.skipif(
    not _ltspice_available(),
    reason="LTspice executable not found; set LTSPICE_EXE to run schematic tests.",
)


@pytest.fixture(autouse=True)
def isolated_aws_environment(monkeypatch, tmp_path):
    """Cut every route the AWS chain has to real credentials, for every test.

    Not defensive tidiness - the suite is specified to be offline and free, and Bedrock's
    credential resolution is designed to succeed from places no test mentions: ambient
    `AWS_*` variables, `~/.aws/credentials`, an SSO cache, an instance role. Left alone,
    a "credentials are missing" test would pass on a laptop with no AWS setup and fail on
    the developer's machine that has one, and a mis-wired test could reach the live API
    and spend money.

    `HOME`/`USERPROFILE` move to tmp_path so `Path.home() / ".aws"` cannot exist, and
    `load_dotenv` is neutered so the repo's own git-ignored `.env` - which does hold real
    settings - never leaks into a test. Tests that want a credential set one explicitly.

    `LTSPICE_EXE` is deliberately left alone: it is not an LLM setting, and the
    `needs_ltspice` tests depend on it.
    """
    for name in list(os.environ):
        if name.startswith("AWS_") or name.startswith("SPICE_MCP_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    # Imported here rather than at module scope so the server-half tests do not acquire a
    # dependency on the app half's packages.
    import spice_mcp_app.config as app_config

    monkeypatch.setattr(app_config, "load_dotenv", lambda *a, **k: None)


@pytest.fixture
def fake_config(tmp_path):
    """A Config that names no real model, region or account.

    `credentials_source` is a label the UI prints, never a credential - so there is
    nothing here to leak even if a test dumps it.
    """
    from spice_mcp_app.config import Config

    return Config(
        model="fake-model",
        aws_region="us-east-1",
        credentials_source="environment keys",
        sessions_dir=tmp_path,
    )


class FakeMCP:
    """Records tool calls; answers the two tools the API paths actually inspect."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def call_tool(self, name, arguments, timeout=300.0):
        self.calls.append((name, arguments))
        if name == "check_netlist_static":
            return json.dumps(
                {"summary": "Static checks clean.", "findings": [], "ok": True}
            )
        return json.dumps(
            {"applied": bool(arguments.get("apply")), "summary": "C1 -> 100n"}
        )


@pytest.fixture
def api(fake_config, tmp_path):
    """An `Api` wired to a fake server and a temp session, with no network anywhere.

    Shared by the approval-gate tests and the write-conflict tests: both need exactly this
    object, and two copies of the setup drifted apart once already.
    """
    from spice_mcp_app.api import Api
    from spice_mcp_app.session import Session

    instance = Api(config=fake_config)
    instance._mcp = FakeMCP()
    instance._session = Session(model="fake-model", sessions_dir=tmp_path)
    return instance


def parse(text: str):
    """Parse netlist text directly, bypassing LTspice."""
    from spice_mcp_server.netlist import parse_netlist_text

    return parse_netlist_text(text, Path("test.net"), Path("test.net"))


def check_ids(text: str) -> set[str]:
    """Return the set of check ids that fire on the given netlist text."""
    from spice_mcp_server.checks import run_static_checks

    return {f.check for f in run_static_checks(parse(text)).findings}

"""Consistency tests between the web UI and the Python bridge.

Nothing here opens a window. These tests exist because the UI talks to Python through
strings - `window.pywebview.api.send_message(...)` and `getElementById("send")` - so a
renamed method or a removed element fails at runtime, in a webview, with no traceback
anywhere Python can see. Cross-referencing the two sides statically catches exactly that
class of mistake, which is otherwise only found by clicking.

What they deliberately do not prove: that the window renders correctly. That needs eyes
on it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from spice_mcp_app.api import Api

WEB = Path(__file__).resolve().parent.parent / "spice_mcp_app" / "web"
INDEX = WEB / "index.html"
APP_JS = WEB / "app.js"
STYLE = WEB / "style.css"


@pytest.fixture(scope="module")
def html() -> str:
    return INDEX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def js() -> str:
    return APP_JS.read_text(encoding="utf-8")


def test_web_assets_exist():
    for path in (INDEX, APP_JS, STYLE):
        assert path.is_file(), f"missing {path.name}"


def test_every_api_method_called_from_js_exists(js):
    called = set(re.findall(r"pywebview\.api\.([A-Za-z_][A-Za-z0-9_]*)", js))
    assert called, "the UI does not call the bridge at all"

    missing = sorted(name for name in called if not callable(getattr(Api, name, None)))
    assert not missing, f"app.js calls Api methods that do not exist: {missing}"


def test_every_dom_id_referenced_from_js_exists(html, js):
    referenced = set(re.findall(r"""getElementById\(["']([^"']+)["']\)""", js))
    referenced |= set(re.findall(r"""\$\(["']([^"']+)["']\)""", js))
    present = set(re.findall(r"""\bid=["']([^"']+)["']""", html))

    missing = sorted(referenced - present)
    assert not missing, f"app.js references ids that are not in index.html: {missing}"


def test_index_loads_its_assets(html):
    assert 'href="style.css"' in html
    assert 'src="app.js"' in html


def test_severity_classes_used_by_js_are_styled(js):
    """Static findings are colour-coded by severity; an unstyled class reads as normal."""
    css = STYLE.read_text(encoding="utf-8")
    for severity in ("error", "warning", "info"):
        assert f".sev-{severity}" in css, f"no style for .sev-{severity}"


def test_the_ui_never_asks_to_apply_a_patch_directly(js):
    """The UI must go through apply_patch, which records the approval.

    Calling the MCP tool with apply=true from the UI would bypass the gate in
    Api._tool_executor, which is the only thing standing between the model and the
    user's schematic.
    """
    assert "apply_patch" in js, "the Apply button must call apply_patch"
    assert "apply: true" not in js and 'apply":true' not in js


def test_api_exposes_everything_the_ui_needs():
    """Guard the other direction: the methods the UI depends on by contract."""
    for name in (
        "start",
        "shutdown",
        "pick_folder",
        "list_folder",
        "select_circuit",
        "send_message",
        "apply_patch",
        "resimulate",
        "mark_resolved",
        "export_session",
        "get_initial_folder",
    ):
        assert callable(getattr(Api, name, None)), f"Api.{name} is missing"

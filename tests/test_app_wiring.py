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


# --- the single-circuit launch path ----------------------------------------------------
#
# All of this is only reachable from Explorer, so nothing else in the suite would notice if
# one half of it were renamed. The launcher tests prove Python passes --file; these prove
# the JS side actually consumes what Python puts in the payload.


def test_the_parser_accepts_a_single_file():
    from spice_mcp_app.__main__ import build_parser

    args = build_parser().parse_args(["--file", r"C:\x\rc.asc"])
    assert args.file == r"C:\x\rc.asc"


def test_main_accepts_the_launcher_keywords():
    """launch.py calls main(argv, startup_note=..., opened_in_ltspice=...)."""
    import inspect

    from spice_mcp_app.__main__ import main

    parameters = inspect.signature(main).parameters
    for name in ("startup_note", "opened_in_ltspice"):
        assert name in parameters, f"main() no longer takes {name}"
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_sidebar_stores_each_path_where_js_can_find_it(js):
    """`li.title` is not usable as a key - for a shadowed .net it holds warning text."""
    assert "li.dataset.path" in js
    assert "dataset.path" in js.split("function findCircuitRow")[1], (
        "findCircuitRow must look up rows by dataset.path"
    )


def test_the_launch_payload_is_fully_consumed(js):
    """Every key Api.get_initial_folder returns has to be read, or it is dead weight."""
    for key in ("initial.folder", "initial.circuit", "initial.note"):
        assert key in js, f"app.js ignores {key} from the launch payload"


def test_selecting_a_circuit_tolerates_no_matching_row(js):
    """findCircuitRow returns null when the path is not in the sidebar."""
    assert "if (li) li.classList.add" in js, "selectCircuit would throw on a null row"


def test_the_write_conflict_warning_is_rendered_and_styled(js):
    """Api.select_circuit and Api.apply_patch both return `warning`; both must show it."""
    assert js.count("result.warning") >= 2, (
        "the LTspice-is-open warning is not rendered on both the select and apply paths"
    )
    css = STYLE.read_text(encoding="utf-8")
    assert ".sev-warning" in css

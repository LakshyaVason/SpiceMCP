"""Tests for the Explorer right-click registration.

**Nothing here writes to the registry.** The installer is split so that a pure function
decides what should exist and a thin winreg shell writes it; only the pure half is
exercised, and one test pins that importing the module cannot write anything even by
accident. A test suite that edited HKCU would be a genuinely nasty surprise to run.

The assertions are exact rather than "contains", following tests/test_checks.py: the point
is to catch a value quietly added to the plan without a matching uninstall, or a key path
that drifts towards claiming the .asc file type instead of adding a verb to it.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytest.importorskip("winreg", reason="the Explorer verb is Windows-only")

from scripts import install_context_menu as cm  # noqa: E402

FAKE_ROOT = Path(r"C:\fake\repo")


def test_the_registry_plan_is_exactly_these_four_values():
    assert cm.registry_plan(FAKE_ROOT) == (
        cm.RegistryValue(
            r"Software\Classes\SystemFileAssociations\.asc\shell\SpiceMCP.Debug",
            "",
            "Debug with SPICE MCP",
        ),
        cm.RegistryValue(
            r"Software\Classes\SystemFileAssociations\.asc\shell\SpiceMCP.Debug",
            "Icon",
            r"C:\fake\repo\.venv\Scripts\pythonw.exe,0",
        ),
        cm.RegistryValue(
            r"Software\Classes\SystemFileAssociations\.asc\shell\SpiceMCP.Debug",
            "NeverDefault",
            "",
        ),
        cm.RegistryValue(
            r"Software\Classes\SystemFileAssociations\.asc\shell\SpiceMCP.Debug\command",
            "",
            r'"C:\fake\repo\.venv\Scripts\pythonw.exe" "C:\fake\repo\spice_mcp_launch.py" "%1"',
        ),
    )


def test_the_file_argument_is_quoted():
    """Unquoted, a path with a space arrives as several arguments."""
    command = cm.launcher_command(FAKE_ROOT)

    assert command.endswith('"%1"')
    assert "%*" not in command, "%* would pass along whatever else Explorer felt like adding"
    # Three independently quoted tokens: interpreter, script, file.
    assert command.count('"') == 6


def test_it_uses_pythonw_so_no_console_flashes():
    assert cm.launcher_exe(FAKE_ROOT).name == "pythonw.exe"


def test_every_key_is_a_per_user_file_association_verb():
    """No admin rights, and no ProgID: this adds a verb, it does not claim the type."""
    prefix = "Software\\Classes\\SystemFileAssociations\\.asc\\shell\\"
    for value in cm.registry_plan(FAKE_ROOT):
        assert value.key.startswith(prefix)
        assert "HKEY_" not in value.key and "HKLM" not in value.key


def test_it_never_touches_the_asc_progid_or_the_open_verb():
    """The regression test for hijacking LTspice's own association."""
    for value in cm.registry_plan(FAKE_ROOT):
        assert value.key != r"Software\Classes\.asc"
        assert r"\shell\open" not in value.key


def test_the_verb_can_never_become_the_double_click_default():
    """Without NeverDefault, Explorer promotes a lone verb when there is no default."""
    names = {v.name for v in cm.registry_plan(FAKE_ROOT)}
    assert "NeverDefault" in names


def test_uninstall_covers_everything_install_creates():
    """Catches a key added to the plan without a matching deletion."""
    doomed = cm.keys_to_delete()
    for value in cm.registry_plan(FAKE_ROOT):
        assert any(
            value.key == root or value.key.startswith(root + "\\") for root in doomed
        ), f"{value.key} would survive an uninstall"


def test_status_reports_nothing_when_not_installed():
    report = "\n".join(cm.status_report(FAKE_ROOT, {}))
    assert "Not installed" in report


def test_status_flags_a_moved_repo():
    """The failure everyone hits after renaming the folder or rebuilding .venv."""
    installed = {
        (key, name): data for key, name, data in ((v.key, v.name, v.data) for v in cm.registry_plan(FAKE_ROOT))
    }
    installed[(cm.COMMAND_KEY, "")] = r'"C:\somewhere\else\pythonw.exe" "x.py" "%1"'

    report = "\n".join(cm.status_report(FAKE_ROOT, installed))
    assert "DIFFERS" in report
    assert "Re-run" in report or "re-run" in report


def test_status_is_clean_when_current():
    installed = {(v.key, v.name): v.data for v in cm.registry_plan(FAKE_ROOT)}
    assert "Installed and current." in cm.status_report(FAKE_ROOT, installed)


def test_importing_the_module_writes_nothing(monkeypatch):
    """Import and every pure call must be inert, even with winreg booby-trapped."""
    import winreg

    def explode(*args, **kwargs):
        raise AssertionError("the installer touched the registry on import")

    for name in ("CreateKeyEx", "SetValueEx", "DeleteKey", "OpenKey"):
        monkeypatch.setattr(winreg, name, explode)

    reloaded = importlib.reload(cm)
    reloaded.registry_plan(FAKE_ROOT)
    reloaded.launcher_command(FAKE_ROOT)
    reloaded.keys_to_delete()
    reloaded.status_report(FAKE_ROOT, {})


def test_the_launcher_it_points_at_actually_exists():
    """Catches a rename of spice_mcp_launch.py leaving a dead registry command."""
    assert cm.launcher_script(cm.REPO_ROOT).is_file()

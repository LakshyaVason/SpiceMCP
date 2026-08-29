"""Add or remove the "Debug with SPICE MCP" right-click entry for .asc files.

    python scripts\\install_context_menu.py              REM install
    python scripts\\install_context_menu.py --status      REM what is actually registered
    python scripts\\install_context_menu.py --uninstall   REM remove

Per-user, no administrator rights. Everything goes under
`HKCU\\Software\\Classes\\SystemFileAssociations\\.asc\\shell\\`, which *adds* a verb rather
than claiming the file type: it creates no ProgID, never touches `HKCU\\Software\\Classes\\.asc`,
and stays away from `shell\\open`, so LTspice's own association and what double-clicking a
schematic does are both left exactly as they were.

Verified against this machine's registry before choosing that location: `.asc` is owned by
the ProgID `Analog Devices Inc..LTspice_1`, and `SystemFileAssociations\\.asc` did not exist
in HKCU or HKCR at all. Hanging the verb off LTspice's own ProgID was the alternative and is
rejected because that name is version-suffixed - it would break the next time LTspice
re-registers itself, whereas the extension key is stable.

**Windows 11 caveat.** A verb registered this way appears in the *legacy* context menu:
right-click then "Show more options", or Shift+right-click. The new compact menu is
populated only by packaged (MSIX) apps implementing the IExplorerCommand COM interface,
which needs an app package and a signing certificate - a different project. There is no
registry setting that promotes a classic verb into the compact menu.

The pure decision - which keys and values should exist - is separated from the winreg calls
so the test suite can assert the exact registry plan without ever writing to the registry.
"""

from __future__ import annotations

import argparse
import sys
import winreg
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Namespaced so uninstall can delete exactly one subtree and nothing a neighbour owns.
VERB_KEY = r"Software\Classes\SystemFileAssociations\.asc\shell\SpiceMCP.Debug"
COMMAND_KEY = VERB_KEY + r"\command"
MENU_LABEL = "Debug with SPICE MCP"


@dataclass(frozen=True)
class RegistryValue:
    """One value to write. `key` is always relative to HKEY_CURRENT_USER."""

    key: str
    name: str  # "" means the key's default value
    data: str  # always REG_SZ


def launcher_exe(repo_root: Path) -> Path:
    """The interpreter to run. pythonw.exe so no console flashes on every click."""
    return repo_root / ".venv" / "Scripts" / "pythonw.exe"


def launcher_script(repo_root: Path) -> Path:
    return repo_root / "spice_mcp_launch.py"


def launcher_command(repo_root: Path) -> str:
    """The command string Explorer runs.

    Every one of the three tokens is quoted independently. `"%1"` in particular *must* be
    quoted: unquoted, `C:\\My Circuits\\rc.asc` arrives as three separate arguments. Not
    `%*`, which would append whatever else Explorer felt like passing.
    """
    return f'"{launcher_exe(repo_root)}" "{launcher_script(repo_root)}" "%1"'


def registry_plan(repo_root: Path) -> tuple[RegistryValue, ...]:
    """Exactly what install() will write. Pure, so it can be asserted in tests."""
    return (
        RegistryValue(VERB_KEY, "", MENU_LABEL),
        # The icon is the very binary the command runs, so it can never dangle. A real .ico
        # can replace this later by editing one line.
        RegistryValue(VERB_KEY, "Icon", f"{launcher_exe(repo_root)},0"),
        # Should .asc ever have no default handler - LTspice uninstalled, or its association
        # broken - Explorer would otherwise promote the only remaining verb to the
        # double-click default. This is the difference between adding a verb and quietly
        # taking over the file type.
        RegistryValue(VERB_KEY, "NeverDefault", ""),
        RegistryValue(COMMAND_KEY, "", launcher_command(repo_root)),
    )


def keys_to_delete() -> tuple[str, ...]:
    """Deleted depth-first, so `\\command` goes with its parent."""
    return (VERB_KEY,)


def status_report(repo_root: Path, installed: dict[tuple[str, str], str]) -> list[str]:
    """Compare what is registered against what should be. Pure.

    Being pure is what makes the interesting case testable: an entry that is present but
    points somewhere else, which is what everyone hits after moving or renaming the repo.
    """
    wanted = registry_plan(repo_root)
    if not installed:
        return ["Not installed. Run this script with no arguments to add the entry."]

    lines: list[str] = []
    stale = False
    for value in wanted:
        actual = installed.get((value.key, value.name))
        if actual is None:
            lines.append(f"MISSING  {value.key}\\{value.name or '(default)'}")
            stale = True
        elif actual != value.data:
            lines.append(f"DIFFERS  {value.key}\\{value.name or '(default)'}")
            lines.append(f"           registered: {actual}")
            lines.append(f"           expected:   {value.data}")
            stale = True
        else:
            lines.append(f"ok       {value.key}\\{value.name or '(default)'}")

    if stale:
        lines.append("")
        lines.append(
            "The entry does not match this repo - most likely the repo or its .venv moved. "
            "Re-run this script with no arguments to point it here again."
        )
    else:
        lines.append("")
        lines.append("Installed and current.")
    return lines


# --- the thin winreg shell ------------------------------------------------------------


def install(repo_root: Path) -> list[str]:
    """Write the plan. Returns a line per value written, for the caller to print."""
    written: list[str] = []
    for value in registry_plan(repo_root):
        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER, value.key, 0, winreg.KEY_WRITE
        ) as key:
            winreg.SetValueEx(key, value.name, 0, winreg.REG_SZ, value.data)
        written.append(
            f"HKCU\\{value.key}\\{value.name or '(default)'} = {value.data or '(empty)'}"
        )
    _notify_shell()
    return written


def read_installed() -> dict[tuple[str, str], str]:
    """What is actually in the registry right now, in registry_plan's shape."""
    found: dict[tuple[str, str], str] = {}
    for key_path in (VERB_KEY, COMMAND_KEY):
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                index = 0
                while True:
                    try:
                        name, data, _kind = winreg.EnumValue(key, index)
                    except OSError:
                        break
                    found[(key_path, name)] = str(data)
                    index += 1
        except FileNotFoundError:
            continue
    return found


def _delete_tree(key_path: str) -> list[str]:
    """Delete a key and its subkeys. DeleteKey refuses a key that still has children.

    Depth is only 1 today, but recursing is three lines and survives someone adding an
    ExtendedSubCommandsKey later.
    """
    removed: list[str] = []
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            children = []
            index = 0
            while True:
                try:
                    children.append(winreg.EnumKey(key, index))
                except OSError:
                    break
                index += 1
    except FileNotFoundError:
        return removed

    for child in children:
        removed += _delete_tree(f"{key_path}\\{child}")
    winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key_path)
    removed.append(f"HKCU\\{key_path}")
    return removed


def uninstall() -> list[str]:
    removed: list[str] = []
    for key_path in keys_to_delete():
        removed += _delete_tree(key_path)

    # The two keys above ours are shared ground: `shell` and `.asc` may hold other people's
    # verbs. Remove them only when we left them empty.
    for parent in (
        r"Software\Classes\SystemFileAssociations\.asc\shell",
        r"Software\Classes\SystemFileAssociations\.asc",
    ):
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, parent) as key:
                subkeys, values, _ = winreg.QueryInfoKey(key)
            if subkeys or values:
                break
        except FileNotFoundError:
            continue
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, parent)
        removed.append(f"HKCU\\{parent} (was empty)")

    _notify_shell()
    return removed


def _notify_shell() -> None:
    """Tell Explorer associations changed. Usually unnecessary; flushes the icon cache."""
    try:
        import ctypes

        ctypes.windll.shell32.SHChangeNotify(0x08000000, 0, None, None)  # SHCNE_ASSOCCHANGED
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="install_context_menu",
        description='Add or remove the "Debug with SPICE MCP" right-click entry for .asc files.',
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--uninstall", action="store_true", help="Remove the entry.")
    group.add_argument(
        "--status", action="store_true", help="Report what is registered, and change nothing."
    )
    args = parser.parse_args(argv)

    if sys.platform != "win32":
        print("This registers a Windows Explorer verb; there is nothing to do here.")
        return 1

    if args.status:
        for line in status_report(REPO_ROOT, read_installed()):
            print(line)
        return 0

    if args.uninstall:
        removed = uninstall()
        if not removed:
            print("Nothing to remove - the entry was not installed.")
        for line in removed:
            print(f"removed  {line}")
        print('\nRight-click a .asc: "Debug with SPICE MCP" should be gone.')
        return 0

    exe = launcher_exe(REPO_ROOT)
    script = launcher_script(REPO_ROOT)
    if not script.is_file():
        print(f"Cannot find the launcher at {script}", file=sys.stderr)
        return 1
    if not exe.is_file():
        print(
            f"Cannot find {exe}.\n"
            "Create the venv first (see README Setup). Without pythonw.exe the entry would "
            "flash a console window on every click.",
            file=sys.stderr,
        )
        return 1

    for line in install(REPO_ROOT):
        print(f"wrote  {line}")
    print(
        f'\nRight-click a .asc file and choose "{MENU_LABEL}".\n'
        "On Windows 11 that means right-click then \"Show more options\" (or "
        "Shift+right-click) - the new compact menu only lists packaged apps.\n\n"
        f"The absolute paths above are baked in, so if you move or rename {REPO_ROOT.name} "
        "or rebuild .venv, re-run this script. `--status` will tell you if it has gone stale."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

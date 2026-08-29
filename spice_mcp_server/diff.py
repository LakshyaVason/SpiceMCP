"""Structured before/after comparison of two circuits.

A unified text diff alone is a poor review artifact for a circuit: reordered netlist
lines look like changes, and a changed net name looks identical to a rewired pin. So
components are matched by reference designator and compared field by field, and the text
diff is carried alongside as supporting evidence rather than as the answer.

Connectivity is compared as well as values, because the two failure modes are different:
a wrong value is out of spec, a wrong net is a different circuit.
"""

from __future__ import annotations

import difflib
from pathlib import Path

from .models import ComponentChange, Netlist, NetlistDiff


def _describe(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def diff_netlists(before: Netlist, after: Netlist) -> NetlistDiff:
    """Compare two parsed circuits by reference designator."""
    before_by_ref = {c.ref: c for c in before.components}
    after_by_ref = {c.ref: c for c in after.components}

    changes: list[ComponentChange] = []

    for ref in sorted(before_by_ref.keys() - after_by_ref.keys()):
        component = before_by_ref[ref]
        changes.append(
            ComponentChange(
                ref=ref,
                change="removed",
                before_value=component.value,
                before_nodes=list(component.nodes),
                detail=f"{ref} ({component.kind}) is no longer present.",
            )
        )

    for ref in sorted(after_by_ref.keys() - before_by_ref.keys()):
        component = after_by_ref[ref]
        changes.append(
            ComponentChange(
                ref=ref,
                change="added",
                after_value=component.value,
                after_nodes=list(component.nodes),
                detail=f"{ref} ({component.kind}) was added, value {component.value!r}.",
            )
        )

    for ref in sorted(before_by_ref.keys() & after_by_ref.keys()):
        old = before_by_ref[ref]
        new = after_by_ref[ref]

        if old.nodes != new.nodes:
            changes.append(
                ComponentChange(
                    ref=ref,
                    change="nodes_changed",
                    before_value=old.value,
                    after_value=new.value,
                    before_nodes=list(old.nodes),
                    after_nodes=list(new.nodes),
                    detail=(
                        f"{ref} was rewired: {' '.join(old.nodes)} -> "
                        f"{' '.join(new.nodes)}."
                    ),
                )
            )
        # Reported separately from rewiring: a value change is an in-spec adjustment,
        # a rewire is a different circuit. Collapsing them would hide that.
        if old.value != new.value:
            changes.append(
                ComponentChange(
                    ref=ref,
                    change="value_changed",
                    before_value=old.value,
                    after_value=new.value,
                    before_nodes=list(old.nodes),
                    after_nodes=list(new.nodes),
                    detail=f"{ref} value changed: {old.value!r} -> {new.value!r}.",
                )
            )

    before_nets = {n.name for n in before.nets}
    after_nets = {n.name for n in after.nets}
    nets_added = sorted(after_nets - before_nets)
    nets_removed = sorted(before_nets - after_nets)

    before_directives = [d.text.strip() for d in before.directives]
    after_directives = [d.text.strip() for d in after.directives]
    directives_added = sorted(set(after_directives) - set(before_directives))
    directives_removed = sorted(set(before_directives) - set(after_directives))

    text_diff = "".join(
        difflib.unified_diff(
            before.raw_text.splitlines(keepends=True),
            after.raw_text.splitlines(keepends=True),
            fromfile=f"{Path(before.source_path).name} (before)",
            tofile=f"{Path(after.source_path).name} (after)",
            n=3,
        )
    )

    identical = not (
        changes or nets_added or nets_removed or directives_added or directives_removed
    )

    if identical:
        summary = "The two circuits are electrically identical."
    else:
        parts: list[str] = []
        value_changes = sum(1 for c in changes if c.change == "value_changed")
        rewires = sum(1 for c in changes if c.change == "nodes_changed")
        added = sum(1 for c in changes if c.change == "added")
        removed = sum(1 for c in changes if c.change == "removed")
        if value_changes:
            parts.append(_describe(value_changes, "value change"))
        if rewires:
            parts.append(_describe(rewires, "rewired component"))
        if added:
            parts.append(_describe(added, "added component"))
        if removed:
            parts.append(_describe(removed, "removed component"))
        if directives_added or directives_removed:
            parts.append(
                _describe(len(directives_added) + len(directives_removed), "directive change")
            )
        summary = ", ".join(parts).capitalize() + "."

    return NetlistDiff(
        before_path=before.source_path,
        after_path=after.source_path,
        identical=identical,
        component_changes=changes,
        nets_added=nets_added,
        nets_removed=nets_removed,
        directives_added=directives_added,
        directives_removed=directives_removed,
        text_diff=text_diff,
        summary=summary,
    )

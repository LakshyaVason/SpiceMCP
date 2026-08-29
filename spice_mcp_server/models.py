"""Pydantic return models for the MCP tools.

In the mcp v2 SDK the return type annotation *is* the output schema, so these classes
are the contract the LLM sees. Every field is annotated on the class body on purpose:
a class whose attributes are only assigned in __init__ produces no schema and fails
silently, showing the model a bare repr instead of structured data.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Component(BaseModel):
    """One circuit element as it appears in the flattened SPICE netlist."""

    ref: str = Field(description="Reference designator, e.g. 'R1'.")
    prefix: str = Field(description="SPICE element letter, e.g. 'R' for a resistor.")
    kind: str = Field(description="Human-readable element type, e.g. 'resistor'.")
    nodes: list[str] = Field(description="Net names this element connects to, in pin order.")
    value: str | None = Field(
        default=None,
        description="Value/model expression as written, e.g. '10k' or 'AC 0.7 3000'. "
        "None when the element has no value field.",
    )
    line_no: int = Field(description="1-indexed line number in the netlist.")
    raw_line: str = Field(description="The netlist line verbatim.")
    nodes_uncertain: bool = Field(
        default=False,
        description="True when the element's pin count could not be determined reliably "
        "(LTspice 'A' special-function devices, for instance). Connectivity checks are "
        "downgraded when any such element is present, to avoid false floating-net reports.",
    )


class Directive(BaseModel):
    """A SPICE dot-directive such as .ac, .tran, .model or .param."""

    kind: str = Field(description="The directive keyword including the dot, e.g. '.ac'.")
    text: str = Field(description="Full directive line verbatim.")
    line_no: int = Field(description="1-indexed line number in the netlist.")


class Subcircuit(BaseModel):
    """A .subckt definition found in the netlist."""

    name: str = Field(description="Subcircuit name.")
    ports: list[str] = Field(description="Port net names in declaration order.")
    line_no: int = Field(description="1-indexed line of the .subckt line.")


class Net(BaseModel):
    """A node in the circuit and what connects to it.

    `connection_count` is the number of element pins attached. A count of 1 means a
    floating pin, which is the single most common LTspice drafting mistake.
    """

    name: str = Field(description="Net name. '0' is ground.")
    connection_count: int = Field(description="Number of element pins attached to this net.")
    connected_refs: list[str] = Field(description="Reference designators touching this net.")


class Netlist(BaseModel):
    """Structured view of a circuit. This is what replaces a screenshot."""

    source_path: str = Field(description="File the caller asked about.")
    netlist_path: str = Field(
        description="Path of the SPICE netlist actually parsed. Differs from source_path "
        "when a .asc schematic was converted via LTspice -netlist."
    )
    title: str | None = Field(default=None, description="Netlist title/comment line, if any.")
    components: list[Component] = Field(description="All circuit elements.")
    directives: list[Directive] = Field(description="All dot-directives.")
    subcircuits: list[Subcircuit] = Field(description="All .subckt definitions.")
    nets: list[Net] = Field(description="All nets with their connection counts.")
    raw_text: str = Field(description="The full SPICE netlist text.")


class Finding(BaseModel):
    """One problem detected by a static check or log parse."""

    check: str = Field(description="Stable machine-readable check id, e.g. 'floating_net'.")
    severity: str = Field(description="One of: error, warning, info.")
    message: str = Field(description="What is wrong, in plain English.")
    refs: list[str] = Field(default_factory=list, description="Components involved.")
    nets: list[str] = Field(default_factory=list, description="Nets involved.")
    line_no: int | None = Field(default=None, description="Netlist line, when known.")
    suggestion: str | None = Field(
        default=None, description="Concrete suggested fix, when one can be inferred."
    )


class StaticCheckResult(BaseModel):
    """Result of check_netlist_static. No simulation was run."""

    source_path: str = Field(description="File that was checked.")
    ok: bool = Field(description="True when no error-severity findings were produced.")
    findings: list[Finding] = Field(description="Problems found, most severe first.")
    summary: str = Field(description="One-line summary suitable for showing the user.")


class Measurement(BaseModel):
    """One .measure result read out of the simulation log."""

    name: str = Field(description="Measurement name as given in the .measure directive.")
    value: str = Field(
        description="Measured value as reported. Kept as text because LTspice reports AC "
        "measurements in a complex '(xdB,y deg)' form that is not a single number."
    )


class SimLog(BaseModel):
    """Parsed LTspice simulation log.

    This is usually where the real diagnosis lives. Read `succeeded` rather than any
    exit code: LTspice exits 0 for several genuinely failed runs, and writes a .raw
    file even when the analysis failed.
    """

    log_path: str = Field(description="Log file that was parsed.")
    ltspice_version: str | None = Field(
        default=None, description="Version banner, e.g. 'LTspice 26.0.2 for Windows'."
    )
    circuit: str | None = Field(default=None, description="Circuit path named in the log.")
    succeeded: bool = Field(
        description="True when LTspice completed an analysis with no fatal error. This is "
        "derived from the log text, never from the process exit code."
    )
    elapsed_s: float | None = Field(
        default=None, description="Reported total elapsed simulation time in seconds."
    )
    solver: str | None = Field(default=None, description="Solver mode, e.g. 'Normal'.")
    method: str | None = Field(default=None, description="Integration method, e.g. 'trap'.")
    findings: list[Finding] = Field(
        description="Errors and warnings from the log, most severe first."
    )
    measurements: list[Measurement] = Field(
        default_factory=list, description=".measure results, when the deck had any."
    )
    step_count: int = Field(
        default=0, description="Number of .step iterations, 0 when the deck was not stepped."
    )
    files_loaded: list[str] = Field(
        default_factory=list,
        description="Files LTspice loaded. Useful for debugging .include and .lib paths.",
    )
    summary: str = Field(description="One-line summary suitable for showing the user.")
    raw_text: str = Field(description="The full log text.")


class PatchResult(BaseModel):
    """Result of patch_component_value.

    With `applied` False nothing was written and this is a proposal for review. The
    `diff` field is the reviewable artifact; the caller is expected to show it before
    asking for the same call again with apply=True.
    """

    asc_path: str = Field(description="Schematic the edit targets.")
    ref: str = Field(description="Component whose value was changed.")
    old_value: str | None = Field(
        default=None,
        description="Value before the edit. None when the component had no value line, "
        "in which case one was inserted.",
    )
    new_value: str = Field(description="Value after the edit.")
    line_no: int = Field(description="1-indexed line of the SYMATTR Value line affected.")
    inserted: bool = Field(
        description="True when the component had no SYMATTR Value line and one was added."
    )
    applied: bool = Field(
        description="True when the file was written. False means this is a preview only "
        "and the schematic on disk is unchanged."
    )
    diff: str = Field(description="Unified diff of the schematic, for the user to review.")
    before_line: str | None = Field(
        default=None, description="The original line, verbatim. None when inserting."
    )
    after_line: str = Field(description="The replacement line, verbatim.")
    encoding: str = Field(
        description="Encoding the file was read and written with. Reported so the caller "
        "can confirm the schematic's bytes were preserved rather than normalised."
    )
    summary: str = Field(description="One-line summary suitable for showing the user.")


class ComponentChange(BaseModel):
    """One component-level difference between two circuits."""

    ref: str = Field(description="Reference designator.")
    change: str = Field(description="One of: added, removed, value_changed, nodes_changed.")
    before_value: str | None = Field(default=None, description="Value in the first circuit.")
    after_value: str | None = Field(default=None, description="Value in the second circuit.")
    before_nodes: list[str] = Field(
        default_factory=list, description="Nets in the first circuit."
    )
    after_nodes: list[str] = Field(
        default_factory=list, description="Nets in the second circuit."
    )
    detail: str = Field(description="The change in plain English.")


class NetlistDiff(BaseModel):
    """Structured comparison of two circuits: values and connectivity."""

    before_path: str = Field(description="First circuit.")
    after_path: str = Field(description="Second circuit.")
    identical: bool = Field(description="True when no component or net differences were found.")
    component_changes: list[ComponentChange] = Field(
        description="Per-component differences, values and connectivity."
    )
    nets_added: list[str] = Field(description="Nets present only in the second circuit.")
    nets_removed: list[str] = Field(description="Nets present only in the first circuit.")
    directives_added: list[str] = Field(description="Directives only in the second circuit.")
    directives_removed: list[str] = Field(description="Directives only in the first circuit.")
    text_diff: str = Field(description="Unified diff of the two flattened SPICE netlists.")
    summary: str = Field(description="One-line summary suitable for showing the user.")


class ExportResult(BaseModel):
    """Result of export_netlist."""

    source_path: str = Field(description="Circuit that was exported.")
    out_path: str = Field(description="File written.")
    line_count: int = Field(description="Lines written.")
    byte_count: int = Field(description="Bytes written.")
    summary: str = Field(description="One-line summary suitable for showing the user.")


class SimulationResult(BaseModel):
    """Result of run_simulation: what happened, plus the parsed log."""

    circuit_path: str = Field(description="Circuit the caller asked to simulate.")
    simulated_path: str = Field(
        description="File actually handed to LTspice. Differs from circuit_path because "
        "inputs are staged into a scratch directory so the source folder is never written to."
    )
    succeeded: bool = Field(
        description="True when the analysis completed without a fatal error, per the log."
    )
    timed_out: bool = Field(description="True when the run exceeded timeout_s and was killed.")
    returncode: int = Field(
        description="LTspice exit code, for diagnostics only. Do not infer success from it."
    )
    log_path: str | None = Field(default=None, description="Path of the .log file produced.")
    raw_path: str | None = Field(
        default=None,
        description="Path of the .raw waveform file. Present even for some failed runs, so "
        "its existence does not imply success.",
    )
    log: SimLog | None = Field(
        default=None, description="Parsed log. None when LTspice produced no log at all."
    )
    summary: str = Field(description="One-line summary suitable for showing the user.")

"""Session log writer: one JSON file per debug session, with per-turn token counts.

The schema is fixed by the brief and an **external** cost comparison depends on it, so
it is not ours to improve:

    {
      "session_id": "<uuid4>",
      "started_at": "<ISO 8601>",
      "model": "<model id>",
      "circuit_file": "<path or null>",
      "turns": [
        {"role": "user"|"assistant"|"tool", "text": "...",
         "tool_calls": [...]        # optional
         "input_tokens": 123, "output_tokens": 45}
      ],
      "total_input_tokens": 0,
      "total_output_tokens": 0,
      "resolved": false
    }

Two things to keep in mind:

  * Bedrock returns `usage.input_tokens` / `usage.output_tokens` - already the schema's
    own names, so `Turn.from_usage` no longer renames anything. What it still does, and
    must keep doing, is **warn when an assistant turn arrives without usage**: silently
    null counts once let a whole session log look complete while under-reporting
    everything, and that is the failure mode this log exists to rule out.
  * `usage` also carries `cache_read_input_tokens` / `cache_creation_input_tokens` on
    Bedrock. Prompt caching is not enabled, so those are ignored - the schema above is
    fixed by an external comparison and must not gain keys. If caching is ever turned on,
    the comparison needs revisiting before this log does.
  * The file is rewritten after every turn. A session that crashes mid-debug still
    leaves a usable log, which matters because a crashed session is exactly the kind
    we want token numbers for.

This module records. It does not analyse - no cost arithmetic lives here.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def usage_value(usage: Any, name: str) -> Any:
    """Read a field from a usage block, tolerating both an SDK object and a plain dict.

    The SDK hands back a pydantic `Usage`, but tests and the agent loop both pass dicts.
    Reading through here rather than converting to a dict up front is what keeps a genuinely
    absent field distinguishable from a zero, so the warning below can still fire.
    """
    if usage is None:
        return None
    if isinstance(usage, Mapping):
        return usage.get(name)
    return getattr(usage, name, None)


@dataclass
class Turn:
    role: str
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list[dict[str, Any]] | None = None

    @classmethod
    def from_usage(
        cls,
        role: str,
        text: str,
        usage: Any,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> Turn:
        """Build a turn from a provider `usage` block - an object or a dict.

        A missing count logs a warning rather than passing silently: null token counts
        would quietly invalidate the cost comparison, which is the whole point of
        keeping this log.
        """
        prompt = usage_value(usage, "input_tokens")
        completion = usage_value(usage, "output_tokens")
        if role == "assistant" and (prompt is None or completion is None):
            log.warning(
                "assistant turn has incomplete usage (input=%s output=%s); "
                "the session log will under-report tokens",
                prompt,
                completion,
            )
        return cls(
            role=role,
            text=text,
            input_tokens=int(prompt or 0),
            output_tokens=int(completion or 0),
            tool_calls=tool_calls or None,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "text": self.text}
        # Optional per the schema, so only present when there is something to say.
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        payload["input_tokens"] = self.input_tokens
        payload["output_tokens"] = self.output_tokens
        return payload


@dataclass
class Session:
    model: str
    sessions_dir: Path
    circuit_file: str | None = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=_utc_now)
    turns: list[Turn] = field(default_factory=list)
    resolved: bool = False

    @property
    def path(self) -> Path:
        return self.sessions_dir / f"{self.session_id}.json"

    @property
    def total_input_tokens(self) -> int:
        return sum(t.input_tokens for t in self.turns)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.turns)

    def add_turn(self, turn: Turn) -> Turn:
        self.turns.append(turn)
        self.save()
        return turn

    def set_circuit_file(self, path: str | os.PathLike[str] | None) -> None:
        self.circuit_file = str(path) if path else None
        self.save()

    def mark_resolved(self, resolved: bool = True) -> None:
        self.resolved = resolved
        self.save()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "model": self.model,
            "circuit_file": self.circuit_file,
            "turns": [t.to_dict() for t in self.turns],
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "resolved": self.resolved,
        }

    def save(self) -> Path:
        """Write the log atomically.

        Written to a temp file in the same directory and then replaced, so a crash
        during the write cannot leave a half-written log where a complete one used to
        be. os.replace is atomic on Windows when both paths share a volume.
        """
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

        handle, tmp_name = tempfile.mkstemp(
            dir=self.sessions_dir, prefix=f".{self.session_id}.", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(payload + "\n")
            os.replace(tmp_name, self.path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return self.path

    def export_to(self, destination: str | os.PathLike[str]) -> Path:
        """Copy the log to a user-chosen location for the external comparison."""
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return target

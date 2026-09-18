"""Topology cache — avoids re-running the LTspice subprocess on re-select.

**IMPORTANT**: This cache reduces latency only. It does NOT reduce billed input
tokens. The topology string is part of the selection note that is injected into
conversation history, which is re-sent to the API on every request regardless of
whether it came from cache or from a fresh MCP call.

Cache key: (path, mtime, size). All three must match. A file that has been written
(by patch_component_value or by the user in LTspice) will have a new mtime/size and
will miss the cache, forcing a fresh read — which is correct behaviour.

The cache is per-Api-instance, in-memory, and discarded on shutdown. It is not
persisted across sessions, not shared across processes, and not bounded in size
(circuits are small; a session will touch at most a handful of distinct files).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    topology: str   # output of circuit_summary(), NOT raw JSON
    mtime: float
    size: int


class TopologyCache:
    """In-memory cache keyed on (path, mtime, size)."""

    def __init__(self) -> None:
        self._store: dict[str, _CacheEntry] = {}

    def get(self, path: str) -> str | None:
        """Return cached topology if path, mtime and size all match; else None."""
        entry = self._store.get(path)
        if entry is None:
            return None
        try:
            st = os.stat(path)
        except OSError as exc:
            log.debug("cache: stat failed for %s, evicting: %s", path, exc)
            del self._store[path]
            return None
        if st.st_mtime != entry.mtime or st.st_size != entry.size:
            log.debug("cache: stale for %s (mtime or size changed)", path)
            del self._store[path]
            return None
        log.debug("cache: hit for %s", path)
        return entry.topology

    def put(self, path: str, topology: str) -> None:
        """Store topology for path; silently skips if stat fails."""
        try:
            st = os.stat(path)
        except OSError as exc:
            log.debug("cache: put skipped for %s: %s", path, exc)
            return
        self._store[path] = _CacheEntry(
            topology=topology, mtime=st.st_mtime, size=st.st_size
        )
        log.debug("cache: stored for %s", path)

    def invalidate(self, path: str) -> None:
        """Remove the entry for path, if any. Idempotent."""
        self._store.pop(path, None)
        log.debug("cache: invalidated %s", path)

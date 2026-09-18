"""Tests for cache.py — the topology cache."""

from __future__ import annotations

import os
import time

import pytest

from spice_mcp_app.cache import TopologyCache


# --- helpers --------------------------------------------------------------------------

@pytest.fixture
def cache():
    return TopologyCache()


@pytest.fixture
def circuit_file(tmp_path):
    f = tmp_path / "circuit.asc"
    f.write_text("Version 4.1\n", encoding="utf-8")
    return f


# --- miss on empty cache -------------------------------------------------------------


def test_miss_on_empty_cache(cache, circuit_file):
    assert cache.get(str(circuit_file)) is None


def test_miss_for_unknown_path(cache):
    assert cache.get("/nonexistent/circuit.asc") is None


# --- hit after put -------------------------------------------------------------------


def test_hit_after_put_same_mtime_size(cache, circuit_file):
    cache.put(str(circuit_file), "R1, C1, V1")
    result = cache.get(str(circuit_file))
    assert result == "R1, C1, V1"


# --- miss after file changes ---------------------------------------------------------


def test_miss_after_size_changes(cache, circuit_file):
    cache.put(str(circuit_file), "R1, C1, V1")
    # Write more content → size changes.
    circuit_file.write_text("Version 4.1\nmore content\n", encoding="utf-8")
    assert cache.get(str(circuit_file)) is None


def test_miss_after_mtime_advances(cache, circuit_file, monkeypatch):
    cache.put(str(circuit_file), "R1, C1, V1")

    # Simulate mtime advancing without changing size.
    original_stat = os.stat(str(circuit_file))
    future_mtime = original_stat.st_mtime + 1.0

    original_os_stat = os.stat

    def fake_stat(path, *args, **kwargs):
        st = original_os_stat(path, *args, **kwargs)
        if os.path.abspath(path) == str(circuit_file.resolve()):
            # Return a stat_result with advanced mtime but same size.
            import stat as stat_module
            return os.stat_result((
                st.st_mode, st.st_ino, st.st_dev, st.st_nlink,
                st.st_uid, st.st_gid, st.st_size,
                st.st_atime, future_mtime, st.st_ctime,
            ))
        return st

    monkeypatch.setattr(os, "stat", fake_stat)
    assert cache.get(str(circuit_file)) is None


# --- invalidate ----------------------------------------------------------------------


def test_invalidate_removes_entry(cache, circuit_file):
    cache.put(str(circuit_file), "R1, C1, V1")
    cache.invalidate(str(circuit_file))
    assert cache.get(str(circuit_file)) is None


def test_invalidate_is_idempotent(cache, circuit_file):
    # Invalidating a path that was never put should not raise.
    cache.invalidate(str(circuit_file))
    cache.invalidate(str(circuit_file))


def test_invalidate_does_not_affect_other_paths(cache, tmp_path):
    a = tmp_path / "a.asc"
    b = tmp_path / "b.asc"
    a.write_text("Version 4.1\n", encoding="utf-8")
    b.write_text("Version 4.1\n", encoding="utf-8")

    cache.put(str(a), "topology-a")
    cache.put(str(b), "topology-b")
    cache.invalidate(str(a))

    assert cache.get(str(a)) is None
    assert cache.get(str(b)) == "topology-b"


# --- stat failure handling -----------------------------------------------------------


def test_put_ignores_stat_failure(cache, monkeypatch):
    """A stat failure during put silently skips the cache entry."""
    def fail_stat(path, *args, **kwargs):
        raise OSError("disk error")

    monkeypatch.setattr(os, "stat", fail_stat)
    cache.put("/some/file.asc", "R1, C1")
    # No exception raised, and the entry was not stored.
    # Since stat fails on get too, we can't check get directly, but the put shouldn't crash.


def test_get_evicts_on_stat_failure(cache, circuit_file, monkeypatch):
    """A stat failure during get evicts the stale entry."""
    # First put successfully (stat works now).
    cache.put(str(circuit_file), "R1, C1, V1")

    # Make stat fail on get.
    def fail_stat(path, *args, **kwargs):
        raise OSError("disk error")

    monkeypatch.setattr(os, "stat", fail_stat)
    result = cache.get(str(circuit_file))
    assert result is None


# --- multiple entries ----------------------------------------------------------------


def test_multiple_circuits_are_independent(cache, tmp_path):
    files = []
    for i in range(3):
        f = tmp_path / f"circuit{i}.asc"
        f.write_text(f"Version {i}\n", encoding="utf-8")
        files.append(f)
        cache.put(str(f), f"topology-{i}")

    for i, f in enumerate(files):
        assert cache.get(str(f)) == f"topology-{i}"

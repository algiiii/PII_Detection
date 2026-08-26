"""Tests for the file stamp and the re-scan decision (B5, incremental scan).

The predicate is pure, so the interesting cases are stated directly on stamps
rather than through a folder scan.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from pii_detection.registry.freshness import (
    FileStamp,
    detector_signature,
    needs_rescan,
    stamp_for,
)

_WHEN = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)


def _stamp(*, shift_seconds: float = 0, size: int = 100) -> FileStamp:
    return FileStamp(modified_at=_WHEN + timedelta(seconds=shift_seconds), size=size)


def test_never_seen_must_be_scanned() -> None:
    assert needs_rescan(_stamp(), None) is True


def test_identical_stamp_is_skipped() -> None:
    assert needs_rescan(_stamp(), _stamp()) is False


def test_different_size_is_a_change() -> None:
    assert needs_rescan(_stamp(size=101), _stamp(size=100)) is True


def test_an_older_file_counts_as_changed() -> None:
    # Difference, not recency: a file restored to a previous version has changed
    # just as much as one that was edited, and a "newer than" test would skip it
    # forever.
    assert needs_rescan(_stamp(shift_seconds=-3600), _stamp()) is True


def test_sub_second_drift_is_tolerated() -> None:
    # File systems disagree on timestamp granularity (bind mounts, FAT, container
    # layers); without tolerance every file would look modified.
    assert needs_rescan(_stamp(shift_seconds=0.4), _stamp()) is False


def test_drift_beyond_tolerance_is_a_change() -> None:
    assert needs_rescan(_stamp(shift_seconds=5), _stamp()) is True


def test_changed_engine_forces_a_rescan() -> None:
    assert (
        needs_rescan(_stamp(), _stamp(), signature="new", recorded_signature="old")
        is True
    )


def test_same_engine_keeps_the_skip() -> None:
    assert (
        needs_rescan(_stamp(), _stamp(), signature="same", recorded_signature="same")
        is False
    )


def test_stamp_for_reads_size_and_mtime(tmp_path: Path) -> None:
    path = tmp_path / "f.txt"
    path.write_text("ciao", encoding="utf-8")
    stamp = stamp_for(path)
    assert stamp.size == 4
    assert stamp.modified_at.tzinfo is not None


def test_signature_changes_with_detectors_and_config(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.mkdir()
    (config / "custom_patterns.yaml").write_text("rules: []\n", encoding="utf-8")

    base = detector_signature(["a", "b"], config_dir=config)
    assert base == detector_signature(["b", "a"], config_dir=config)  # order-free
    assert base != detector_signature(["a", "c"], config_dir=config)

    (config / "custom_patterns.yaml").write_text("rules: [x]\n", encoding="utf-8")
    assert base != detector_signature(["a", "b"], config_dir=config)

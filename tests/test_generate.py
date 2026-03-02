"""Smoke tests for SecureRelayFL data generation."""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent.parent


def test_generator_imports():
    """Generator module imports without error."""
    sys.path.insert(0, str(ROOT))
    from data.generator import generate  # noqa: F401


def test_generate_small_batch(tmp_path):
    """Generate 10 samples per facility and verify shapes / labels."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "data" / "generator" / "generate.py"),
            "--n-samples", "10",
            "--seed", "123",
            "--output-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Generator failed:\n{result.stderr}"

    for fid in range(5):
        fdir = tmp_path / f"facility_{fid}"
        assert fdir.exists(), f"Missing facility dir: {fdir}"

        wf = np.load(fdir / "waveforms.npy")
        ft = np.load(fdir / "fault_type.npy")
        fz = np.load(fdir / "fault_zone.npy")
        pa = np.load(fdir / "protection_action.npy")

        # shape checks
        assert wf.shape == (10, 6, 2560), f"Bad waveform shape: {wf.shape}"
        assert ft.shape == (10,)
        assert fz.shape == (10,)
        assert pa.shape == (10,)

        # label range checks
        assert set(ft).issubset({0, 1, 2, 3})
        assert set(fz).issubset({-1, 0, 1, 2})
        assert set(pa).issubset({0, 1, 2, 3, 4})

        # waveforms should not be all zeros or NaN
        assert not np.any(np.isnan(wf)), "NaN in waveforms"
        assert np.std(wf) > 0, "Waveforms are flat"


def test_non_iid_across_facilities(tmp_path):
    """Verify facilities produce different fault distributions (non-IID)."""
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "data" / "generator" / "generate.py"),
            "--n-samples", "200",
            "--seed", "42",
            "--output-dir", str(tmp_path),
        ],
        capture_output=True,
    )
    distributions = []
    for fid in range(5):
        ft = np.load(tmp_path / f"facility_{fid}" / "fault_type.npy")
        hist = np.bincount(ft.astype(int), minlength=4) / len(ft)
        distributions.append(hist)

    # At least two facilities should have different dominant fault types
    dominant = [np.argmax(d[1:]) + 1 for d in distributions]  # exclude no-fault
    assert len(set(dominant)) >= 2, (
        f"Facilities not sufficiently non-IID: dominant fault types = {dominant}"
    )

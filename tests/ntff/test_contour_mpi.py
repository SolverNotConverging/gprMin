"""Actual MPI contour partitioning and hash-command integration tests."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest


def launcher():
    for candidate in (Path(sys.executable).parent / "mpiexec", Path(sys.executable).parent / "Library/bin/mpiexec.exe"):
        if candidate.is_file():
            return str(candidate)
    return shutil.which("mpiexec")


@pytest.mark.integration
@pytest.mark.parametrize("family", ("TE", "TM"))
@pytest.mark.parametrize("axis", range(3))
@pytest.mark.skipif(launcher() is None, reason="MPI launcher unavailable")
def test_mpi_contour_matches_serial(tmp_path, family, axis):
    domain, source, lower, upper = ["0.16"] * 3, ["0.08"] * 3, ["0.04"] * 3, ["0.12"] * 3
    for point in (domain, source, lower, upper):
        point[axis] = "inf"
    pml = ["10"] * 6
    pml[axis] = pml[axis + 3] = "0"
    source_command = "hertzian_dipole" if family == "TM" else "magnetic_dipole"
    model = tmp_path / "contour.in"
    lines = [
        f"#domain_mode: {family}",
        "#domain: " + " ".join(domain),
        "#dx_dy_dz: 0.002 0.002 0.002",
        "#time_window: 3e-9",
        "#num_threads: 1",
        "#pml_cells: " + " ".join(pml),
        "#waveform: ricker 1 1e9 pulse",
        f"#{source_command}: {'xyz'[axis]} {' '.join(source)} pulse",
    ]
    for formulation in ("ksir", "equivalent_current"):
        lines.append(f"#ntff_contour: {' '.join(lower + upper)} {formulation} 0 350 10 1e9 rectangular {formulation}")
    model.write_text("\n".join(lines) + "\n")
    environment = os.environ.copy()
    environment.pop("MPI4PY_RC_FINALIZE", None)
    environment.pop("FI_PROVIDER", None)
    environment["OMP_NUM_THREADS"] = "1"
    for split in (None, (axis + 1) % 3, (axis + 2) % 3):
        name = "serial" if split is None else f"mpi{split}"
        command = [
            sys.executable,
            "-m",
            "gprMax",
            str(model),
            "-outputfile",
            str(tmp_path / name),
            "--hide-progress-bars",
            "--no-dft-convergence",
        ]
        if split is not None:
            shape = ["1"] * 3
            shape[split] = "2"
            command = [launcher(), "-n", "2", *command, "--mpi", *shape]
        result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=90)
        (tmp_path / f"{name}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        if split is None:
            continue
        with h5py.File(tmp_path / "serial.h5") as serial, h5py.File(tmp_path / f"{name}.h5") as distributed:
            for method in ("ksir", "equivalent_current"):
                reference = serial[f"ntff_contours/{method}/scalar"][:]
                actual = distributed[f"ntff_contours/{method}/scalar"][:]
                assert np.max(abs(actual - reference)) / np.max(abs(reference)) < 1e-10
                assert distributed[f"ntff_contours/{method}"].attrs["mpi_ranks"] == 2

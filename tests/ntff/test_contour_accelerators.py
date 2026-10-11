"""Real accelerator parity for both 2-D contour formulations."""

import sys

import numpy as np
import pytest

import gprMax
from tests.ntff.test_contour import scene


@pytest.mark.gpu
@pytest.mark.parametrize("family", ("TE", "TM"))
@pytest.mark.parametrize("formulation", ("ksir", "equivalent_current"))
@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("backend", ("cuda", "opencl", "metal"))
@pytest.mark.parametrize("precision", ("single", "double"))
def test_device_contour_matches_cpu(tmp_path, request, family, formulation, axis, backend, precision):
    if backend == "cuda":
        options = {"gpu": [request.getfixturevalue("gpu_device")]}
    elif backend == "opencl":
        options = {"opencl": [request.getfixturevalue("opencl_device")]}
    else:
        if sys.platform != "darwin":
            pytest.skip("Metal requires macOS hardware")
        if precision == "double":
            pytest.skip("Metal supports single precision")
        pytest.importorskip("Metal")
        options = {"metal": True}
    cpu_scene, cpu = scene(family, axis, formulation=formulation)
    device_scene, device = scene(family, axis, formulation=formulation)
    gprMax.run(scenes=[cpu_scene], outputfile=tmp_path / "cpu", cpu_precision=precision, hide_progress_bars=True)
    gprMax.run(
        scenes=[device_scene],
        outputfile=tmp_path / "device",
        gpu_precision=precision,
        hide_progress_bars=True,
        **options
    )
    reference = cpu.result["scalar"]
    error = np.max(abs(device.result["scalar"] - reference)) / np.max(abs(reference))
    assert error < (2e-4 if precision == "single" else 1e-10)
    assert np.all(np.isfinite(device.result["scalar"]))

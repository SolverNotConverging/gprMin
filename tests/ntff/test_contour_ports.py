"""Matched reduced-mode ports, with real CUDA parity and antenna power checks."""

import h5py
import numpy as np
import pytest

import gprMax
from testing.validation.impedance_surface.validate_2d import scene_2d


@pytest.mark.gpu
@pytest.mark.parametrize("family", ["TE", "TM"])
@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("direction", ["+", "-"])
@pytest.mark.parametrize("normal_offset", [1, 2])
def test_cuda_matched_guide(tmp_path, gpu_device, family, axis, direction, normal_offset):
    values = []
    for backend in ("cpu", "cuda"):
        scene = scene_2d(
            polarization=family,
            invariant=axis,
            normal=(axis + normal_offset) % 3,
            direction=direction,
            resistance="pec",
            virtual=True,
            active=True,
            steps=900,
        )
        waveform = next(obj for obj in scene.grid_objects if isinstance(obj, gprMax.Waveform))
        waveform.kwargs["wave_type"] = "ricker"
        path = tmp_path / backend
        gprMax.run(
            scenes=[scene],
            outputfile=path,
            gpu=[gpu_device] if backend == "cuda" else None,
            cpu_precision="double",
            gpu_precision="double",
            hide_progress_bars=True,
            log_level=30,
        )
        with h5py.File(path.with_suffix(".h5")) as f:
            p = f["eigenmode_ports/port1"]
            assert np.all(p["power_wave_valid"][...])
            assert np.all(np.isfinite(p["S"][...]))
            assert np.max(abs(p["S"][...])) < 0.03
            # A unit-power mode in an infinite uniform lossless guide has
            # S11=0 and accepted spectral power equal to the Ricker spectrum
            # squared. This reference has no fitted amplitude or phase.
            expected = (2 / (np.sqrt(np.pi) * 22e9) * np.exp(-1)) ** 2
            a, b, w = p["incident"][...], p["outgoing"][...], p["power_matrix"][...]
            accepted = np.real(np.vdot(a[:, 0], w[0] @ a[:, 0]) - np.vdot(b[:, 0], w[0] @ b[:, 0]))
            np.testing.assert_allclose(accepted, expected, rtol=0.01)
            values.append(np.stack([p[name][...] for name in ("incident", "outgoing")]))
    np.testing.assert_allclose(values[1], values[0], atol=np.max(abs(values[0])) * 1e-10, rtol=1e-9)


@pytest.mark.gpu
@pytest.mark.parametrize("family", ["TE", "TM"])
@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("direction", ["+", "-"])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_cuda_antenna(tmp_path, gpu_device, family, axis, direction, precision):
    from testing.validation.ntff_2d.matched_antenna import antenna_scene

    results = []
    for backend in ("cpu", "cuda"):
        scene, contours = antenna_scene(family, axis=axis, direction=direction)
        gprMax.run(
            scenes=[scene],
            outputfile=tmp_path / backend,
            gpu=[gpu_device] if backend == "cuda" else None,
            cpu_precision=precision,
            gpu_precision=precision,
            hide_progress_bars=True,
            log_level=30,
        )
        results.append([c.result for c in contours])
        for result in results[-1]:
            assert np.all(result["port_power_valid"])
            assert np.all(result["gain_valid"])
            assert np.all(np.isfinite(result["scalar"]))
            expected_incident = (2 / (np.sqrt(np.pi) * 22e9) * np.exp(-1)) ** 2
            np.testing.assert_allclose(result["incident_power_per_port_per_length"], expected_incident, rtol=0.005)
            np.testing.assert_allclose(result["radiation_efficiency"], 1, atol=0.05)
        scale = np.max(abs(results[-1][0]["scalar"]))
        assert np.max(abs(results[-1][0]["scalar"] - results[-1][1]["scalar"])) / scale < 0.04
    for cpu, cuda in zip(*results):
        tolerance = 5e-5 if precision == "single" else 1e-9
        np.testing.assert_allclose(
            cuda["scalar"], cpu["scalar"], atol=np.max(abs(cpu["scalar"])) * tolerance, rtol=tolerance
        )

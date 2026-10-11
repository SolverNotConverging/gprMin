"""DFT stopping must include phase, weak bins, all monitors and valid records."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.dft_convergence import (
    DFTConvergence, DFTConvergenceSettings, SpectralStability,
    _source_end, completed_output_window,
)


@pytest.mark.unit
def test_every_monitor_and_weak_frequency_must_be_stable():
    check = SpectralStability(1e-3, 3)
    stationary = np.ones((2, 3), dtype=complex)
    for step in range(5):
        changing = np.array([[1.0], [1e-8 * (step + 1)]])
        assert not check.observe([stationary, changing])
    for _ in range(3):
        converged = check.observe([stationary, changing])
    assert converged


@pytest.mark.unit
def test_phase_changes_and_cumulative_drift_prevent_stop():
    phase = SpectralStability(1e-3, 3)
    drift = SpectralStability(1e-3, 3)
    for step in range(8):
        assert not phase.observe([np.array([[np.exp(0.1j * step)]])])
        assert not drift.observe([np.array([[1 + 0.0005 * step]])])


@pytest.mark.unit
def test_silent_or_nonfinite_spectra_do_not_stop():
    check = SpectralStability(1e-3, 3)
    for _ in range(5):
        assert not check.observe([np.zeros((2, 1))])
    assert not check.observe([np.array([[np.nan]])])
    assert not check.observe([np.array([[np.inf]])])
    for _ in range(3):
        assert not check.observe([np.ones((1, 1))])
    assert check.observe([np.ones((1, 1))])


@pytest.mark.unit
def test_future_delayed_pulse_and_continuous_drives_block_early_stop():
    waveform = np.zeros(100)
    waveform[5] = 1
    waveform[80] = 0.1
    grid = SimpleNamespace(iterations=100, voltagesources=[SimpleNamespace(waveformvalues_halfdt=waveform)])
    assert _source_end(grid, 1e-3) == 81
    grid.voltagesources[0].waveformvalues_halfdt[:] = 1
    assert _source_end(grid, 1e-3) == 100


@pytest.mark.unit
@pytest.mark.parametrize("values", [dict(rtol=0), dict(rtol=np.nan), dict(stable_checks=1),
                                    dict(check_interval=-1), dict(minimum_time=-1)])
def test_invalid_controls_are_rejected(values):
    with pytest.raises(ValueError):
        DFTConvergenceSettings(**values)


@pytest.mark.unit
def test_completed_output_views_restore_allocations_even_on_failure():
    history = np.arange(21)
    receiver = SimpleNamespace(outputs={"Ez": np.arange(20)})
    grid = SimpleNamespace(iterations=20, completed_iterations=8, timewindow=19e-12, dt=1e-12,
                           rxs=[receiver], snapshots=[SimpleNamespace(iteration=i) for i in (4, 15)],
                           transmissionlines=[SimpleNamespace(Vtotal=history)])
    with pytest.raises(RuntimeError):
        with completed_output_window(grid):
            assert grid.iterations == 8
            assert grid.timewindow == 7e-12
            assert receiver.outputs["Ez"].size == 8
            assert grid.transmissionlines[0].Vtotal.size == 9
            assert len(grid.snapshots) == 1
            raise RuntimeError("writer error")
    assert grid.iterations == 20
    assert receiver.outputs["Ez"].size == 20
    assert grid.transmissionlines[0].Vtotal is history
    assert len(grid.snapshots) == 2


def _scene(port=False, window="rectangular", iterations=1200, ntff=True, sar=False):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(0.002,) * 3))
    scene.add(gprMax.Domain(p1=(0.04,) * 3))
    scene.add(gprMax.PMLThickness(thickness=4))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.TimeWindow(iterations=iterations))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="pulse"))
    if sar:
        scene.add(gprMax.Material(er=1, se=0.2, mr=1, sm=0, id="target_material"))
        scene.add(gprMax.MaterialDensity(density=1000, material_ids="target_material"))
        scene.add(gprMax.Box(p1=(0.016,) * 3, p2=(0.024,) * 3, material_id="target_material", tag="target"))
        scene.add(gprMax.SAR(frequencies=(8e9, 10e9, 12e9), waveform_id="pulse", tags="target", id="target_sar"))
    if port == "network":
        scene.add(gprMax.RationalNetwork(id="source_50", conductance=1 / 50))
        scene.add(gprMax.NetworkTerminal(p1=(0.02,) * 3, polarisation="z", network_id="source_50", id="feed"))
        scene.add(gprMax.NetworkExcitation("feed", "pulse"))
        scene.add(gprMax.NetworkPort("feed", reference_impedance=50))
    elif port == "tl":
        scene.add(gprMax.TransmissionLine(polarisation="z", p1=(0.02,) * 3,
                                          waveform_id="pulse", resistance=50))
    elif port:
        scene.add(gprMax.VoltageSource(polarisation="z", p1=(0.02,) * 3,
                                       waveform_id="pulse", resistance=0 if port == "hard" else 50, id="feed"))
    else:
        scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.02,) * 3, waveform_id="pulse"))
    scene.add(gprMax.Rx(p1=(0.022, 0.02, 0.02), id="probe"))
    if ntff:
        scene.add(gprMax.NTFFSurface(p1=(0.012,) * 3, p2=(0.028,) * 3, id="surface", origin=(0.02,) * 3))
        scene.add(gprMax.KSIRFrequencyTransform("surface", "spectrum", (8e9, 10e9, 12e9), window=window))
        scene.add(gprMax.KSIRFarField(theta=np.array([90.0]), phi=np.array([0.0]),
                                     transform_id="spectrum", id="pattern", outputs=("Etheta",)))
    return scene


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["hard", "tl", "network"])
def test_native_port_output_matches_fixed_run_after_early_stop(tmp_path, monkeypatch, kind):
    # Isolate output finalisation from the physical decay of these feeds. Some
    # near-zero DC current bins legitimately prevent their natural convergence.
    monkeypatch.setattr(DFTConvergence, "should_stop", lambda self, count: count == 512)
    path = tmp_path / "port"
    gprMax.run(scenes=[_scene(port=kind, ntff=False, iterations=3000)], outputfile=path,
               hide_progress_bars=True, log_level=30, cpu_precision="double")
    with h5py.File(str(path) + ".h5") as output:
        count = output.attrs["Iterations"]
        assert 0 < count < 3000
        group = output["tls/tl1"] if kind == "tl" else output["ports/feed"]
        assert group["time_voltage" if kind == "tl" else "time"].size == count
        assert np.all(np.isfinite(group["Vtotal_spectrum"][:]))
        spectrum = group["Vtotal_spectrum"][:]
    reference = tmp_path / "reference"
    gprMax.run(scenes=[_scene(port=kind, ntff=False, iterations=512)], outputfile=reference,
               hide_progress_bars=True, log_level=30, cpu_precision="double", dft_convergence=False)
    with h5py.File(str(reference) + ".h5") as output:
        group = output["tls/tl1"] if kind == "tl" else output["ports/feed"]
        np.testing.assert_allclose(spectrum, group["Vtotal_spectrum"][:], rtol=1e-12, atol=1e-20)


@pytest.mark.integration
@pytest.mark.parametrize("port", [False, True])
def test_real_pulse_stops_and_matches_full_record(tmp_path, port):
    common = dict(hide_progress_bars=True, log_level=30, cpu_precision="double")
    early = tmp_path / "early"
    full = tmp_path / "full"
    maximum = 3000 if port else 1200
    gprMax.run(scenes=[_scene(port=port, iterations=maximum)], outputfile=early, **common)
    gprMax.run(scenes=[_scene(port=port, iterations=maximum)], outputfile=full, dft_convergence=False, **common)
    with h5py.File(str(early) + ".h5") as a, h5py.File(str(full) + ".h5") as b:
        status = a["dft_convergence"].attrs
        assert status["Converged"]
        assert status["StopReason"] == "dft_converged"
        assert 0 < a.attrs["Iterations"] < b.attrs["Iterations"] == maximum
        assert a["rxs/rx1/Ez"].shape == (a.attrs["Iterations"],)
        np.testing.assert_allclose(a["rxs/rx1/Ez"][:], b["rxs/rx1/Ez"][:a.attrs["Iterations"]])
        path = "ntff/surface/frequency/spectrum/far_field/pattern/fields/Etheta"
        np.testing.assert_allclose(a[path][:], b[path][:], rtol=2e-3, atol=1e-20)
        if port:
            assert a["ports/feed/time"].size == a.attrs["Iterations"] - 1
            assert np.all(np.isfinite(a["ports/feed/Vtotal_spectrum"][:]))


@pytest.mark.integration
@pytest.mark.parametrize("ntff,window,reason", [(False, "rectangular", "no_dft_bins"),
                                               (True, "hann", "fixed_taper_requires_full_window")])
def test_no_bins_and_fixed_taper_keep_maximum_record(tmp_path, ntff, window, reason):
    path = tmp_path / "fixed"
    gprMax.run(scenes=[_scene(ntff=ntff, window=window, iterations=180)], outputfile=path,
               hide_progress_bars=True, log_level=30)
    with h5py.File(str(path) + ".h5") as output:
        assert output.attrs["Iterations"] == 180
        assert output["dft_convergence"].attrs["StopReason"] == reason
        assert not output["dft_convergence"].attrs["Converged"]


@pytest.mark.integration
def test_sar_early_stop_preserves_source_normalisation(tmp_path):
    arrays = []
    for name, enabled in (("early", True), ("full", False)):
        path = tmp_path / name
        gprMax.run(scenes=[_scene(sar=True)], outputfile=path, dft_convergence=enabled,
                   hide_progress_bars=True, log_level=30, cpu_precision="double")
        with h5py.File(str(path) + ".h5") as output:
            arrays.append(output["sar/target_sar/sar"][:])
            if enabled:
                assert output["dft_convergence"].attrs["Converged"]
                assert output.attrs["Iterations"] < 1200
    np.testing.assert_allclose(arrays[0], arrays[1], rtol=2e-3, atol=1e-25)


@pytest.mark.unit
def test_modal_and_ntff_and_sar_arrays_all_participate():
    accumulator = SimpleNamespace(inside_dft=np.ones((2, 3)), outside_dft=np.ones((2, 3)))
    ntff = SimpleNamespace(frequencies=np.array([1e9, 2e9]), _accumulators={"Ez": accumulator})
    modal = SimpleNamespace(frequency=np.array([1e9, 2e9]), electric_dft=np.ones((2, 2)),
                            magnetic_dft=np.ones((2, 2)))
    sar = SimpleNamespace(frequencies=np.array([1e9, 2e9]), accumulators={"Ez": np.ones((2, 4))})
    grid = SimpleNamespace(iterations=1000, ntff_monitors=[ntff], eigenmodeports=[modal], sar_monitors=[sar],
                           dl=np.ones(3), nx=0, ny=0, nz=0, dt=1e-12, materials=[])
    convergence = DFTConvergence(SimpleNamespace(grid=grid), DFTConvergenceSettings(check_interval=10), "cpu")
    spectra = convergence._spectra(10)
    assert len(spectra) == 5
    for step in range(1, 6):
        modal.magnetic_dft[1, 1] = step
        assert not convergence.should_stop(10 * step)
    for step in range(6, 9):
        stopped = convergence.should_stop(10 * step)
    assert stopped


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("port,sar", [(False, False), (True, False), (False, True)])
def test_cuda_early_stop_matches_cpu_and_full_cuda(tmp_path, gpu_device, port, sar):
    maximum = 3000 if port else 1200
    common = dict(hide_progress_bars=True, log_level=30, dft_check_interval=64)
    results = []
    sar_results = []
    for name, backend in (("cpu", dict(cpu_precision="double")),
                          ("cuda", dict(gpu=[gpu_device], gpu_precision="double")),
                          ("cuda_full", dict(gpu=[gpu_device], gpu_precision="double", dft_convergence=False))):
        path = tmp_path / name
        gprMax.run(scenes=[_scene(port=port, iterations=maximum, sar=sar)], outputfile=path, **backend, **common)
        with h5py.File(str(path) + ".h5") as output:
            dft = output["ntff/surface/frequency/spectrum/far_field/pattern/fields/Etheta"][:]
            results.append((output.attrs["Iterations"], bool(output["dft_convergence"].attrs["Converged"]), dft))
            if sar:
                sar_results.append(output["sar/target_sar/sar"][:])
    assert results[0][1] and results[1][1]
    assert results[0][0] == results[1][0] < results[2][0] == maximum
    np.testing.assert_allclose(results[0][2], results[1][2], rtol=1e-9, atol=1e-20)
    np.testing.assert_allclose(results[1][2], results[2][2], rtol=2e-3, atol=1e-20)
    if sar:
        np.testing.assert_allclose(sar_results[0], sar_results[1], rtol=1e-9, atol=1e-25)
        np.testing.assert_allclose(sar_results[1], sar_results[2], rtol=2e-3, atol=1e-25)

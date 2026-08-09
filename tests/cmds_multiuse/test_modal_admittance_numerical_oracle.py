"""One-way raw-Yee oracle for modal terminal-admittance development."""

from types import SimpleNamespace

import numpy as np
import pytest

import gprMax
import gprMax.config as config
from gprMax.updates.cpu_updates import CPUUpdates


def _one_way_microstrip_scene():
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(0.5e-3, 0.5e-3, 0.5e-3)))
    scene.add(gprMax.Domain(p1=(0.060, 0.012, 0.006)))
    scene.add(gprMax.PMLThickness(thickness=(0, 0, 0, 10, 0, 0)))
    scene.add(gprMax.TimeWindow(time=5e-9))
    scene.add(gprMax.Material(er=4.4, se=0, mr=1, sm=0, id="substrate"))
    scene.add(
        gprMax.Box(
            p1=(0, 0, 0),
            p2=(0.060, 0.012, 0.0005),
            material_id="pec",
        )
    )
    scene.add(
        gprMax.Box(
            p1=(0, 0, 0.0005),
            p2=(0.060, 0.012, 0.0020),
            material_id="substrate",
        )
    )
    scene.add(
        gprMax.Box(
            p1=(0, 0.005, 0.0020),
            p2=(0.060, 0.007, 0.0025),
            material_id="pec",
        )
    )
    scene.add(
        gprMax.EigenmodeBand(
            id="microstrip_band",
            fmin=4e9,
            fmax=5e9,
            points=21,
        )
    )
    scene.add(
        gprMax.EigenmodePort(
            port=1,
            p1=(0.003, 0, 0.0005),
            p2=(0.003, 0.012, 0.006),
            direction="+",
            modes=(1,),
            anchors="auto",
            plot_fields=False,
        )
    )
    scene.add(gprMax.EigenmodeMatch(port=1, depth_cells=6))
    scene.add(
        gprMax.EigenmodeExcitation(
            port=1,
            mode=1,
            waveform="auto",
            plot_waveform=False,
        )
    )
    return scene


def _raw_modal_sample(boundary, grid, plane_index):
    u_axis, v_axis = boundary.owner.transverse_axes
    electric_fields = (grid.Ex, grid.Ey, grid.Ez)
    magnetic_fields = (grid.Hx, grid.Hy, grid.Hz)
    electric = boundary._flatten_tangential_fields(
        {
            axis: boundary._tangential_component_view(
                electric_fields[axis], axis, plane_index
            )
            for axis in boundary.owner.transverse_axes
        }
    )
    raw_hu = boundary._local_component_view(
        magnetic_fields[u_axis], 0, "H", plane_index
    )
    raw_hv = boundary._local_component_view(
        magnetic_fields[v_axis], 1, "H", plane_index
    )
    handedness = boundary.owner._modal_basis_handedness()
    measure = boundary._port_measure(grid)
    magnetic_covector = measure * np.concatenate(
        (handedness * raw_hv.ravel(), -handedness * raw_hu.ravel())
    )
    centre_covector = measure * np.concatenate(
        (
            handedness * boundary.modal_hv[0].ravel(),
            -handedness * boundary.modal_hu[0].ravel(),
        )
    )
    voltage = float(electric @ centre_covector / boundary.power_gram[0, 0])
    current = float(boundary.basis[0] @ magnetic_covector / boundary.power_gram[0, 0])
    return voltage, current


@pytest.mark.integration
def test_one_way_raw_yee_microstrip_admittance_oracle(tmp_path, monkeypatch):
    """Measure the numerical travelling-wave ratio before the remote PML return."""

    samples = SimpleNamespace(voltage=[], current=[], dt=None, boundary=None)
    original = CPUUpdates.observe_eigenmode_ports

    def observe_and_sample(self, iteration):
        original(self, iteration)
        if not self.grid.eigenmodematches:
            return
        boundary = self.grid.eigenmodematches[0]
        voltage, current = _raw_modal_sample(boundary, self.grid, plane_index=20)
        samples.voltage.append(voltage)
        samples.current.append(current)
        samples.dt = float(self.grid.dt)
        samples.boundary = boundary

    monkeypatch.setattr(CPUUpdates, "observe_eigenmode_ports", observe_and_sample)
    gprMax.run(
        scenes=[_one_way_microstrip_scene()],
        n=1,
        outputfile=tmp_path / "one_way_modal_admittance",
        hide_progress_bars=True,
    )

    voltage = np.asarray(samples.voltage, dtype=np.float64)
    current = np.asarray(samples.current, dtype=np.float64)
    assert voltage.shape == current.shape
    assert voltage.size > 1000
    assert np.all(np.isfinite(voltage))
    assert np.all(np.isfinite(current))

    boundary = samples.boundary
    monitor = boundary.owner.port_monitor
    frequency = np.asarray(monitor.frequency, dtype=np.float64)
    indices = np.arange(voltage.size, dtype=np.float64)
    electric_phase = np.exp(
        -2j * np.pi * frequency[:, None] * indices[None, :] * samples.dt
    )
    magnetic_phase = np.exp(
        -2j
        * np.pi
        * frequency[:, None]
        * (indices[None, :] + 0.5)
        * samples.dt
    )
    voltage_dft = electric_phase @ voltage
    current_dft = magnetic_phase @ current
    assert np.all(np.abs(voltage_dft) > 1e-16)

    beta = (
        2
        * np.pi
        * frequency
        * np.real(np.asarray(monitor.neff[:, 0], dtype=np.complex128))
        / config.c
    )
    colocated = current_dft / voltage_dft * np.exp(
        0.5j * beta * 0.5e-3
    )
    anchor_frequency = np.asarray(monitor.anchor_frequencies, dtype=np.float64)
    anchor_admittance = boundary.fixed_basis_admittance_samples.admittances
    expected = np.interp(frequency, anchor_frequency, np.real(anchor_admittance))
    expected = expected + 1j * np.interp(
        frequency,
        anchor_frequency,
        np.imag(anchor_admittance),
    )
    inband = (frequency >= 4.1e9) & (frequency <= 4.9e9)

    assert np.all(np.isfinite(colocated))
    assert np.max(np.abs(colocated[inband] - expected[inband])) < 0.04

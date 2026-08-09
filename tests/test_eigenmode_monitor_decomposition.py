"""Focused travelling-wave decomposition tests for eigenmode monitors."""

from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
from gprMax.eigenmode_ports import EigenmodePortMonitor


@pytest.mark.parametrize("direction", ("+", "-"))
@pytest.mark.parametrize("magnetic_side", (-1, 1))
@pytest.mark.parametrize(
    ("incident", "outgoing"),
    (
        (1.25 - 0.75j, 0.0j),
        (0.0j, -0.5 + 1.5j),
    ),
    ids=("pure-incident", "pure-outgoing"),
)
def test_finalise_decomposes_pure_staggered_travelling_waves(
    monkeypatch,
    direction,
    magnetic_side,
    incident,
    outgoing,
):
    """Recover pure waves for either port direction and magnetic half-cell.

    The observation kernel has already multiplied the physical magnetic
    overlap by ``direction_sign`` when ``finalise`` is entered.  Consequently
    finalisation receives the direction-normalised coefficients

        E = a + b,
        M = exp(-j beta delta) a - exp(+j beta delta) b,

    where ``delta = magnetic_side * dl / 2``.  Both port directions must then
    have the same decomposition; direction changes the raw magnetic field
    sign upstream, not the final algebra.
    """

    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(dtypes={"complex": np.complex128}),
    )

    frequency = np.asarray([1.7e9, 4.9e9], dtype=np.float64)
    neff = np.asarray([[1.45], [1.62]], dtype=np.complex128)
    cell_size = 2.3e-3
    magnetic_offset = magnetic_side * 0.5 * cell_size
    beta = 2 * np.pi * frequency[:, np.newaxis] * neff / config.c
    forward_phase = np.exp(-1j * beta * magnetic_offset)
    backward_phase = np.exp(1j * beta * magnetic_offset)

    electric_coeff = np.full((frequency.size, 1), incident + outgoing)
    direction_sign = 1 if direction == "+" else -1
    direction_normalised_magnetic = (
        forward_phase * incident - backward_phase * outgoing
    )
    raw_physical_magnetic = direction_sign * direction_normalised_magnetic
    magnetic_coeff_seen_by_finalise = direction_sign * raw_physical_magnetic

    monitor = EigenmodePortMonitor.__new__(EigenmodePortMonitor)
    monitor.frequency = frequency
    monitor.neff = neff
    monitor.electric_dft = electric_coeff
    monitor.magnetic_dft = magnetic_coeff_seen_by_finalise
    monitor.electric_gram = np.ones((frequency.size, 1, 1), dtype=np.complex128)
    monitor.magnetic_gram = np.ones((frequency.size, 1, 1), dtype=np.complex128)
    monitor.mode_power_valid = np.ones((frequency.size, 1), dtype=bool)
    monitor.power_matrix_valid = np.ones(frequency.size, dtype=bool)
    monitor.owner = SimpleNamespace(normal_axis=0, direction=direction)
    monitor.magnetic_side = magnetic_side

    result = monitor.finalise(
        SimpleNamespace(dl=np.asarray([cell_size, 1.0, 1.0]))
    )

    np.testing.assert_allclose(
        result.incident[0],
        incident,
        rtol=2e-14,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        result.outgoing[0],
        outgoing,
        rtol=2e-14,
        atol=2e-14,
    )
    assert result.valid[0].all()
    np.testing.assert_allclose(result.condition_number, 1.0)


@pytest.mark.parametrize(
    ("direction", "expected_hplane"),
    (("+", 1), ("-", 2)),
)
def test_observe_and_finalise_recover_a_pure_forward_yee_wave(
    monkeypatch,
    direction,
    expected_hplane,
):
    """Exercise physical plane selection, staggering, DFT, and final split."""

    sample_count = 64
    cycles = 7
    dt = 1e-12
    frequency = cycles / (sample_count * dt)
    beta = 2 * np.pi * frequency / config.c
    cell_size = 0.82 / beta
    half_cell_phase = 0.5 * beta * cell_size
    amplitude = 0.8 + 0.3j
    direction_sign = 1 if direction == "+" else -1
    owner = SimpleNamespace(
        normal_axis=0,
        direction=direction,
        transverse_start=(0, 0),
        transverse_stop=(1, 1),
        plane_index=2,
    )
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            dtypes={
                "C_float_or_double": "double",
                "complex": np.complex128,
            }
        ),
    )
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: SimpleNamespace(ompthreads=1),
    )

    monitor = EigenmodePortMonitor.__new__(EigenmodePortMonitor)
    monitor.owner = owner
    monitor.magnetic_side = -1
    monitor.frequency = np.asarray([frequency])
    monitor.neff = np.ones((1, 1), dtype=np.complex128)
    monitor.measure = 1.0
    monitor.handedness = 1
    monitor.electric_dft = np.zeros((1, 1), dtype=np.complex128)
    monitor.magnetic_dft = np.zeros((1, 1), dtype=np.complex128)
    monitor.phase_step = np.exp(-2j * np.pi * monitor.frequency * dt)
    monitor.electric_phase = np.ones(1, dtype=np.complex128)
    monitor.magnetic_phase = np.exp(
        -1j * np.pi * monitor.frequency * dt
    ).astype(np.complex128)
    ones = np.ones((1, 1, 1, 1), dtype=np.complex128)
    zeros = np.zeros_like(ones)
    monitor.conj_eu = ones
    monitor.conj_ev = zeros
    monitor.conj_hu = zeros
    monitor.conj_hv = ones
    monitor.electric_gram = np.full((1, 1, 1), 0.5, dtype=np.complex128)
    monitor.magnetic_gram = np.full((1, 1, 1), 0.5, dtype=np.complex128)
    monitor.mode_power_valid = np.ones((1, 1), dtype=bool)
    monitor.power_matrix_valid = np.ones(1, dtype=bool)
    monitor._next_iteration = 0

    shape = (4, 2, 2)
    ex = np.zeros(shape)
    ey = np.zeros(shape)
    ez = np.zeros(shape)
    hx = np.zeros(shape)
    hy = np.zeros(shape)
    hz = np.zeros(shape)
    grid = SimpleNamespace(
        dt=dt,
        dl=np.asarray((cell_size, 1.0, 1.0)),
        Ex=ex,
        Ey=ey,
        Ez=ez,
        Hx=hx,
        Hy=hy,
        Hz=hz,
    )

    for iteration in range(sample_count):
        electric_phasor = amplitude * np.exp(
            2j * np.pi * frequency * iteration * dt
        )
        magnetic_phasor = (
            direction_sign
            * amplitude
            * np.exp(1j * half_cell_phase)
            * np.exp(2j * np.pi * frequency * (iteration + 0.5) * dt)
        )
        ey[2, 0, :] = np.real(electric_phasor)
        hz[1:3, 0, :] = 99.0
        hz[expected_hplane, 0, :] = np.real(magnetic_phasor)
        monitor.observe(grid, iteration)

    result = monitor.finalise(grid)
    expected = 0.5 * sample_count * dt * amplitude

    np.testing.assert_allclose(result.incident[0, 0], expected, rtol=2e-13, atol=1e-25)
    assert abs(result.outgoing[0, 0]) < 1e-12 * abs(result.incident[0, 0])

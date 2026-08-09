import numpy as np
import pytest

from gprMax.matched_eigenmode_ports import constant_modal_admittance_step
from gprMax.modal_admittance import PoleResidueAdmittance
from gprMax.modal_admittance_ade import RationalModalAdmittanceADE
from gprMax.modal_admittance_ade import bilinear_prewarp_angular_frequency
from gprMax.modal_admittance_ade import pole_residue_to_real_state_space


def _model(poles=(), residues=(), direct=1.0):
    return PoleResidueAdmittance(
        poles=np.asarray(poles, dtype=np.complex128),
        residues=np.asarray(residues, dtype=np.complex128),
        direct=direct,
    )


def _explicit_response(model, s):
    values = np.asarray(s, dtype=np.complex128)
    response = np.full(values.shape, complex(model.direct), dtype=np.complex128)
    for pole, residue in zip(model.poles, model.residues):
        response += residue / (values - pole)
    return response


def test_bilinear_prewarp_matches_trapezoidal_mapping_and_is_odd():
    dt = 2.5e-12
    omega = np.asarray((0.0, 0.1, 0.4, 0.85)) * np.pi / dt
    expected = (2.0 / dt) * np.tan(0.5 * dt * omega)

    actual = bilinear_prewarp_angular_frequency(omega, dt)

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(
        bilinear_prewarp_angular_frequency(-omega, dt),
        -actual,
    )
    assert isinstance(bilinear_prewarp_angular_frequency(float(omega[1]), dt), float)


@pytest.mark.parametrize("factor", [-1.0, 1.0])
def test_bilinear_prewarp_rejects_nyquist_and_above(factor):
    dt = 0.125
    with pytest.raises(ValueError, match="strictly below the Nyquist"):
        bilinear_prewarp_angular_frequency(factor * np.pi / dt, dt)


def test_real_state_conversion_preserves_real_and_conjugate_pole_response():
    model = _model(
        poles=(-2.0, -3.0 + 4.0j, -3.0 - 4.0j),
        residues=(0.75, 0.2 + 0.35j, 0.2 - 0.35j),
        direct=1.25,
    )
    realization = pole_residue_to_real_state_space(model)
    s = 1j * np.geomspace(0.1, 100.0, 61)

    assert realization.state_count == 3
    assert np.isrealobj(realization.A)
    assert np.isrealobj(realization.B)
    assert np.isrealobj(realization.C)
    np.testing.assert_allclose(
        realization.evaluate(s),
        _explicit_response(model, s),
        rtol=2e-14,
        atol=2e-14,
    )
    for coefficients in (realization.A, realization.B, realization.C):
        assert not coefficients.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            coefficients.flat[0] = 0.0
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
            coefficients.setflags(write=True)


def test_real_state_conversion_is_independent_of_conjugate_pair_order():
    first = _model(
        poles=(-4.0 + 7.0j, -4.0 - 7.0j),
        residues=(0.5 - 0.3j, 0.5 + 0.3j),
        direct=0.8,
    )
    second = _model(
        poles=first.poles[::-1],
        residues=first.residues[::-1],
        direct=first.direct,
    )
    points = np.asarray((0.25j, 1.5j, 13.0j))

    np.testing.assert_allclose(
        pole_residue_to_real_state_space(first).evaluate(points),
        pole_residue_to_real_state_space(second).evaluate(points),
    )


def test_real_state_conversion_rejects_unpaired_or_unstable_poles():
    with pytest.raises(ValueError, match="explicit conjugate partner"):
        pole_residue_to_real_state_space(
            _model(poles=(-1.0 + 2.0j,), residues=(0.4 - 0.1j,))
        )
    with pytest.raises(ValueError, match="open left half-plane"):
        pole_residue_to_real_state_space(
            _model(poles=(0.25,), residues=(0.1,))
        )


def test_zero_pole_update_exactly_matches_constant_modal_admittance_recurrence():
    rng = np.random.default_rng(7812)
    dt = 0.037
    half_cell_storage = 0.23
    ade = RationalModalAdmittanceADE(
        _model(direct=1.0),
        dt=dt,
        half_cell_storage=half_cell_storage,
    )
    reference_voltage = 0.0
    for outward_current, incident in zip(
        rng.standard_normal(128),
        rng.standard_normal(128),
    ):
        # The legacy routine receives Q=-Iout.
        reference_voltage = constant_modal_admittance_step(
            reference_voltage,
            -outward_current,
            incident,
            half_cell_storage / dt,
        )
        actual = ade.step(outward_current, incident)
        assert actual == pytest.approx(reference_voltage, rel=3e-15, abs=3e-15)
    assert ade.state.size == 0


def test_ade_rejects_stable_but_nonpassive_characteristic_model():
    model = _model(poles=(-1.0,), residues=(-2.0,), direct=1.0)

    with pytest.raises(ValueError, match="globally positive-real"):
        RationalModalAdmittanceADE(
            model,
            dt=0.01,
            half_cell_storage=0.1,
        )


def test_frequency_response_includes_external_half_cell_storage_and_prewarp():
    model = _model(
        poles=(-2.0, -5.0 + 3.0j, -5.0 - 3.0j),
        residues=(0.6, 0.15 + 0.05j, 0.15 - 0.05j),
        direct=0.9,
    )
    dt = 0.025
    storage = 0.17
    ade = RationalModalAdmittanceADE(
        model,
        dt=dt,
        half_cell_storage=storage,
    )
    omega = np.asarray((0.1, 1.0, 8.0, 0.7 * np.pi / dt))
    mapped = bilinear_prewarp_angular_frequency(omega, dt)
    expected = 1j * storage * mapped + _explicit_response(model, 1j * mapped)

    np.testing.assert_allclose(
        ade.discrete_load_admittance(omega),
        expected,
        rtol=3e-14,
        atol=3e-14,
    )


def test_coupled_step_satisfies_both_trapezoidal_equations():
    ade = RationalModalAdmittanceADE(
        _model(
            poles=(-3.0, -5.0 + 8.0j, -5.0 - 8.0j),
            residues=(0.7, 0.2 - 0.1j, 0.2 + 0.1j),
            direct=1.1,
        ),
        dt=0.015,
        half_cell_storage=0.13,
    )
    ade.reset(voltage=-0.4, state=np.asarray((0.3, -0.2, 0.1)))
    old_voltage = ade.voltage
    old_state = ade.state
    current = 0.65
    incident = -0.25

    new_voltage = ade.step(current, incident)
    new_state = ade.state
    centred_voltage = 0.5 * (new_voltage + old_voltage) - 2.0 * incident
    centred_state = 0.5 * (new_state + old_state)
    state_residual = (
        (new_state - old_state) / ade.dt
        - ade.realization.A @ centred_state
        - ade.realization.B * centred_voltage
    )
    boundary_residual = (
        ade.half_cell_storage * (new_voltage - old_voltage) / ade.dt
        + ade.realization.direct * centred_voltage
        + ade.realization.C @ centred_state
        - current
    )

    np.testing.assert_allclose(state_residual, 0.0, rtol=0.0, atol=2e-14)
    assert boundary_residual == pytest.approx(0.0, abs=2e-14)


def test_reset_restores_voltage_and_state_without_aliasing():
    ade = RationalModalAdmittanceADE(
        _model(poles=(-3.0,), residues=(0.5,), direct=1.0),
        dt=0.02,
        half_cell_storage=0.1,
    )
    supplied_state = np.asarray((0.75,))
    ade.reset(voltage=-1.25, state=supplied_state)
    supplied_state[0] = 99.0

    assert ade.voltage == -1.25
    np.testing.assert_allclose(ade.state, (0.75,))
    ade.step(0.4, -0.2)
    assert ade.voltage != -1.25
    ade.reset()
    assert ade.voltage == 0.0
    np.testing.assert_array_equal(ade.state, (0.0,))


def test_passive_zero_input_realization_decays_without_growth():
    ade = RationalModalAdmittanceADE(
        _model(poles=(-3.0,), residues=(1.0,), direct=1.0),
        dt=0.01,
        half_cell_storage=0.2,
    )
    ade.reset(voltage=1.0, state=np.asarray((0.3,)))
    norm_history = []
    for _ in range(3000):
        ade.step(0.0, 0.0)
        norm_history.append(np.linalg.norm(np.r_[ade.voltage, ade.state]))

    assert np.all(np.isfinite(norm_history))
    assert max(norm_history) < 1.1
    assert norm_history[-1] < 1e-10


def test_half_cell_storage_suppresses_the_nyquist_computational_mode():
    dt = 1.0
    storage = 1e-4
    ade = RationalModalAdmittanceADE(
        _model(direct=1.0),
        dt=dt,
        half_cell_storage=storage,
    )
    ade.reset(voltage=1.0)
    expected_pole = (storage / dt - 0.5) / (storage / dt + 0.5)
    voltages = [ade.voltage]
    for _ in range(4000):
        voltages.append(ade.step(0.0, 0.0))

    assert -1.0 < expected_pole < 0.0
    np.testing.assert_allclose(
        np.asarray(voltages[1:]),
        expected_pole * np.asarray(voltages[:-1]),
        rtol=2e-14,
        atol=2e-14,
    )
    assert np.max(np.abs(voltages)) == pytest.approx(1.0)
    assert abs(voltages[-1]) < 1.0


def test_active_source_uses_characteristic_operator_but_not_half_cell_derivative():
    incident = 0.8
    ade = RationalModalAdmittanceADE(
        _model(
            poles=(-2.0, -6.0 + 4.0j, -6.0 - 4.0j),
            residues=(0.4, 0.1 + 0.2j, 0.1 - 0.2j),
            direct=1.2,
        ),
        dt=0.01,
        half_cell_storage=0.3,
    )
    # V=2a makes the characteristic-operator input V-2a zero. It must
    # therefore remain an exact equilibrium with zero outward current,
    # irrespective of the pole states and the half-cell storage.
    ade.reset(voltage=2.0 * incident)
    for _ in range(100):
        assert ade.step(0.0, incident) == pytest.approx(
            2.0 * incident,
            rel=2e-14,
            abs=2e-14,
        )
        np.testing.assert_allclose(ade.state, 0.0, rtol=0.0, atol=2e-14)

from dataclasses import replace

import numpy as np
import pytest

import gprMax.modal_admittance as modal_admittance
from gprMax.modal_admittance import PoleResidueAdmittance
from gprMax.modal_admittance import fit_fixed_poles
from gprMax.modal_admittance import reflection_equivalent_error
from gprMax.modal_admittance import seed_stable_poles
from gprMax.modal_admittance import synthesize_passive_admittance
from gprMax.modal_admittance import vector_fit_scalar
from gprMax.modal_admittance import yee_staggered_characteristic_admittance


def _sample(model, omega):
    return model.evaluate(1j * np.asarray(omega, dtype=np.float64))


def test_pole_residue_model_evaluates_scalar_and_array_without_mutating_inputs():
    poles = np.asarray((-2.0, -3.0 + 4.0j, -3.0 - 4.0j), dtype=np.complex128)
    residues = np.asarray((0.5, 0.2 - 0.1j, 0.2 + 0.1j), dtype=np.complex128)
    model = PoleResidueAdmittance(poles, residues, 0.8)
    points = 1j * np.asarray((0.0, 1.0, 7.0))
    expected = 0.8 + np.sum(
        residues[None, :] / (points[:, None] - poles[None, :]), axis=1
    )

    assert model.order == 3
    assert isinstance(model.evaluate(1j), complex)
    np.testing.assert_allclose(model.evaluate(points), expected)
    np.testing.assert_array_equal(poles, model.poles)
    np.testing.assert_array_equal(residues, model.residues)

    poles[0] = 99.0
    residues[0] = 99.0
    assert model.poles[0] == -2.0
    assert model.residues[0] == 0.5
    with pytest.raises(ValueError, match="read-only"):
        model.poles[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        model.residues[0] = 1.0
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        model.poles.setflags(write=True)


def test_seeded_poles_are_deterministic_stable_and_conjugate_symmetric():
    first = seed_stable_poles(6, 1e8, 1e11)
    second = seed_stable_poles(6, 1e8, 1e11)

    np.testing.assert_array_equal(first, second)
    assert np.all(first.real < 0)
    np.testing.assert_allclose(first[0::2], np.conj(first[1::2]))
    assert np.all(np.diff(np.abs(first[0::2].imag)) > 0)


def test_fixed_pole_solve_recovers_real_and_conjugate_residues():
    reference = PoleResidueAdmittance(
        poles=np.asarray((-2.5, -4.0 + 7.0j, -4.0 - 7.0j)),
        residues=np.asarray((0.7, 0.25 + 0.18j, 0.25 - 0.18j)),
        direct=1.1,
    )
    omega = np.geomspace(0.05, 100.0, 61)
    values = _sample(reference, omega)

    fitted = fit_fixed_poles(omega, values, reference.poles)

    np.testing.assert_allclose(fitted.poles, reference.poles)
    np.testing.assert_allclose(fitted.residues, reference.residues, rtol=2e-13, atol=2e-13)
    assert fitted.direct == pytest.approx(reference.direct, rel=2e-13, abs=2e-13)
    np.testing.assert_allclose(_sample(fitted, omega), values, rtol=2e-13, atol=2e-13)


def test_vector_fit_relocates_a_conjugate_pair_and_recovers_withheld_response():
    scale = 2 * np.pi * 4.5e9
    reference = PoleResidueAdmittance(
        poles=scale * np.asarray((-0.12 + 0.82j, -0.12 - 0.82j)),
        residues=scale * np.asarray((0.09 + 0.045j, 0.09 - 0.045j)),
        direct=0.93,
    )
    train = np.geomspace(0.12 * scale, 2.0 * scale, 31)
    result = vector_fit_scalar(
        train,
        _sample(reference, train),
        order=2,
        direct=reference.direct,
    )
    withheld = np.geomspace(0.14 * scale, 1.8 * scale, 47)

    assert np.all(result.model.poles.real < 0)
    assert result.diagnostics.converged
    assert result.diagnostics.iterations <= 20
    np.testing.assert_allclose(
        _sample(result.model, withheld),
        _sample(reference, withheld),
        rtol=2e-8,
        atol=2e-9,
    )
    assert result.diagnostics.maximum_reflection_error < 1e-8


def test_vector_fit_final_model_is_deterministic():
    scale = 2 * np.pi * 4.5e9
    reference = PoleResidueAdmittance(
        poles=scale * np.asarray((-0.12 + 0.82j, -0.12 - 0.82j)),
        residues=scale * np.asarray((0.09 + 0.045j, 0.09 - 0.045j)),
        direct=0.93,
    )
    omega = np.geomspace(0.12 * scale, 2.0 * scale, 31)
    values = _sample(reference, omega)

    first = vector_fit_scalar(omega, values, order=2, direct=reference.direct)
    second = vector_fit_scalar(omega, values, order=2, direct=reference.direct)

    np.testing.assert_array_equal(first.model.poles, second.model.poles)
    np.testing.assert_array_equal(first.model.residues, second.model.residues)
    assert first.model.direct == second.model.direct


def test_fixed_pole_fit_canonicalizes_an_approximately_conjugate_pair():
    canonical_poles = np.asarray((-1.000000005 + 2j, -1.000000005 - 2j))
    reference = PoleResidueAdmittance(
        poles=canonical_poles,
        residues=np.asarray((0.2 + 0.1j, 0.2 - 0.1j)),
        direct=1.0,
    )
    approximate_poles = np.asarray((-1.0 + 2j, -1.00000001 - 2j))
    omega = np.geomspace(0.05, 30.0, 31)

    fitted = fit_fixed_poles(
        omega,
        _sample(reference, omega),
        approximate_poles,
        direct=1.0,
    )

    assert fitted.poles[1] == fitted.poles[0].conjugate()
    np.testing.assert_allclose(fitted.poles, canonical_poles, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(_sample(fitted, omega), _sample(reference, omega))


def test_vector_fit_order_zero_returns_best_real_direct_term():
    omega = np.asarray((1.0, 2.0, 4.0, 8.0))
    values = np.asarray((0.8 + 0.2j, 1.0 + 0.1j, 1.2 - 0.1j, 1.4 - 0.2j))

    result = vector_fit_scalar(omega, values, order=0)

    assert result.model.order == 0
    assert result.model.direct == pytest.approx(np.mean(values.real))
    assert result.diagnostics.converged


def test_reflection_equivalent_error_handles_exact_and_singular_cases():
    reference = np.asarray((1.0 + 0.2j, 1.0, 0.0))
    approximation = np.asarray((1.0 + 0.2j, -1.0, 0.0))
    error = reflection_equivalent_error(reference, approximation)

    assert error[0] == pytest.approx(0.0)
    assert np.isinf(error[1])
    assert np.isinf(error[2])


def test_vector_fit_rejects_underdetermined_order():
    omega = np.geomspace(1.0, 10.0, 7)
    with pytest.raises(ValueError, match="requires at least 8"):
        vector_fit_scalar(omega, np.ones(omega.size), order=2)


def test_vector_fit_rejects_fractional_iteration_count():
    omega = np.geomspace(1.0, 10.0, 8)
    with pytest.raises(ValueError, match="maximum_iterations must be an integer"):
        vector_fit_scalar(
            omega,
            np.ones(omega.size),
            order=0,
            maximum_iterations=1.5,
        )


def test_passive_synthesis_selects_order_zero_for_constant_admittance():
    omega = np.geomspace(1.0, 100.0, 12)

    result = synthesize_passive_admittance(
        omega,
        np.ones(omega.size),
        candidate_orders=(0, 2),
        maximum_relative_error=1e-12,
        maximum_reflection_error=1e-12,
    )

    assert result.model.order == 0
    assert result.attempted_orders == (0,)
    assert result.final_passivity_certificate.is_passive


def test_passive_synthesis_rejects_fractional_candidate_order():
    omega = np.geomspace(1.0, 100.0, 12)
    with pytest.raises(ValueError, match="candidate_orders must contain"):
        synthesize_passive_admittance(
            omega,
            np.ones(omega.size),
            candidate_orders=(0, 2.5),
        )


def test_passive_synthesis_increases_order_until_validation_passes():
    reference = PoleResidueAdmittance(
        poles=np.asarray((-2.0,)),
        residues=np.asarray((0.7,)),
        direct=1.0,
    )
    train = np.geomspace(0.05, 100.0, 19)
    validation = np.geomspace(0.07, 80.0, 23)

    result = synthesize_passive_admittance(
        train,
        _sample(reference, train),
        candidate_orders=(0, 1),
        validation_angular_frequencies=validation,
        validation_admittances=_sample(reference, validation),
        maximum_relative_error=1e-8,
        maximum_reflection_error=1e-8,
    )

    assert result.model.order == 1
    assert result.attempted_orders == (0, 1)
    assert result.final_passivity_certificate.is_passive
    np.testing.assert_allclose(
        _sample(result.model, validation),
        _sample(reference, validation),
        rtol=2e-8,
        atol=2e-9,
    )


def test_passive_synthesis_rejects_when_no_supported_order_meets_error():
    omega = np.geomspace(1.0, 10.0, 5)
    values = 1.0 + 0.2j * np.linspace(-1.0, 1.0, omega.size)

    with pytest.raises(ValueError, match="no passive rational admittance model"):
        synthesize_passive_admittance(
            omega,
            values,
            candidate_orders=(0, 2),
            maximum_relative_error=1e-12,
            maximum_reflection_error=1e-12,
        )


def test_passive_synthesis_rejects_a_nonconverged_vector_fit(monkeypatch):
    reference = PoleResidueAdmittance(
        poles=np.asarray((-0.2 + 1.5j, -0.2 - 1.5j)),
        residues=np.asarray((0.1 + 0.03j, 0.1 - 0.03j)),
        direct=1.0,
    )
    omega = np.geomspace(0.05, 20.0, 17)
    original = modal_admittance.vector_fit_scalar

    def nonconverged(*args, **kwargs):
        result = original(*args, **kwargs)
        diagnostics = replace(
            result.diagnostics,
            converged=False,
            pole_movement=0.2,
        )
        return replace(result, diagnostics=diagnostics)

    monkeypatch.setattr(modal_admittance, "vector_fit_scalar", nonconverged)
    with pytest.raises(ValueError, match="vector fitting did not converge"):
        synthesize_passive_admittance(
            omega,
            _sample(reference, omega),
            candidate_orders=(2,),
            direct=1.0,
            maximum_relative_error=1.0,
            maximum_reflection_error=1.0,
        )


def test_yee_staggered_samples_remove_exact_half_cell_storage():
    dt = 0.8e-12
    spacing = 0.5e-3
    omega = 2 * np.pi * 4.5e9
    velocity = 1.62e8
    beta = omega / velocity
    theta = 0.5 * omega * dt
    half_phase = 0.5 * beta * spacing
    exact_storage = 0.5 * dt * np.sin(half_phase) / np.sin(theta)

    characteristic = yee_staggered_characteristic_admittance(
        omega,
        beta,
        1.0,
        normal_spacing=spacing,
        dt=dt,
        half_cell_storage=exact_storage,
    )

    assert characteristic.imag == pytest.approx(0.0, abs=2e-15)
    assert characteristic.real == pytest.approx(
        np.cos(half_phase) / np.cos(theta), rel=2e-14, abs=2e-14
    )

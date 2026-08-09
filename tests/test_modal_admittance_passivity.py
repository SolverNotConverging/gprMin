import numpy as np
import pytest

from gprMax.modal_admittance import PoleResidueAdmittance
from gprMax.modal_admittance_passivity import certify_scalar_positive_real
from gprMax.modal_admittance_passivity import repair_scalar_positive_real


def _model(poles=(), residues=(), direct=1.0):
    return PoleResidueAdmittance(
        poles=np.asarray(poles, dtype=np.complex128),
        residues=np.asarray(residues, dtype=np.complex128),
        direct=direct,
    )


def _evaluate(model, angular_frequencies):
    frequencies = np.asarray(angular_frequencies, dtype=np.float64)
    return model.direct + np.sum(
        model.residues[None, :]
        / (1j * frequencies[..., None] - model.poles[None, :]),
        axis=-1,
    )


def test_global_certificate_finds_violation_missed_by_dense_scan():
    centre = 1.00037
    damping = 1e-6
    model = _model(
        poles=(-damping + 1j * centre, -damping - 1j * centre),
        residues=(-2e-6, -2e-6),
        direct=1.0,
    )

    # A conventional-looking grid misses this approximately microradian-wide
    # negative notch entirely.
    scan = np.linspace(0.0, 2.0, 1001)
    assert np.min(_evaluate(model, scan).real) > 0.99

    certificate = certify_scalar_positive_real(model)

    assert not certificate.is_passive
    assert certificate.is_stable
    assert certificate.direct_is_passive
    assert certificate.minimum_real_admittance < -0.9
    assert certificate.minimum_angular_frequency == pytest.approx(centre, abs=2e-6)


def test_large_positive_resonance_cannot_mask_negative_dc_admittance():
    model = _model(
        poles=(-1.0, -1e-12 + 100j, -1e-12 - 100j),
        residues=(-2.0, 1e4, 1e4),
        direct=1.0,
    )

    certificate = certify_scalar_positive_real(model)

    assert not certificate.is_passive
    assert certificate.minimum_real_admittance == pytest.approx(-1.0, abs=1e-10)
    assert certificate.minimum_angular_frequency == pytest.approx(0.0)


def test_global_certificate_rejects_right_half_plane_pole():
    model = _model(poles=(0.1,), residues=(1.0,), direct=1.0)

    certificate = certify_scalar_positive_real(model)

    assert not certificate.is_passive
    assert not certificate.is_stable
    assert "open left half-plane" in certificate.message
    with pytest.raises(ValueError, match="does not relocate unstable poles"):
        repair_scalar_positive_real(model)


def test_global_certificate_rejects_negative_high_frequency_direct_term():
    model = _model(poles=(-1.0,), residues=(2.0,), direct=-0.1)

    certificate = certify_scalar_positive_real(model)

    assert not certificate.is_passive
    assert certificate.is_stable
    assert not certificate.direct_is_passive
    assert certificate.minimum_real_admittance == pytest.approx(-0.1)
    assert np.isinf(certificate.minimum_angular_frequency)


def test_fixed_pole_repair_makes_mild_violation_globally_passive():
    model = _model(poles=(-1.0,), residues=(-0.205,), direct=0.2)
    initial = certify_scalar_positive_real(model, margin=1e-5)
    assert initial.minimum_real_admittance == pytest.approx(-0.005)

    result = repair_scalar_positive_real(
        model,
        margin=1e-5,
        fit_angular_frequencies=np.geomspace(1e-2, 1e2, 101),
    )

    assert result.iterations >= 1
    assert result.certificate.is_passive
    assert result.certificate.minimum_real_admittance >= 1e-5 - 1e-10
    np.testing.assert_array_equal(result.model.poles, model.poles)
    assert result.relative_parameter_change < 0.02


def test_repair_change_metric_is_invariant_to_frequency_units():
    base = _model(poles=(-1.0,), residues=(-0.205,), direct=0.2)
    scaled = _model(poles=(-1e6,), residues=(-0.205e6,), direct=0.2)
    base_result = repair_scalar_positive_real(
        base,
        margin=1e-5,
        fit_angular_frequencies=np.geomspace(1e-2, 1e2, 101),
    )
    scaled_result = repair_scalar_positive_real(
        scaled,
        margin=1e-5,
        fit_angular_frequencies=np.geomspace(1e4, 1e8, 101),
    )

    assert scaled_result.relative_parameter_change == pytest.approx(
        base_result.relative_parameter_change,
        rel=2e-8,
        abs=2e-12,
    )


def test_passive_model_is_returned_without_perturbation():
    model = _model(
        poles=(-1.0, -0.2 + 2j, -0.2 - 2j),
        residues=(0.5, 0.1 + 0.02j, 0.1 - 0.02j),
        direct=0.3,
    )

    result = repair_scalar_positive_real(model)

    assert result.iterations == 0
    assert result.model is model
    assert result.certificate.is_passive

import numpy as np
import pytest

from gprMax.matched_eigenmode_ports import fixed_basis_admittance_samples


def _fixed_power_fixture():
    electric_basis = np.asarray((1.0, 2.0, -1.0, 0.5))
    magnetic_covector = np.asarray((2.0, 1.0, 0.5, 2.0))
    power = electric_basis @ magnetic_covector
    assert power > 0
    return electric_basis, magnetic_covector


def test_fixed_basis_samples_recover_complex_admittance_and_ignore_anchor_gauge():
    electric_basis, magnetic_covector = _fixed_power_fixture()
    voltages = np.asarray((1.0, 0.7 - 0.4j, -1.2 + 0.3j))
    admittances = np.asarray((1.0, 0.82 + 0.11j, 1.17 - 0.08j))
    arbitrary_gauges = np.asarray((2.0j, -0.4 + 1.3j, 3.1 - 0.7j))
    anchor_electric = (
        arbitrary_gauges * voltages
    )[:, None] * electric_basis[None, :]
    anchor_magnetic = (
        arbitrary_gauges * voltages * admittances
    )[:, None] * magnetic_covector[None, :]

    samples = fixed_basis_admittance_samples(
        electric_basis,
        magnetic_covector,
        anchor_electric,
        anchor_magnetic,
    )

    np.testing.assert_allclose(samples.admittances, admittances, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        samples.voltages,
        arbitrary_gauges * voltages,
        rtol=2e-14,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        samples.currents,
        arbitrary_gauges * voltages * admittances,
        rtol=2e-14,
        atol=2e-14,
    )
    np.testing.assert_allclose(samples.electric_residuals, 0.0, atol=2e-16)
    np.testing.assert_allclose(samples.magnetic_residuals, 0.0, atol=2e-16)


def test_fixed_basis_samples_accept_a_tiny_but_well_conditioned_joint_gauge():
    electric_basis, magnetic_covector = _fixed_power_fixture()
    gauge = 1e-80 * np.exp(0.37j)
    expected = 0.83 - 0.12j

    samples = fixed_basis_admittance_samples(
        electric_basis,
        magnetic_covector,
        np.asarray((gauge * electric_basis,)),
        np.asarray((gauge * expected * magnetic_covector,)),
    )

    assert samples.admittances[0] == pytest.approx(expected, rel=2e-14, abs=0.0)


def test_fixed_basis_samples_report_profile_content_outside_scalar_span():
    electric_basis, magnetic_covector = _fixed_power_fixture()
    electric_orthogonal = np.asarray((magnetic_covector[1], -magnetic_covector[0], 0, 0))
    magnetic_orthogonal = np.asarray((electric_basis[1], -electric_basis[0], 0, 0))
    assert electric_orthogonal @ magnetic_covector == pytest.approx(0)
    assert magnetic_orthogonal @ electric_basis == pytest.approx(0)
    voltage = 1.4 - 0.2j
    admittance = 0.91 + 0.07j
    anchor_electric = np.asarray(
        (voltage * electric_basis + 0.15j * electric_orthogonal,)
    )
    anchor_magnetic = np.asarray(
        (voltage * admittance * magnetic_covector - 0.12 * magnetic_orthogonal,)
    )

    samples = fixed_basis_admittance_samples(
        electric_basis,
        magnetic_covector,
        anchor_electric,
        anchor_magnetic,
    )

    np.testing.assert_allclose(samples.admittances, (admittance,), rtol=2e-14, atol=2e-14)
    assert samples.electric_residuals[0] > 0
    assert samples.magnetic_residuals[0] > 0


@pytest.mark.parametrize(
    ("electric_basis", "magnetic_covector", "anchor_electric", "anchor_magnetic", "message"),
    [
        (np.ones((1, 2)), np.ones(2), np.ones((1, 2)), np.ones((1, 2)), "one-dimensional"),
        (np.ones(2), np.ones(3), np.ones((1, 2)), np.ones((1, 2)), "equal-length"),
        (np.ones(2), np.ones(2), np.ones(2), np.ones((1, 2)), "two-dimensional"),
        (np.ones(2), np.ones(2), np.ones((1, 3)), np.ones((1, 3)), "incompatible"),
        (np.ones(2), np.ones(2), np.empty((0, 2)), np.empty((0, 2)), "incompatible"),
    ],
)
def test_fixed_basis_samples_validate_shapes(
    electric_basis,
    magnetic_covector,
    anchor_electric,
    anchor_magnetic,
    message,
):
    with pytest.raises(ValueError, match=message):
        fixed_basis_admittance_samples(
            electric_basis,
            magnetic_covector,
            anchor_electric,
            anchor_magnetic,
        )


def test_fixed_basis_samples_reject_nonpositive_power_pairing():
    with pytest.raises(ValueError, match="positive real power"):
        fixed_basis_admittance_samples(
            np.asarray((1.0, 0.0)),
            np.asarray((-1.0, 0.0)),
            np.asarray(((1.0, 0.0),)),
            np.asarray(((-1.0, 0.0),)),
        )


def test_fixed_basis_samples_reject_zero_projected_voltage():
    electric_basis, magnetic_covector = _fixed_power_fixture()
    electric_orthogonal = np.asarray((magnetic_covector[1], -magnetic_covector[0], 0, 0))
    with pytest.raises(ValueError, match="zero fixed-basis modal voltage"):
        fixed_basis_admittance_samples(
            electric_basis,
            magnetic_covector,
            np.asarray((electric_orthogonal,)),
            np.asarray((magnetic_covector,)),
        )

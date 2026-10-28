"""Real Maxwell CPW fields check identity and the single-mirror limitation."""

import numpy as np
import pytest

from testing.eigenmode_support.identification.cpw_mode_tracking import cpw, track_bank
from testing.eigenmode_support.validate_fdfd_eigenmodes import _configured_solver_constants


@pytest.mark.integration
@pytest.mark.parametrize("seed", (42, 91))
def test_substrate_cpw_tracks_even_and_odd_after_order_and_phase_scrambling(seed):
    with _configured_solver_constants():
        result = track_bank([cpw(f * 1e9) for f in (2, 6, 10)], seed=seed)
    assert result["numerical_valid"]
    assert result["forward_power_valid"]
    assert result["minimum_overlap"] > 0.8
    np.testing.assert_allclose(result["final_parity"], [[1, -1, 1]] * 3, atol=1e-8)
    np.testing.assert_allclose(result["tracked_solver_gap_phase_correlation"], [[1, -1, 1]] * 3, atol=1e-8)
    assert result["automatic_groups"] == ()
    assert result["detected_symmetry"] == "cs_u"
    # Public odd/even channels must actually follow changed candidate numbers.
    mapping = np.asarray(result["candidate_indices"])
    assert np.any(mapping[:, 1:] != mapping[1, 1:])


@pytest.mark.integration
def test_degenerate_cpw_recovers_single_mirror_parity():
    with _configured_solver_constants():
        result = track_bank([cpw(f * 1e9, eps=1.0) for f in (6, 8, 10)], mixed=True)
    assert result["numerical_valid"]
    assert result["forward_power_valid"]
    assert result["minimum_overlap"] > 0.999
    assert result["automatic_groups"] == ((1, 2, 3),)
    assert result["detected_symmetry"] == "cs_u"
    parity = np.asarray(result["final_parity"])
    np.testing.assert_allclose(parity, np.tile(parity[1], (3, 1)), atol=1e-7)
    # The surviving mirror now certifies pure odd/even channels.
    np.testing.assert_allclose(abs(parity), 1.0, atol=1e-8)
    assert result["applied_symmetry"] == ["cs_u"]


@pytest.mark.integration
@pytest.mark.parametrize("seed", (42, 91))
def test_centered_degenerate_cpw_c2v_recovers_odd_even_parity(seed):
    with _configured_solver_constants():
        result = track_bank(
            [cpw(f * 1e9, eps=1.0, height=0.00825, metal_center=0.0) for f in (6, 8, 10)],
            seed=seed,
            mixed=True,
        )
    assert result["numerical_valid"]
    assert result["forward_power_valid"]
    assert result["minimum_overlap"] > 0.999
    assert result["automatic_groups"] == ((1, 2, 3),)
    assert result["detected_symmetry"] == "c2v"
    assert result["applied_symmetry"] == ["c2v"]
    parity = np.asarray(result["final_parity"])
    np.testing.assert_allclose(abs(parity), 1.0, atol=1e-8)
    # Both even channels may mix within their same-symmetry sector, but the
    # odd sector must be separated from them at every anchor.
    np.testing.assert_array_equal(np.sum(parity < 0, axis=1), np.ones(3))
    np.testing.assert_allclose(parity, np.tile(parity[1], (3, 1)), atol=1e-8)

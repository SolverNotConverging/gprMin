import numpy as np
import pytest
from scipy.sparse import block_diag

from gprMax.conformal.operator import CurlOperator


def periodic_operator(n=32):
    curl = np.eye(n) - np.roll(np.eye(n), 1, axis=1)
    return CurlOperator(curl, np.ones(n), np.ones(n))


def test_cfl_bound_does_not_exceed_spectral_limit():
    rng = np.random.default_rng(239)
    for _ in range(20):
        curl = rng.normal(size=(7, 11))
        me = rng.uniform(0.1, 2, 11)
        mh = rng.uniform(0.1, 2, 7)
        operator = CurlOperator(curl, me, mh)
        norm = np.linalg.svd(curl / np.sqrt(mh[:, None] * me[None, :]), compute_uv=False)[0]
        assert operator.cfl_limit <= 2 / norm * (1 + 1e-14)


def test_cfl_bound_preserves_extreme_coefficient_scales():
    for scale in (1e-200, 1e200):
        operator = CurlOperator([[scale]], [1], [1])
        assert operator.cfl_limit / (2 / scale) == pytest.approx(1)


def test_compiled_coefficients_are_detached_and_readonly():
    curl, mass = np.eye(2), np.ones(2)
    operator = CurlOperator(curl, mass, mass)
    curl[:] = 0
    mass[:] = 0
    np.testing.assert_array_equal(operator.curl.toarray(), np.eye(2))
    for array in (operator.curl.data, operator.electric_mass, operator.magnetic_mass):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_unforced_leapfrog_conserves_staggered_energy():
    operator = periodic_operator()
    rng = np.random.default_rng(23)
    electric, magnetic = rng.normal(size=(2, 32))
    dt = 0.8 * operator.cfl_limit
    energy = operator.energy(electric, magnetic, dt)
    for _ in range(1000):
        operator.step(electric, magnetic, dt)
    assert operator.energy(electric, magnetic, dt) == pytest.approx(energy, rel=1e-13)


def test_periodic_yee_mode_has_expected_discrete_frequency():
    n, mode, dt = 32, 3, 0.7
    operator = periodic_operator(n)
    electric = np.cos(2 * np.pi * mode * np.arange(n) / n)
    initial = electric.copy()
    magnetic = np.zeros(n)
    samples = []
    for _ in range(100):
        samples.append(np.dot(initial, electric) / np.dot(initial, initial))
        operator.step(electric, magnetic, dt)
    omega_dt = 2 * np.arcsin(dt * np.sin(np.pi * mode / n))
    # H(-1/2)=0 gives an E cosine offset by half a time step.
    expected = np.cos((np.arange(100) + 0.5) * omega_dt) / np.cos(0.5 * omega_dt)
    np.testing.assert_allclose(samples, expected, atol=2e-14)


def test_enlargement_cancels_internal_flux_and_improves_cfl():
    operator = CurlOperator([[1, -1, 0], [0, 1, -1]], [1, 1, 1], [0.01, 0.99])
    enlarged = operator.aggregate_magnetic([[0, 1]], open_adjacency=[(0, 1)], region_labels=[0, 0])
    np.testing.assert_array_equal(enlarged.operator.curl.toarray(), [[1, 0, -1]])
    np.testing.assert_allclose(enlarged.operator.magnetic_mass, [1])
    assert enlarged.operator.cfl_limit > operator.cfl_limit
    h = np.array([0.7])
    fine_h = enlarged.prolongation @ h
    assert np.dot(operator.magnetic_mass * fine_h, fine_h) == pytest.approx(
        np.dot(enlarged.operator.magnetic_mass * h, h))


def test_enlargement_cannot_cross_sheet_or_unconnected_cells():
    operator = periodic_operator(3)
    with pytest.raises(ValueError, match="boundary sides"):
        operator.aggregate_magnetic([[0, 1], [2]], open_adjacency=[(0, 1)], region_labels=[0, 1, 1])
    with pytest.raises(ValueError, match="open faces"):
        operator.aggregate_magnetic([[0, 1], [2]], open_adjacency=[], region_labels=[0, 0, 0])


def test_separate_opaque_sides_do_not_transmit():
    curl = block_diag([periodic_operator(8).curl, periodic_operator(8).curl])
    operator = CurlOperator(curl, np.ones(16), np.ones(16))
    electric = np.zeros(16)
    electric[2] = 1
    magnetic = np.zeros(16)
    for _ in range(100):
        operator.step(electric, magnetic, 0.5)
    np.testing.assert_array_equal(electric[8:], 0)
    np.testing.assert_array_equal(magnetic[8:], 0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cython_matches_reference(dtype):
    operator = periodic_operator()
    rng = np.random.default_rng(101)
    er, hr = rng.normal(size=(2, 32)).astype(dtype)
    ec, hc = er.copy(), hr.copy()
    for _ in range(100):
        operator.step(er, hr, 0.6)
        operator.step(ec, hc, 0.6, backend="cython")
    tolerance = 2e-5 if dtype == np.float32 else 1e-13
    np.testing.assert_allclose(ec, er, atol=tolerance)
    np.testing.assert_allclose(hc, hr, atol=tolerance)


@pytest.mark.parametrize("dt", [0, -1, np.nan, np.inf, 1, 1.1])
def test_invalid_or_unstable_timestep_is_rejected(dt):
    with pytest.raises(ValueError, match="CFL"):
        periodic_operator().step(np.zeros(32), np.zeros(32), dt)

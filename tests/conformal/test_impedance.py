import numpy as np
import pytest

from gprMax.conformal.impedance import ImpedancePorts, MagneticImpedancePorts
from gprMax.conformal.operator import CurlOperator
from gprMax.impedance_surfaces import SurfaceImpedanceModel


@pytest.mark.parametrize("backend", ["reference", "cython"])
def test_resistive_boundary_matches_midpoint_decay(backend):
    operator = CurlOperator([[0]], [2], [1])
    model = SurfaceImpedanceModel("resistance", D=3)
    ports = ImpedancePorts(operator, [[1]], [4], [model], 0.1)
    electric, magnetic = np.ones(1), np.zeros(1)
    factor = (2 - 0.1 * 4 / 6) / (2 + 0.1 * 4 / 6)
    for _ in range(20):
        old_energy = float(electric[0]**2)
        operator.step(electric, magnetic, 0.1, backend=backend, impedance=ports)
        dissipated = 0.1 * np.dot(ports.weights * ports.last_current, ports.last_midpoint_voltage)
        assert old_energy - electric[0]**2 == pytest.approx(dissipated, abs=1e-14)
    assert electric[0] == pytest.approx(factor**20)


def test_diagonal_trace_damps_only_its_tangent_component():
    operator = CurlOperator([[0, 0, 0]], [1, 1, 1], [1])
    tangent = np.array([1, 2, 3]) / np.sqrt(14)
    ports = ImpedancePorts(operator, tangent[None, :], [1],
                           [SurfaceImpedanceModel("resistance", D=2)], 0.2)
    electric = np.array([2., -1., 0.])  # Orthogonal to the tangential trace.
    original = electric.copy()
    operator.step(electric, np.zeros(1), 0.2, impedance=ports)
    np.testing.assert_allclose(electric, original, atol=1e-15)


def test_sheet_sides_have_independent_ade_states_and_reset():
    operator = CurlOperator(np.zeros((2, 2)), [1, 1], [1, 1])
    model = SurfaceImpedanceModel("relaxation", A=[[-2]], B=[1], C=[1], D=1)
    ports = ImpedancePorts(operator, np.eye(2), [1, 1], [model, model], 0.1)
    electric, magnetic = np.array([1., 0.]), np.zeros(2)
    for _ in range(50):
        operator.step(electric, magnetic, 0.1, impedance=ports)
    assert electric[1] == 0
    assert ports.state[0][0] != 0
    assert ports.state[1][0] == 0
    ports.reset()
    assert all(np.all(state == 0) for state in ports.state)


def test_resistive_port_obeys_discrete_poynting_balance_with_curl():
    operator = CurlOperator([[1, -1], [-1, 1]], [1, 1], [1, 1])
    ports = ImpedancePorts(operator, [[1, 0]], [0.7],
                           [SurfaceImpedanceModel("resistance", D=2)], 0.2)
    electric, magnetic = np.array([1., 0.3]), np.array([0.2, -0.1])
    for _ in range(100):
        energy = operator.energy(electric, magnetic, 0.2)
        operator.step(electric, magnetic, 0.2, impedance=ports)
        dissipation = 0.2 * np.dot(ports.weights * ports.last_current, ports.last_midpoint_voltage)
        assert energy - operator.energy(electric, magnetic, 0.2) == pytest.approx(dissipation, abs=1e-14)


def test_pmc_has_zero_current():
    operator = CurlOperator([[0]], [1], [1])
    ports = ImpedancePorts(operator, [[1]], [1], [SurfaceImpedanceModel("pmc", D=np.inf)], 1)
    electric = np.ones(1)
    operator.step(electric, np.zeros(1), 1, impedance=ports)
    np.testing.assert_array_equal(electric, 1)
    np.testing.assert_array_equal(ports.last_current, 0)


def test_magnetic_bulk_and_surface_loss_share_one_midpoint_solve():
    operator = CurlOperator([[0], [0]], [1], [2, 3])
    trace = np.array([[1, -0.5]])
    bulk = np.array([0.3, 0.8])
    dt = 0.2
    ports = MagneticImpedancePorts(operator, trace, [0.7],
                                   [SurfaceImpedanceModel("resistance", D=2)], dt,
                                   bulk_loss=bulk)
    h = np.array([0.7, -0.3])
    mass = np.diag(operator.magnetic_mass)
    resistance = np.diag(bulk) + trace.T @ np.diag([1.4]) @ trace
    for _ in range(20):
        old = h.copy()
        expected = np.linalg.solve(mass + dt/2*resistance, (mass-dt/2*resistance) @ old)
        ports.correct(old, h)
        np.testing.assert_allclose(h, expected, rtol=1e-14, atol=1e-15)
        midpoint = (h+old)/2
        lost = 0.5*(old @ mass @ old - h @ mass @ h)
        dissipated = dt*(np.dot(bulk*midpoint, midpoint)
                         + np.dot(ports.weights*ports.last_current, ports.last_voltage))
        assert lost == pytest.approx(dissipated, abs=1e-15)


@pytest.mark.parametrize("loss", [[-1], [np.nan], [1, 2]])
def test_magnetic_ports_reject_invalid_bulk_loss(loss):
    operator = CurlOperator([[0]], [1], [1])
    with pytest.raises(ValueError, match="bulk magnetic loss"):
        MagneticImpedancePorts(operator, [[1]], [1],
                               [SurfaceImpedanceModel("resistance", D=1)], 0.1,
                               bulk_loss=loss)

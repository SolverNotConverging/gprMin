from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.conformal.enlargement import plan_enlargement
from gprMax.conformal.operator import CurlOperator
from gprMax.conformal.yee import YeePacking
from gprMax.conformal.topology import MagneticBoundaryPort


def topology(curl, masses, indices=None):
    operator = CurlOperator(curl, np.ones(np.shape(curl)[1]), masses)
    if indices is None:
        indices = [[0, 1, i, 1] for i in range(len(masses))]
    electric_indices = np.array([[1, i, 1, 1] for i in range(operator.curl.shape[1])])
    return SimpleNamespace(packing=YeePacking(operator, electric_indices,
                           np.asarray(indices), np.zeros(operator.curl.shape[1]), np.zeros(len(masses))),
                           diagnostics=(), boundary_ports=())


def test_small_cell_enlarges_and_preserves_requested_timestep():
    cut = topology([[1, -1, 0], [0, 1, -1]], [0.001, 0.999])
    assert cut.packing.operator.cfl_limit < 0.5
    plan = plan_enlargement(cut, 0.5)
    assert plan.accepted
    assert plan.groups == ((0, 1),)
    np.testing.assert_array_equal(plan.operator.curl.toarray(), [[1, 0, -1]])
    assert plan.operator.cfl_limit > 0.5


def test_small_cell_without_open_neighbour_requests_fallback():
    cut = topology([[1, -1, 0, 0], [0, 0, 1, -1]], [0.001, 0.999])
    plan = plan_enlargement(cut, 0.5)
    assert not plan.accepted
    assert plan.reason == "no_compatible_open_neighbour"


def test_opposite_sheet_sides_with_same_dense_index_stay_separate():
    cut = topology([[1, -1, 0, 0], [0, 0, 1, -1]], [0.001, 0.999],
                   [[0, 1, 1, 1], [0, 1, 1, 1]])
    plan = plan_enlargement(cut, 0.5)
    assert not plan.accepted
    assert plan.groups == ((0,), (1,))


def test_different_magnetic_components_cannot_be_merged():
    cut = topology([[1, -1, 0], [0, 1, -1]], [0.001, 0.999],
                   [[0, 1, 1, 1], [1, 1, 1, 1]])
    assert not plan_enlargement(cut, 0.5).accepted


def test_enlargement_transports_bulk_loss_and_independent_sibc_states():
    from gprMax.impedance_surfaces import SurfaceImpedanceModel

    cut = topology([[1, -1, 0], [0, 1, -1]], [0.001, 0.999])
    cut.packing.magnetic_loss[:] = [0.0002, 0.1998]
    cut.boundary_ports = (MagneticBoundaryPort(0, 0.2, "wall", 0),
                          MagneticBoundaryPort(1, 0.3, "wall", 0))
    plan = plan_enlargement(cut, 0.5)
    packing = plan.packing(cut)
    assert packing.magnetic_loss == pytest.approx([0.2])
    original = np.array([2., 3.])
    restricted = plan.restrict_magnetic(original)
    assert np.dot(packing.operator.magnetic_mass, restricted) == pytest.approx(
        np.dot(cut.packing.operator.magnetic_mass, original))
    ports = plan.impedance_ports(cut, {"wall": SurfaceImpedanceModel("wall", D=0.1)}, 0.5)
    assert ports.trace.shape == (2, 1)
    assert len(ports.state) == 2
    e, h = np.array([1., 0., -1.]), restricted.copy()
    for _ in range(20):
        energy = packing.operator.energy(e, h, 0.5)
        old_h = h.copy()
        packing.step(e, h, 0.5, magnetic_impedance=ports)
        midpoint = (h+old_h)/2
        loss = 0.5*(np.dot(packing.magnetic_loss*midpoint, midpoint)
                    + np.dot(ports.weights*ports.last_current, ports.last_voltage))
        assert energy - packing.operator.energy(e, h, 0.5) == pytest.approx(loss, abs=1e-14)


def test_rejected_or_unrelated_plan_cannot_supply_runtime_packing():
    cut = topology([[1, -1]], [0.001])
    plan = plan_enlargement(cut, 0.5)
    with pytest.raises(ValueError, match="requires staircase"):
        plan.packing(cut)
    with pytest.raises(ValueError, match="rejected"):
        plan.restrict_magnetic([1])
    cut = topology([[1, -1, 0], [0, 1, -1]], [0.001, 0.999])
    plan = plan_enlargement(cut, 0.5)
    other = topology([[1, -1, 0], [0, 1, -1]], [0.001, 0.999])
    with pytest.raises(ValueError, match="different topology"):
        plan.packing(other)

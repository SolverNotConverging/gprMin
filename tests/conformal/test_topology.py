from types import SimpleNamespace

import numpy as np
import pytest

import gprMax
from gprMax.conformal.scene_geometry import ContinuousGeometry
from gprMax.conformal.sections import SectionCompiler
from gprMax.conformal.topology import ExpandedTopologyError, compile_topology
from gprMax.conformal.yee import compile_yee, full_yee_measures
from gprMax.materials import Material


def material_grid(size=(4, 4, 4)):
    material = Material(0, "free_space")
    grid = SimpleNamespace(size=np.array(size), nx=size[0], ny=size[1], nz=size[2],
                           dl=np.ones(3), materials=[material],
                           ID=np.zeros((6, *(np.array(size)+1)), dtype=np.uint32), name="main_grid",
                           surface_impedance_models={})
    return grid


def compile_scene(grid, *objects):
    pec = SimpleNamespace(ID="pec", is_pec=True)
    capture_grid = SimpleNamespace(name="main_grid", materials=[*grid.materials, pec],
                                   surface_impedance_models={"copper": object()})
    geometry = ContinuousGeometry.capture(objects, capture_grid)
    scene = geometry.compile(domain=grid.size*grid.dl, dl=grid.dl)
    compiler = SectionCompiler(scene, surface_error=0.01)
    return compile_topology(grid, compiler, epsilon0=1, mu0=1)


def test_empty_geometry_recovers_integral_yee_operator():
    grid = material_grid()
    topology = compile_scene(grid)
    reference = compile_yee(grid, *full_yee_measures(grid), epsilon0=1, mu0=1)
    np.testing.assert_array_equal(topology.packing.electric_indices, reference.electric_indices)
    np.testing.assert_array_equal(topology.packing.magnetic_indices, reference.magnetic_indices)
    np.testing.assert_allclose(topology.packing.operator.curl.toarray(), reference.operator.curl.toarray())
    np.testing.assert_allclose(topology.packing.operator.electric_mass, reference.operator.electric_mass)
    np.testing.assert_allclose(topology.packing.operator.magnetic_mass, reference.operator.magnetic_mass)


def test_off_grid_opaque_sheet_allocates_extra_fields_and_isolates_sides():
    grid = material_grid()
    topology = compile_scene(grid, gprMax.Plate(p1=[1.4, 0, 0], p2=[1.4, 4, 4], material_id="pec"))
    assert topology.extra_electric_dofs > 0
    assert topology.extra_magnetic_dofs > 0
    packing = topology.packing
    positions = packing.electric_indices[:, 1:].astype(float)
    for i, component in enumerate(packing.electric_indices[:, 0]):
        positions[i, component] += np.mean(topology.electric_intervals[i])
    left, right = positions[:, 0] < 1.4, positions[:, 0] > 1.4
    rng = np.random.default_rng(21)
    electric = rng.normal(size=len(positions)) * left
    magnetic = np.zeros(len(packing.magnetic_indices))
    dt = 0.8 * packing.operator.cfl_limit
    energy = packing.operator.energy(electric, magnetic, dt)
    for _ in range(200):
        packing.step(electric, magnetic, dt)
    np.testing.assert_array_equal(electric[right], 0)
    assert packing.operator.energy(electric, magnetic, dt) == pytest.approx(energy, rel=1e-13)


def test_conductor_volume_removes_only_physical_exterior_dofs():
    grid = material_grid()
    topology = compile_scene(grid, gprMax.Box(p1=[0, 0, 0], p2=[1.4, 4, 4], material_id="pec"))
    indices = topology.packing.electric_indices
    starts = indices[:, 1].astype(float)
    starts[indices[:, 0] == 0] += topology.electric_intervals[indices[:, 0] == 0, 0]
    assert np.all(starts >= 1.4)
    assert not topology.extra_electric_dofs
    assert not topology.extra_magnetic_dofs


def test_partial_sheet_tip_does_not_become_an_extended_wall():
    grid = material_grid()
    with pytest.raises(ExpandedTopologyError, match="finite_sheet_tip"):
        compile_scene(grid, gprMax.Plate(p1=[1.4, 0.3, 0], p2=[1.4, 2.7, 4], material_id="pec"))


@pytest.mark.parametrize("backend", ["reference", "cython"])
def test_two_sided_sibc_sheet_has_separate_ports_and_correct_power_loss(backend):
    from gprMax.impedance_surfaces import SurfaceImpedanceModel

    grid = material_grid()
    topology = compile_scene(grid, gprMax.Plate(p1=[1.4, 0, 0], p2=[1.4, 4, 4], material_id="copper"))
    assert topology.extra_magnetic_dofs > 0
    assert not topology.diagnostics
    assert topology.boundary_ports
    packing = topology.packing
    dt = 0.7 * packing.operator.cfl_limit
    ports = topology.impedance_ports({"copper": SurfaceImpedanceModel("copper", D=0.1)}, dt)
    rng = np.random.default_rng(509)
    electric = rng.normal(size=len(packing.electric_indices))
    magnetic = rng.normal(size=len(packing.magnetic_indices))
    for _ in range(30):
        energy = packing.operator.energy(electric, magnetic, dt)
        packing.step(electric, magnetic, dt, backend=backend, magnetic_impedance=ports)
        loss = dt * np.dot(ports.weights*ports.last_current, ports.last_voltage)
        assert energy-packing.operator.energy(electric, magnetic, dt) == pytest.approx(loss, abs=1e-12)


@pytest.mark.parametrize("boundary", ["pec", "copper"])
def test_diagonal_infinitesimal_boundary_isolates_its_two_sides(boundary):
    from gprMax.impedance_surfaces import SurfaceImpedanceModel

    grid = material_grid()
    # The plane x+y=4.4 spans this domain. Its finite triangle rim lies
    # outside the domain, so this exercises the sheet rather than its tips.
    topology = compile_scene(grid, gprMax.Triangle(
        p1=[-10, 14.4, -10], p2=[14.4, -10, -10], p3=[2.2, 2.2, 20],
        thickness=0, material_id=boundary))
    assert not topology.diagnostics
    assert topology.extra_electric_dofs > 0
    assert topology.extra_magnetic_dofs > 0
    packing = topology.packing
    positions = packing.electric_indices[:, 1:].astype(float)
    for row, component in enumerate(packing.electric_indices[:, 0]):
        positions[row, component] += np.mean(topology.electric_intervals[row])
    left = positions[:, 0] + positions[:, 1] < 4.4
    electric = np.random.default_rng(429).normal(size=len(positions)) * left
    magnetic = np.zeros(len(packing.magnetic_indices))
    dt = 0.6 * packing.operator.cfl_limit
    model = SurfaceImpedanceModel("copper", A=[[-2]], B=[1], C=[0.1], D=0.2)
    ports = topology.impedance_ports({"copper": model}, dt)
    for _ in range(200):
        packing.step(electric, magnetic, dt, magnetic_impedance=ports)
    np.testing.assert_array_equal(electric[~left], 0)
    assert np.all(np.isfinite(electric))
    if ports is not None:
        assert any(np.any(state != 0) for state in ports.state)
        assert any(np.all(state == 0) for state in ports.state)

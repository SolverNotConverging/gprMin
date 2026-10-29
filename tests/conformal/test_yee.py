import numpy as np
import pytest

import gprMax
import gprMax.config as config
from gprMax.conformal.yee import compile_yee, full_yee_measures
from gprMax.cython.fields_updates_normal import update_electric, update_magnetic
from gprMax.grid.fdtd_grid import FDTDGrid


@pytest.fixture
def averaged_grid(tmp_path, monkeypatch):
    captured = []
    original = FDTDGrid.build

    def capture(grid):
        original(grid)
        captured.append(grid)

    monkeypatch.setattr(FDTDGrid, "build", capture)
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=(0.008, 0.010, 0.012)),
        gprMax.Discretisation(p1=(0.001, 0.001, 0.001)),
        gprMax.PMLThickness(thickness=0), gprMax.OMPThreads(1),
        gprMax.TimeStepStabilityFactor(f=0.5), gprMax.TimeWindow(iterations=2),
        gprMax.Material(er=4.4, se=0.005, mr=1.3, sm=0.01, id="dielectric"),
        gprMax.Box(p1=(0.002, 0.002, 0.002), p2=(0.006, 0.008, 0.010),
                   material_id="dielectric", averaging="y"),
    ):
        scene.add(obj)
    gprMax.run(scenes=[scene], geometry_only=True, outputfile=tmp_path / "averaged",
               hide_progress_bars=True, log_level=40, cpu_precision="double")
    return captured[0]


def compile_grid(grid):
    return compile_yee(grid, *full_yee_measures(grid),
                       epsilon0=config.sim_config.em_consts["e0"], mu0=config.sim_config.em_consts["m0"])


@pytest.mark.parametrize("backend", ["reference", "cython"])
def test_averaged_dielectric_and_losses_match_existing_yee_update(averaged_grid, backend):
    grid = averaged_grid
    packed = compile_grid(grid)
    # Confirm this is exercising averaged interface materials, not only the
    # original homogeneous dielectric and free-space records.
    assert any(1 < material.er < 4.4 for material in grid.materials)
    rng = np.random.default_rng(44005)
    electric = rng.normal(size=len(packed.electric_indices))
    magnetic = rng.normal(size=len(packed.magnetic_indices)) / config.sim_config.em_consts["z0"]
    packed.scatter(grid, electric, magnetic)
    for _ in range(20):
        update_magnetic(grid.nx, grid.ny, grid.nz, -1, 1, grid.updatecoeffsH, grid.ID,
                        grid.Ex, grid.Ey, grid.Ez, grid.Hx, grid.Hy, grid.Hz)
        update_electric(grid.nx, grid.ny, grid.nz, -1, 1, grid.updatecoeffsE, grid.ID,
                        grid.Ex, grid.Ey, grid.Ez, grid.Hx, grid.Hy, grid.Hz)
        packed.step(electric, magnetic, grid.dt, backend=backend)
    expected_e, expected_h = packed.gather(grid)
    np.testing.assert_allclose(electric, expected_e, atol=3e-14, rtol=1e-12)
    np.testing.assert_allclose(magnetic, expected_h, atol=1e-16, rtol=1e-12)


def test_cut_measures_change_geometry_without_reaveraging_dielectric(averaged_grid):
    grid = averaged_grid
    lengths, areas = full_yee_measures(grid)
    original_ids = grid.ID.copy()
    original_coefficients = grid.updatecoeffsE.copy()
    base = compile_grid(grid)
    edge = tuple(base.electric_indices[50])
    lengths[edge] *= 0.6
    cut = compile_yee(grid, lengths, areas, epsilon0=config.sim_config.em_consts["e0"],
                      mu0=config.sim_config.em_consts["m0"])
    assert cut.operator.electric_mass[50] == pytest.approx(base.operator.electric_mass[50] * 0.6)
    assert cut.electric_loss[50] == pytest.approx(base.electric_loss[50] * 0.6)
    np.testing.assert_array_equal(grid.ID, original_ids)
    np.testing.assert_array_equal(grid.updatecoeffsE, original_coefficients)


def test_zero_measures_remove_conductor_interior_dofs(averaged_grid):
    grid = averaged_grid
    lengths, areas = full_yee_measures(grid)
    base = compile_grid(grid)
    edge, face = tuple(base.electric_indices[50]), tuple(base.magnetic_indices[50])
    lengths[edge] = 0
    areas[face] = 0
    cut = compile_yee(grid, lengths, areas, epsilon0=config.sim_config.em_consts["e0"],
                      mu0=config.sim_config.em_consts["m0"])
    assert len(cut.electric_indices) == len(base.electric_indices)-1
    assert len(cut.magnetic_indices) == len(base.magnetic_indices)-1


@pytest.mark.parametrize("boundary", ["pec", "copper"])
def test_physical_conductor_cut_preserves_averaged_dielectric(averaged_grid, boundary):
    from types import SimpleNamespace
    from gprMax.conformal.scene_geometry import ContinuousGeometry
    from gprMax.conformal.sections import SectionCompiler
    from gprMax.conformal.topology import compile_topology
    from gprMax.impedance_surfaces import SurfaceImpedanceModel

    grid = averaged_grid
    ids = grid.ID.copy()
    coefficients = grid.updatecoeffsE.copy()
    capture_grid = SimpleNamespace(name=grid.name, materials=grid.materials,
                                   surface_impedance_models={"copper": object()})
    geometry = ContinuousGeometry.capture([
        gprMax.Box(p1=(0.002, 0.002, 0.002), p2=(0.006, 0.008, 0.010),
                   material_id="dielectric", averaging="y"),
        gprMax.Plate(p1=(0.0034, 0, 0), p2=(0.0034, 0.010, 0.012), material_id=boundary),
    ], capture_grid)
    compiler = SectionCompiler(geometry.compile(domain=np.asarray(grid.size)*grid.dl, dl=grid.dl),
                               surface_error=1e-5)
    topology = compile_topology(grid, compiler, epsilon0=config.sim_config.em_consts["e0"],
                                mu0=config.sim_config.em_consts["m0"])
    packing = topology.packing
    volume = np.prod(grid.dl)
    averaged_count = 0
    for row, index in enumerate(packing.electric_indices):
        material = grid.materials[int(grid.ID[tuple(index)])]
        fraction = np.diff(topology.electric_intervals[row]).item()
        assert packing.operator.electric_mass[row] == pytest.approx(
            config.sim_config.em_consts["e0"] * material.er * fraction * volume, rel=1e-12, abs=0)
        assert packing.electric_loss[row] == pytest.approx(material.se * fraction * volume, rel=1e-12, abs=0)
        averaged_count += 1 < material.er < 4.4
    assert averaged_count > 0
    assert topology.extra_electric_dofs > 0
    assert topology.extra_magnetic_dofs > 0
    assert bool(topology.boundary_ports) == (boundary == "copper")
    np.testing.assert_array_equal(grid.ID, ids)
    np.testing.assert_array_equal(grid.updatecoeffsE, coefficients)

    # Both backends must advance the same averaged lossy bulk alongside the
    # split boundary fields, including simultaneous magnetic bulk/SIBC loss.
    dt = 0.5 * packing.operator.cfl_limit
    rng = np.random.default_rng(440056)
    e = rng.normal(size=len(packing.electric_indices))
    h = rng.normal(size=len(packing.magnetic_indices)) / config.sim_config.em_consts["z0"]
    results = []
    for backend in ("reference", "cython"):
        electric, magnetic = e.copy(), h.copy()
        ports = topology.impedance_ports({"copper": SurfaceImpedanceModel("copper", D=0.1)}, dt)
        for _ in range(10):
            packing.step(electric, magnetic, dt, backend=backend, magnetic_impedance=ports)
        results.append((electric, magnetic))
    for reference, actual in zip(*results):
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-14)

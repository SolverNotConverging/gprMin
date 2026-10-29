from types import SimpleNamespace
from copy import deepcopy
import pickle

import numpy as np
import pytest

import gprMax
from gprMax.conformal.scene_geometry import ContinuousGeometry, UnsupportedGeometry


def geometry(*objects):
    grid = SimpleNamespace(name="main_grid", materials=[
        SimpleNamespace(ID="pec", is_pec=True),
        SimpleNamespace(ID="free_space", is_pec=False),
    ], surface_impedance_models={"copper": object()})
    return ContinuousGeometry.capture(objects, grid)


def compile_geometry(*objects):
    return geometry(*objects).compile(domain=[1, 1, 1], dl=[0.1, 0.1, 0.1])


def test_capture_preserves_subcell_coordinates_and_detaches_scene():
    point = [0.1234, 0.2345, 0.3456]
    obj = gprMax.Sphere(p1=point, r=0.0123, material_id="copper")
    captured = geometry(obj)
    digest = captured.digest
    point[0] = 0.9
    obj.kwargs["r"] = 0.8
    assert captured.records[0].parameters["p1"][0] == 0.1234
    assert captured.records[0].parameters["r"] == 0.0123
    assert captured.digest == digest
    assert captured.records[0].boundary_kind == "sibc"
    assert deepcopy(captured) is captured
    transferred = pickle.loads(pickle.dumps(captured))
    assert transferred.digest == captured.digest
    with pytest.raises(TypeError):
        transferred.records[0].parameters["r"] = 1


def test_continuous_order_removes_old_conductor():
    scene = compile_geometry(
        gprMax.Box(p1=[0, 0, 0], p2=[1, 1, 1], material_id="pec"),
        gprMax.Sphere(p1=[0.5, 0.5, 0.5], r=0.2, material_id="free_space"),
    )
    np.testing.assert_array_equal(scene.owners([[0.1, 0.5, 0.5], [0.5, 0.5, 0.5]]), [0, 1])
    regions, sheets = scene.partition_segment([0, 0.5, 0.5], [1, 0.5, 0.5])
    assert [region.material_id for region in regions] == ["pec", "free_space", "pec"]
    np.testing.assert_allclose([(r.start, r.stop) for r in regions], [[0, 0.3], [0.3, 0.7], [0.7, 1]])
    assert not sheets


@pytest.mark.parametrize("shape", [
    gprMax.Box(p1=[0.3, 0.3, 0.3], p2=[0.7, 0.7, 0.7], material_id="pec"),
    gprMax.Sphere(p1=[0.5]*3, r=0.2, material_id="pec"),
    gprMax.Ellipsoid(p1=[0.5]*3, xr=0.2, yr=0.3, zr=0.4, material_id="pec"),
    gprMax.Cylinder(p1=[0.5, 0.5, 0], p2=[0.5, 0.5, 1], r=0.2, material_id="pec"),
])
def test_exact_primitive_edge_intersections(shape):
    primitive, = compile_geometry(shape).primitives
    assert primitive.segment_interval([0, 0.5, 0.5], [1, 0.5, 0.5]) == pytest.approx((0.3, 0.7))
    assert primitive.segment_interval([0, 2, 0.5], [1, 2, 0.5]) is None


def test_diagonal_cylinder_is_not_snapped():
    primitive, = compile_geometry(gprMax.Cylinder(
        p1=[0.1]*3, p2=[0.9]*3, r=0.05, material_id="copper")).primitives
    assert primitive.segment_interval([0, 0, 0], [1, 1, 1]) == pytest.approx((0.1, 0.9))
    np.testing.assert_allclose(primitive.axis, np.ones(3) / np.sqrt(3))


def test_infinitesimal_plate_is_boundary_crossing_not_volume():
    scene = compile_geometry(gprMax.Plate(p1=[0.1234, 0, 0], p2=[0.1234, 1, 1], material_id="copper"))
    regions, sheets = scene.partition_segment([0, 0.5, 0.5], [1, 0.5, 0.5])
    assert len(regions) == 2
    assert all(region.boundary_kind == "bulk" for region in regions)
    assert sheets == ((0, (0.1234, 0.1234)),)


def test_diagonal_triangle_sheet_intersection():
    scene = compile_geometry(gprMax.Triangle(
        p1=[0, 0, 0], p2=[1, 0, 1], p3=[0, 1, 1], thickness=0, material_id="copper"))
    _, sheets = scene.partition_segment([0.2, 0.2, 0], [0.2, 0.2, 1])
    assert len(sheets) == 1
    assert sheets[0][1] == pytest.approx((0.4, 0.4))
    assert not scene.partition_segment([0.8, 0.8, 0], [0.8, 0.8, 1])[1]


def test_later_volume_overwrites_sheet_at_physical_crossing():
    scene = compile_geometry(
        gprMax.Plate(p1=[0.45, 0, 0], p2=[0.45, 1, 1], material_id="copper"),
        gprMax.Box(p1=[0.4, 0.4, 0.4], p2=[0.6, 0.6, 0.6], material_id="free_space"),
    )
    assert not scene.partition_segment([0, 0.5, 0.5], [1, 0.5, 0.5])[1]
    assert scene.partition_segment([0, 0.2, 0.5], [1, 0.2, 0.5])[1]


def test_unsupported_geometry_cannot_silently_expose_earlier_surfaces():
    scene = compile_geometry(gprMax.Cone(
        p1=[0, 0, 0], p2=[1, 1, 1], r1=0.1, r2=0.2, material_id="pec"))
    with pytest.raises(UnsupportedGeometry, match="#cone"):
        scene.owners([[0, 0, 0]])


def test_real_scene_build_retains_physical_geometry(tmp_path, monkeypatch):
    from gprMax.grid.fdtd_grid import FDTDGrid
    captured = []
    original = FDTDGrid.build

    def build(grid):
        captured.append(grid.continuous_geometry)
        original(grid)

    monkeypatch.setattr(FDTDGrid, "build", build)
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=[0.01]*3), gprMax.Discretisation(p1=[0.001]*3),
        gprMax.TimeWindow(iterations=2), gprMax.PMLThickness(thickness=0),
        gprMax.OMPThreads(1),
        gprMax.Sphere(p1=[0.0053]*3, r=0.0022, material_id="pec"),
    ):
        scene.add(obj)
    gprMax.run(scenes=[scene], geometry_only=True, outputfile=tmp_path / "geometry",
               hide_progress_bars=True, log_level=40)
    assert captured[0].records[0].parameters["p1"] == (0.0053,)*3

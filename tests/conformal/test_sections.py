import numpy as np
import pytest

import gprMax
from gprMax.conformal.sections import SectionCompiler
from tests.conformal.test_scene_geometry import compile_geometry


SQUARE = np.array([[0, 0, 0.5], [1, 0, 0.5], [1, 1, 0.5], [0, 1, 0.5]])


def section(*objects, vertices=SQUARE, error=0.002):
    return SectionCompiler(compile_geometry(*objects), surface_error=error).section(vertices)


def test_box_section_preserves_conductor_hole_and_exact_area():
    result = section(gprMax.Box(p1=[0.2, 0.3, 0], p2=[0.8, 0.7, 1], material_id="pec"))
    assert result.total_area == pytest.approx(1)
    assert result.retained_area == pytest.approx(1 - 0.6 * 0.4)


def test_dielectric_overwrite_preserves_bulk_material_ownership():
    result = section(
        gprMax.Box(p1=[0, 0, 0], p2=[1, 1, 1], material_id="pec"),
        gprMax.Box(p1=[0.2, 0.3, 0], p2=[0.8, 0.7, 1], material_id="free_space"),
    )
    assert result.total_area == pytest.approx(1)
    assert result.retained_area == pytest.approx(0.6 * 0.4)
    assert all(piece.material_id == "free_space" for piece in result.pieces if piece.boundary_kind == "bulk")


def test_sphere_section_meets_geometric_area_error_budget():
    radius, error = 0.3, 0.001
    result = section(gprMax.Sphere(p1=[0.5]*3, r=radius, material_id="pec"), error=error)
    removed = 1 - result.retained_area
    assert np.pi * (radius - error)**2 <= removed <= np.pi * radius**2
    assert result.total_area == pytest.approx(1, abs=1e-12)


def test_cylinder_section_meets_geometric_area_error_budget():
    radius, error = 0.3, 0.001
    result = section(gprMax.Cylinder(p1=[0.5, 0.5, 0], p2=[0.5, 0.5, 1],
                                    r=radius, material_id="copper"), error=error)
    removed = 1 - result.retained_area
    assert np.pi * (radius - error)**2 <= removed <= np.pi * radius**2


def test_diagonal_sheet_has_two_sides_and_zero_volume():
    result = section(gprMax.Triangle(p1=[0, 0, 0], p2=[2, 0, 2], p3=[0, 2, 2],
                                     thickness=0, material_id="copper"))
    assert result.total_area == pytest.approx(1)
    assert result.retained_area == pytest.approx(1)
    assert len(result.sheets) == 1
    assert {p.sides for p in result.pieces} == {((0, "negative"),), ((0, "positive"),)}
    assert np.linalg.norm(result.sheets[0].stop - result.sheets[0].start) == pytest.approx(np.sqrt(0.5))


def test_finite_sheet_tip_is_retained_without_extending_opaque_segment():
    result = section(gprMax.Plate(p1=[0.4, 0.2, 0], p2=[0.4, 0.7, 1], material_id="pec"))
    sheet, = result.sheets
    assert np.linalg.norm(sheet.stop - sheet.start) == pytest.approx(0.5)
    assert result.retained_area == pytest.approx(1)


def test_later_volume_removes_only_overwritten_sheet_segment():
    result = section(
        gprMax.Plate(p1=[0.4, 0, 0], p2=[0.4, 1, 1], material_id="pec"),
        gprMax.Box(p1=[0.3, 0.3, 0], p2=[0.5, 0.6, 1], material_id="free_space"),
    )
    assert sum(np.linalg.norm(s.stop-s.start) for s in result.sheets) == pytest.approx(0.7)
    assert result.retained_area == pytest.approx(1)


def test_coplanar_sheet_blocks_normal_magnetic_section():
    result = section(gprMax.Plate(p1=[0.2, 0.3, 0.5], p2=[0.8, 0.7, 0.5], material_id="pec"))
    assert result.retained_area == pytest.approx(1-0.6*0.4)


@pytest.mark.parametrize("vertices", [
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.1]],
    [[0, 0, 0], [1, 0, 0], [0.2, 0.2, 0], [0, 1, 0]],
])
def test_invalid_section_polygons_are_rejected(vertices):
    with pytest.raises(ValueError):
        section(vertices=vertices)

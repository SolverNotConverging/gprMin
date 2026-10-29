"""Geometric conservation checks independent of the time-stepper."""

import numpy as np
import pytest

from gprMax.conformal.geometry import Boundary, UnresolvedGeometry, box_region, cut_cell


def assert_closed(region):
    area = sum(face.area for face in region.faces)
    np.testing.assert_allclose(region.closure_error, 0, atol=area * 1e-12)
    assert region.volume > 0


def test_box_orientation_and_translation():
    region = box_region([123, -42, 17], [125, -39, 21])
    assert region.volume == pytest.approx(24)
    assert sorted(face.area for face in region.faces) == [6, 6, 8, 8, 12, 12]
    assert_closed(region)


@pytest.mark.parametrize("kind", ["pec", "sibc"])
def test_infinitesimal_diagonal_sheet_keeps_independent_sides(kind):
    wall = Boundary("sheet", [1, 1, 0], 1, kind, "both",
                    "copper" if kind == "sibc" else None)
    regions = cut_cell([0, 0, 0], [1, 1, 1], [wall])
    assert len(regions) == 2
    assert regions[0].sides != regions[1].sides
    caps = []
    for region in regions:
        assert region.volume == pytest.approx(0.5)
        assert_closed(region)
        cap, = [face for face in region.faces if face.wall]
        assert cap.area == pytest.approx(np.sqrt(2))
        assert cap.wall.kind == kind
        assert cap.wall.side == region.sides[0][1]
        caps.append(cap)
    np.testing.assert_allclose(caps[0].area_vector, -caps[1].area_vector)
    assert caps[0].wall != caps[1].wall


def test_off_grid_sheet_does_not_acquire_voxel_thickness():
    regions = cut_cell([0, 0, 0], [1, 1, 1],
                       [Boundary("foil", [1, 0, 0], 0.12345, keep="both")])
    assert [region.volume for region in regions] == pytest.approx([0.12345, 0.87655])
    for region in regions:
        cap, = [face for face in region.faces if face.wall]
        np.testing.assert_allclose(cap.vertices[:, 0], 0.12345)


def test_small_cell_is_retained_for_later_enlargement():
    region, = cut_cell([0, 0, 0], [1, 1, 1],
                       [Boundary("metal", [1, 0, 0], 1e-8, keep="negative")])
    assert region.volume == pytest.approx(1e-8)
    assert_closed(region)


def test_thin_gap_keeps_both_boundaries_for_topology_decision():
    region, = cut_cell([0, 0, 0], [1, 1, 1], [
        Boundary("left", [1, 0, 0], 0.4999, keep="positive"),
        Boundary("right", [1, 0, 0], 0.5001, keep="negative"),
    ])
    assert region.volume == pytest.approx(0.0002)
    assert {face.wall.boundary_id for face in region.faces if face.wall} == {"left", "right"}
    assert_closed(region)


def test_random_sheet_cuts_conserve_volume_and_close_contours():
    rng = np.random.default_rng(4701)
    for _ in range(100):
        normal = rng.normal(size=3)
        point = rng.uniform(0.05, 0.95, size=3)
        regions = cut_cell([0, 0, 0], [1, 1, 1], [
            Boundary("sheet", normal, np.dot(point, normal), keep="both")])
        assert len(regions) == 2
        assert sum(region.volume for region in regions) == pytest.approx(1, abs=1e-12)
        for region in regions:
            assert_closed(region)


def test_intersecting_sheets_have_four_independent_regions():
    regions = cut_cell([0, 0, 0], [1, 1, 1], [
        Boundary("x", [1, 0, 0], 0.5, keep="both"),
        Boundary("y", [0, 1, 0], 0.5, keep="both"),
    ])
    assert len(regions) == 4
    assert len({region.sides for region in regions}) == 4
    for region in regions:
        assert region.volume == pytest.approx(0.25)
        assert_closed(region)


@pytest.mark.parametrize("scale", [1e-6, 1, 1e6])
def test_geometric_scale_invariance(scale):
    regions = cut_cell([0, 0, 0], np.ones(3) * scale, [
        Boundary("sheet", [1, 2, 3], 2.3 * scale, keep="both")])
    assert sum(region.volume for region in regions) / scale**3 == pytest.approx(1)
    for region in regions:
        assert_closed(region)


def test_grid_aligned_boundary_is_not_duplicated():
    regions = cut_cell([0, 0, 0], [1, 1, 1], [
        Boundary("wall", [1, 0, 0], 0, keep="both")])
    assert len(regions) == 1
    assert len(regions[0].faces) == 6
    cap, = [face for face in regions[0].faces if face.wall]
    assert cap.wall.side == "positive"
    assert cap.area == pytest.approx(1)


def test_fully_covered_cell_has_no_retained_region():
    assert not cut_cell([0, 0, 0], [1, 1, 1], [Boundary("wall", [1, 0, 0], 2)])


def test_coincident_boundaries_are_explicitly_unresolved():
    with pytest.raises(UnresolvedGeometry, match="coincident"):
        cut_cell([0, 0, 0], [1, 1, 1], [
            Boundary("a", [1, 0, 0], 0.5),
            Boundary("b", [1, 0, 0], 0.5),
        ])


@pytest.mark.parametrize("kwargs", [
    {"normal": [0, 0, 0]}, {"normal": [0, np.nan, 0]},
    {"offset": np.inf}, {"kind": "dielectric"}, {"keep": "outside"},
    {"kind": "sibc"}, {"impedance_id": "copper"}, {"identifier": ""},
])
def test_invalid_boundaries_rejected(kwargs):
    args = dict(identifier="wall", normal=[1, 0, 0], offset=0)
    args.update(kwargs)
    with pytest.raises(ValueError):
        Boundary(**args)

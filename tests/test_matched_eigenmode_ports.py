from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.matched_eigenmode_ports import MatchedEigenmodeBoundary


def _location_boundary(*, direction, plane, depth, pml=None):
    boundary = MatchedEigenmodeBoundary.__new__(MatchedEigenmodeBoundary)
    boundary.normal_axis = 0
    boundary.direction_sign = 1 if direction == "+" else -1
    boundary.boundary_index = 0 if direction == "+" else 30
    boundary.expansion_plane_index = plane
    boundary.depth_cells = depth
    boundary.owner = SimpleNamespace(
        port_index=1,
        transverse_axes=(1, 2),
        transverse_start=(0, 0),
        transverse_stop=(20, 12),
    )
    thickness = {
        "x0": 0,
        "y0": 0,
        "z0": 0,
        "xmax": 0,
        "ymax": 0,
        "zmax": 0,
    }
    thickness.update(pml or {})
    grid = SimpleNamespace(
        size=np.asarray((30, 20, 12)),
        pmls={"thickness": thickness},
        symmetry_boundaries={},
    )
    return boundary, grid


@pytest.mark.parametrize(
    ("direction", "plane", "depth", "pml"),
    [
        ("+", 25, 25, {"xmax": 5}),
        ("-", 5, 25, {"x0": 5}),
        ("+", 29, 29, {}),
        ("-", 1, 29, {}),
    ],
)
def test_location_accepts_opposite_pml_interface_and_interior_limit(
    direction,
    plane,
    depth,
    pml,
):
    boundary, grid = _location_boundary(
        direction=direction,
        plane=plane,
        depth=depth,
        pml=pml,
    )

    boundary._validate_location(grid)


@pytest.mark.parametrize(
    ("direction", "plane", "depth", "pml", "message"),
    [
        ("+", 26, 26, {"xmax": 5}, "opposite longitudinal PML slab"),
        ("-", 4, 26, {"x0": 5}, "opposite longitudinal PML slab"),
        ("+", 30, 30, {}, "reference plane to be strictly inside"),
        ("-", 0, 30, {}, "reference plane to be strictly inside"),
    ],
)
def test_location_rejects_opposite_pml_and_outer_reference_plane(
    direction,
    plane,
    depth,
    pml,
    message,
):
    boundary, grid = _location_boundary(
        direction=direction,
        plane=plane,
        depth=depth,
        pml=pml,
    )

    with pytest.raises(ValueError, match=message):
        boundary._validate_location(grid)


@pytest.mark.parametrize(
    ("direction", "boundary_index", "plane", "expected_half_cell_planes"),
    [
        ("+", 0, 2, (0, 1)),
        ("-", 4, 2, (2, 3)),
    ],
)
def test_uniform_section_checks_only_buffer_side_half_cells(
    direction,
    boundary_index,
    plane,
    expected_half_cell_planes,
):
    boundary = MatchedEigenmodeBoundary.__new__(MatchedEigenmodeBoundary)
    boundary.normal_axis = 0
    boundary.direction_sign = 1 if direction == "+" else -1
    boundary.boundary_index = boundary_index
    boundary.expansion_plane_index = plane
    boundary.owner = SimpleNamespace(
        port_index=1,
        transverse_axes=(1, 2),
        transverse_start=(0, 0),
        transverse_stop=(1, 1),
    )
    calls = []

    def component_view(array, local_axis, field_kind, plane_index):
        calls.append((local_axis, field_kind, plane_index))
        return np.zeros((1, 1), dtype=np.uint32)

    boundary._local_component_view = component_view
    material = SimpleNamespace(
        numID=0,
        ID="free_space",
        type="",
        thin_wire_axis=None,
        poles=0,
        er=1.0,
        mr=1.0,
        se=0.0,
        sm=0.0,
    )
    grid = SimpleNamespace(
        size=np.asarray((4, 1, 1)),
        solid=np.zeros((4, 1, 1), dtype=np.uint32),
        ID=[np.zeros((1, 1, 1), dtype=np.uint32) for _ in range(6)],
        materials=[material],
    )

    boundary._validate_uniform_section(grid)

    half_cell_components = {(2, "E"), (0, "H"), (1, "H")}
    for component in half_cell_components:
        assert tuple(
            call[2] for call in calls if call[:2] == component
        ) == expected_half_cell_planes

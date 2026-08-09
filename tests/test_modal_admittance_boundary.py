import logging
from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
from gprMax.matched_eigenmode_ports import MatchedEigenmodeBoundary
from gprMax.matched_eigenmode_ports import constant_modal_admittance_step


def test_constant_modal_admittance_step_satisfies_discrete_energy_identity():
    rng = np.random.default_rng(1731)
    previous = rng.standard_normal(256)
    magnetic = rng.standard_normal(256)
    incident = rng.standard_normal(256)
    ratio = np.geomspace(1e-3, 1e3, previous.size)

    following = constant_modal_admittance_step(
        previous,
        magnetic,
        incident,
        ratio,
    )
    centred_voltage = 0.5 * (following + previous)
    left = (
        0.5 * ratio * (following**2 - previous**2)
        + centred_voltage * magnetic
        + (centred_voltage - incident) ** 2
    )
    right = incident**2

    np.testing.assert_allclose(left, right, rtol=3e-12, atol=3e-12)


def test_constant_modal_admittance_step_broadcasts_modal_inputs():
    previous = np.asarray(((1.0,), (-2.0,)))
    magnetic = np.asarray((0.25, -0.5, 1.5))
    incident = 0.75
    ratio = np.asarray((0.1, 1.0, 10.0))
    expected = (
        (ratio - 0.5) * previous
        + 2 * incident
        - magnetic
    ) / (ratio + 0.5)

    actual = constant_modal_admittance_step(
        previous,
        magnetic,
        incident,
        ratio,
    )

    assert actual.shape == (2, 3)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("ratio", [0.0, -1.0, np.asarray((1.0, 0.0))])
def test_constant_modal_admittance_step_rejects_nonpositive_ratio(ratio):
    with pytest.raises(ValueError, match="tau_over_dt must be positive"):
        constant_modal_admittance_step(0.0, 0.0, 0.0, ratio)


@pytest.mark.parametrize(
    ("previous", "magnetic", "incident", "ratio"),
    [
        (np.nan, 0.0, 0.0, 1.0),
        (0.0, np.inf, 0.0, 1.0),
        (0.0, 0.0, -np.inf, 1.0),
        (0.0, 0.0, 0.0, np.nan),
        (0.0, 0.0, 0.0, np.inf),
    ],
)
def test_constant_modal_admittance_step_rejects_nonfinite_inputs(
    previous,
    magnetic,
    incident,
    ratio,
):
    with pytest.raises(ValueError, match="finite"):
        constant_modal_admittance_step(
            previous,
            magnetic,
            incident,
            ratio,
        )


def test_constant_modal_admittance_step_rejects_incompatible_shapes():
    with pytest.raises(ValueError, match="broadcast-compatible"):
        constant_modal_admittance_step(
            np.zeros(2),
            np.zeros(3),
            0.0,
            1.0,
        )


@pytest.mark.parametrize(
    ("mode_indices", "invariant_axis", "message"),
    [
        ((1, 2), None, "exactly one retained mode"),
        ((1,), 2, "supports 3D ports only"),
    ],
)
def test_modal_admittance_boundary_rejects_multimode_and_2d_before_setup(
    monkeypatch,
    mode_indices,
    invariant_axis,
    message,
):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(dtypes={"float_or_double": np.dtype(np.float64)}),
    )
    owner = SimpleNamespace(
        match_depth_cells=1,
        normal_axis=0,
        direction="+",
        plane_index=1,
        mode_indices=mode_indices,
        invariant_axis=invariant_axis,
    )
    grid = SimpleNamespace(size=np.asarray((2, 2, 2)))

    with pytest.raises(ValueError, match=message):
        MatchedEigenmodeBoundary(owner, grid)


def _synthetic_raw_power_boundary(normal_axis, direction_sign):
    transverse_axes = tuple(axis for axis in range(3) if axis != normal_axis)
    coordinate_basis = np.eye(3, dtype=np.int32)
    handedness = int(
        np.dot(
            np.cross(
                coordinate_basis[transverse_axes[0]],
                coordinate_basis[transverse_axes[1]],
            ),
            coordinate_basis[normal_axis],
        )
    )
    transverse_start = (1, 2)
    transverse_stop = (3, 5)

    def local_component_ranges(local_axis, field_kind):
        u0, v0 = transverse_start
        u1, v1 = transverse_stop
        if field_kind == "E":
            return (
                (slice(u0, u1), slice(v0, v1 + 1))
                if local_axis == 0
                else (slice(u0, u1 + 1), slice(v0, v1))
            )
        return (
            (slice(u0, u1 + 1), slice(v0, v1))
            if local_axis == 0
            else (slice(u0, u1), slice(v0, v1 + 1))
        )

    boundary = MatchedEigenmodeBoundary.__new__(MatchedEigenmodeBoundary)
    boundary.owner = SimpleNamespace(
        transverse_axes=transverse_axes,
        transverse_start=transverse_start,
        transverse_stop=transverse_stop,
        _local_component_ranges=local_component_ranges,
        _modal_basis_handedness=lambda: handedness,
    )
    boundary.normal_axis = normal_axis
    boundary.direction_sign = direction_sign
    boundary.boundary_index = 0 if direction_sign > 0 else 5
    boundary.mode_indices = (1,)
    electric_u = np.arange(1.0, 9.0).reshape(2, 4)
    electric_v = np.arange(11.0, 20.0).reshape(3, 3)
    boundary.basis = np.asarray(
        (np.concatenate((electric_u.ravel(), electric_v.ravel())),)
    )
    boundary.modal_hu = np.asarray((-handedness * electric_v,))
    boundary.modal_hv = np.asarray((handedness * electric_u,))

    field_shape = (7, 7, 7)
    grid = SimpleNamespace(
        dl=np.asarray((1.25, 1.5, 1.75)),
        Ex=np.zeros(field_shape),
        Ey=np.zeros(field_shape),
        Ez=np.zeros(field_shape),
        Hx=np.zeros(field_shape),
        Hy=np.zeros(field_shape),
        Hz=np.zeros(field_shape),
    )
    boundary.power_gram = boundary._prepare_power_gram(grid)
    return boundary, grid, electric_u, electric_v


@pytest.mark.parametrize("normal_axis", [0, 1, 2])
@pytest.mark.parametrize("direction_sign", [-1, 1])
def test_raw_power_pairing_and_boundary_write_are_axis_and_sign_invariant(
    normal_axis,
    direction_sign,
):
    boundary, grid, electric_u, electric_v = _synthetic_raw_power_boundary(
        normal_axis,
        direction_sign,
    )
    amplitude = 2.75
    hplane = (
        boundary.boundary_index
        if direction_sign > 0
        else boundary.boundary_index - 1
    )
    magnetic_fields = (grid.Hx, grid.Hy, grid.Hz)
    u_axis, v_axis = boundary.owner.transverse_axes
    raw_hu = boundary._local_component_view(
        magnetic_fields[u_axis], 0, "H", hplane
    )
    raw_hv = boundary._local_component_view(
        magnetic_fields[v_axis], 1, "H", hplane
    )
    raw_hu[...] = direction_sign * amplitude * boundary.modal_hu[0]
    raw_hv[...] = direction_sign * amplitude * boundary.modal_hv[0]

    incoming = boundary._read_boundary_magnetic_coefficients(grid)
    raw_hu *= -1
    raw_hv *= -1
    outgoing = boundary._read_boundary_magnetic_coefficients(grid)

    measure = grid.dl[u_axis] * grid.dl[v_axis]
    expected_gram = measure * (
        np.sum(electric_u**2) + np.sum(electric_v**2)
    )
    np.testing.assert_allclose(boundary.power_gram, ((expected_gram,),))
    np.testing.assert_allclose(incoming, (amplitude,))
    np.testing.assert_allclose(outgoing, (-amplitude,))

    reconstructed = amplitude * boundary.basis[0]
    boundary._write_boundary_field(grid, reconstructed)
    electric_fields = (grid.Ex, grid.Ey, grid.Ez)
    np.testing.assert_allclose(
        boundary._tangential_component_view(
            electric_fields[u_axis], u_axis, boundary.boundary_index
        ),
        amplitude * electric_u,
    )
    np.testing.assert_allclose(
        boundary._tangential_component_view(
            electric_fields[v_axis], v_axis, boundary.boundary_index
        ),
        amplitude * electric_v,
    )
    assert np.count_nonzero(electric_fields[normal_axis]) == 0
    assert sum(np.count_nonzero(field) for field in electric_fields) == (
        electric_u.size + electric_v.size
    )


def _synthetic_time_constant_boundary(frequencies):
    boundary = MatchedEigenmodeBoundary.__new__(MatchedEigenmodeBoundary)
    boundary.owner = SimpleNamespace(
        port_index=7,
        port_monitor=SimpleNamespace(
            anchor_frequencies=np.asarray(frequencies, dtype=np.float64),
            anchor_neff=np.full((len(frequencies), 1), 2.0 + 0.0j),
        ),
        dft_start=4e9,
        dft_stop=6e9,
    )
    boundary.basis_frequency = 5e9
    boundary.mode_indices = (1,)
    boundary.normal_axis = 0
    grid = SimpleNamespace(dl=np.ones(3))
    return boundary, grid


def test_modal_admittance_boundary_warns_for_broad_fixed_admittance(caplog):
    boundary, grid = _synthetic_time_constant_boundary((4e9, 5e9, 6e9))

    with caplog.at_level(logging.WARNING):
        boundary._prepare_modal_time_constants(grid)

    assert "40.0% of the centre frequency" in caplog.text
    assert "prefer a narrower band" in caplog.text


def test_modal_admittance_boundary_warns_when_group_delay_is_unverified(caplog):
    boundary, grid = _synthetic_time_constant_boundary((5e9,))

    with caplog.at_level(logging.WARNING):
        boundary._prepare_modal_time_constants(grid)

    assert "has only one anchor" in caplog.text
    assert "group delay cannot be verified" in caplog.text

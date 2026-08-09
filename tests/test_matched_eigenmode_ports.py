from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.matched_eigenmode_ports as matched_ports
from gprMax.matched_eigenmode_ports import (
    CausalModalFIR,
    MatchedEigenmodeBoundary,
    cascade_impulse_response,
    cascaded_impulse_responses,
    one_cell_impulse_response,
)


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
    direction, plane, depth, pml
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
    direction, plane, depth, pml, message
):
    boundary, grid = _location_boundary(
        direction=direction,
        plane=plane,
        depth=depth,
        pml=pml,
    )

    with pytest.raises(ValueError, match=message):
        boundary._validate_location(grid)


def _direct_semiline_impulse_response(
    sample_count,
    dt,
    spacing,
    cutoff_wavenumber,
    wave_speed,
):
    """Reference Eq. (35) on a long line, without the four-node closure."""

    courant_squared = (wave_speed * dt / spacing) ** 2
    cutoff_squared = (wave_speed * dt * cutoff_wavenumber) ** 2
    coefficient_a = 2 - 2 * courant_squared - cutoff_squared
    coefficient_b = courant_squared

    # The far fixed boundary cannot return a reflection to node 1 within the
    # retained time interval. Node 0 is a unit impulse at time zero.
    node_count = sample_count + 3
    previous = np.zeros(node_count, dtype=np.float64)
    current = np.zeros(node_count, dtype=np.float64)
    current[0] = 1
    response = np.zeros(sample_count, dtype=np.float64)

    for time_index in range(1, sample_count):
        following = np.zeros(node_count, dtype=np.float64)
        following[1:-1] = (
            coefficient_a * current[1:-1]
            - previous[1:-1]
            + coefficient_b * (current[:-2] + current[2:])
        )
        previous, current = current, following
        response[time_index] = current[1]

    return response


@pytest.mark.parametrize("cutoff_wavenumber", [0.0, 0.7])
def test_one_cell_recurrence_matches_independent_direct_line(cutoff_wavenumber):
    sample_count = 24
    dt = 0.2
    spacing = 1.0
    wave_speed = 1.5

    actual = one_cell_impulse_response(
        sample_count,
        dt,
        spacing,
        cutoff_wavenumber,
        wave_speed=wave_speed,
    )
    expected = _direct_semiline_impulse_response(
        sample_count,
        dt,
        spacing,
        cutoff_wavenumber,
        wave_speed,
    )

    courant_squared = (wave_speed * dt / spacing) ** 2
    assert actual[0] == 0
    assert actual[1] == pytest.approx(courant_squared)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)


def test_one_cell_recurrence_accepts_stability_limit_and_rejects_larger_step():
    spacing = 0.5
    cutoff_wavenumber = 3.0
    wave_speed = 2.0
    stability_limit = 2 / (
        wave_speed * np.hypot(cutoff_wavenumber, 2 / spacing)
    )

    at_limit = one_cell_impulse_response(
        32,
        stability_limit,
        spacing,
        cutoff_wavenumber,
        wave_speed=wave_speed,
    )
    assert np.all(np.isfinite(at_limit))

    with pytest.raises(ValueError, match="stability limit"):
        one_cell_impulse_response(
            32,
            stability_limit * (1 + 1e-12),
            spacing,
            cutoff_wavenumber,
            wave_speed=wave_speed,
        )


@pytest.mark.parametrize("depth_cells", [1, 2, 4])
def test_cascade_depth_sets_delay_and_p_2d_is_twice_the_depth(depth_cells):
    one_cell = np.asarray([0.0, 0.5, -0.25, 0.125])
    sample_count = 32

    depth_response, double_depth_response = cascaded_impulse_responses(
        one_cell,
        depth_cells,
        sample_count=sample_count,
    )
    direct_double_depth = cascade_impulse_response(
        one_cell,
        2 * depth_cells,
        sample_count=sample_count,
    )

    assert np.flatnonzero(depth_response)[0] == depth_cells
    assert np.flatnonzero(double_depth_response)[0] == 2 * depth_cells
    assert depth_response[depth_cells] == pytest.approx(one_cell[1] ** depth_cells)
    assert double_depth_response[2 * depth_cells] == pytest.approx(
        one_cell[1] ** (2 * depth_cells)
    )
    np.testing.assert_allclose(double_depth_response, direct_double_depth)
    np.testing.assert_allclose(
        double_depth_response,
        np.convolve(depth_response, depth_response)[:sample_count],
    )


def test_causal_modal_fir_online_steps_match_offline_convolution_and_reset():
    kernels = np.asarray(
        [
            [0.0, 0.5, -0.25, 0.125],
            [0.0, -0.2, 0.3, 0.1],
        ]
    )
    samples = np.asarray(
        [
            [1.0, -2.0],
            [0.5, 1.5],
            [-0.75, 0.25],
            [2.0, -1.0],
            [0.0, 0.5],
            [-1.5, 2.0],
        ]
    )
    expected = np.column_stack(
        [
            np.convolve(samples[:, mode], kernels[mode])[: len(samples)]
            for mode in range(kernels.shape[0])
        ]
    )
    fir = CausalModalFIR(kernels)

    first_pass = np.vstack([fir.step(sample) for sample in samples])

    np.testing.assert_allclose(first_pass, expected)
    assert np.any(fir.history != 0)
    history_copy = fir.history
    history_copy.fill(123)
    assert not np.all(fir.history == 123)

    fir.reset()
    np.testing.assert_array_equal(fir.history, np.zeros_like(fir.history))
    second_pass = np.vstack([fir.step(sample) for sample in samples])
    np.testing.assert_allclose(second_pass, expected)


@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [
        (np.dtype(np.float32), 2e-5, 2e-5),
        (np.dtype(np.float64), 3e-13, 3e-13),
        (np.dtype(np.complex64), 2e-5, 2e-5),
        (np.dtype(np.complex128), 3e-13, 3e-13),
    ],
)
def test_causal_modal_fir_public_wrapper_preserves_dtype_across_wraps(
    dtype, rtol, atol
):
    rng = np.random.default_rng(9182)
    kernels = rng.standard_normal((3, 7))
    samples = rng.standard_normal((29, 3))
    if dtype.kind == "c":
        kernels = kernels + 1j * rng.standard_normal(kernels.shape)
        samples = samples + 1j * rng.standard_normal(samples.shape)
    kernels = np.ascontiguousarray(kernels, dtype=dtype)
    samples = np.ascontiguousarray(samples, dtype=dtype)
    kernels[:, 0] = 0
    expected = np.column_stack(
        [
            np.convolve(samples[:, mode], kernels[mode])[: len(samples)]
            for mode in range(kernels.shape[0])
        ]
    )

    fir = CausalModalFIR(kernels)
    actual = np.vstack([fir.step(sample) for sample in samples])

    assert fir.backend == "cython"
    assert actual.dtype == dtype
    assert actual.flags.c_contiguous
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def test_causal_modal_fir_preserves_float16_numpy_fallback():
    kernels = np.asarray(((0.0, 0.5, 0.25),), dtype=np.float16)
    fir = CausalModalFIR(kernels)

    actual = np.vstack([fir.step(value) for value in (2.0, 4.0, 8.0)])

    assert fir.backend == "numpy"
    assert actual.dtype == np.dtype(np.float16)
    np.testing.assert_array_equal(
        actual[:, 0], np.asarray((0.0, 1.0, 2.5), dtype=np.float16)
    )
    np.testing.assert_array_equal(
        fir.history, np.asarray(((8.0, 4.0),), dtype=np.float16)
    )


def test_numpy_fir_fallback_supports_aliasing_samples_and_output():
    kernels = np.asarray(((0.0, 0.5, -0.25),))
    history = np.asarray(((2.0, 4.0),))
    values = np.asarray((7.0,))

    next_write = matched_ports._causal_modal_fir_step_numpy(
        kernels,
        history,
        1,
        2,
        values,
        values,
    )

    np.testing.assert_array_equal(values, np.asarray((0.0,)))
    np.testing.assert_array_equal(history, np.asarray(((2.0, 7.0),)))
    assert next_write == 0


def test_causal_modal_fir_history_is_newest_first_after_wrap_and_is_a_copy():
    fir = CausalModalFIR(np.zeros((2, 4)))
    samples = np.asarray(
        [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)]
    )
    for sample in samples:
        fir.step(sample)

    np.testing.assert_array_equal(
        fir.history,
        np.asarray(((6, 5, 4), (60, 50, 40))),
    )
    returned = fir.history
    returned.fill(-999)
    np.testing.assert_array_equal(
        fir.history,
        np.asarray(((6, 5, 4), (60, 50, 40))),
    )


def test_causal_modal_fir_reset_after_wrap_matches_fresh_instance():
    kernels = np.asarray(((0.0, 0.4, -0.2, 0.1), (0.0, -0.3, 0.25, 0.2)))
    fir = CausalModalFIR(kernels)
    for sample in np.arange(20, dtype=np.float64).reshape(10, 2):
        fir.step(sample)

    fir.reset()
    fresh = CausalModalFIR(kernels)
    replay = np.asarray(((2.0, -1.0), (0.5, 3.0), (-4.0, 0.25)))

    np.testing.assert_array_equal(fir.history, np.zeros((2, 3)))
    np.testing.assert_array_equal(
        np.vstack([fir.step(sample) for sample in replay]),
        np.vstack([fresh.step(sample) for sample in replay]),
    )


def test_causal_modal_fir_accepts_scalar_and_noncontiguous_samples():
    scalar_fir = CausalModalFIR(np.asarray((0.0, 0.5), dtype=np.float32))
    first = scalar_fir.step(np.float32(3.0))
    second = scalar_fir.step(np.float32(4.0))
    np.testing.assert_array_equal(first, np.asarray((0.0,), dtype=np.float32))
    np.testing.assert_array_equal(second, np.asarray((1.5,), dtype=np.float32))

    kernels = np.zeros((3, 3), dtype=np.float64)
    kernels[:, 1] = 1
    fir = CausalModalFIR(kernels)
    backing = np.asarray((1.0, 99.0, 2.0, 99.0, 3.0, 99.0))
    strided = backing[::2]
    assert not strided.flags.c_contiguous
    np.testing.assert_array_equal(fir.step(strided), np.zeros(3))
    np.testing.assert_array_equal(
        fir.step(np.asarray((4.0, 5.0, 6.0))),
        np.asarray((1.0, 2.0, 3.0)),
    )
    np.testing.assert_array_equal(backing, (1.0, 99.0, 2.0, 99.0, 3.0, 99.0))


@pytest.mark.parametrize(
    "samples,message",
    [
        (1.0, "shape"),
        (np.zeros((2, 1)), "shape"),
        (np.asarray((np.nan, 0.0)), "finite"),
        (np.asarray((0.0, np.inf)), "finite"),
        (np.asarray((1.0 + 1.0j, 0.0)), "complex samples"),
    ],
)
def test_causal_modal_fir_public_wrapper_validates_samples(samples, message):
    fir = CausalModalFIR(np.zeros((2, 3)))
    with pytest.raises(ValueError, match=message):
        fir.step(samples)


def test_causal_modal_fir_copies_kernels_and_does_not_mutate_samples():
    kernels = np.asarray(((0.0, 0.5, 0.25), (0.0, -0.5, 0.1)))
    expected_kernels = kernels.copy()
    fir = CausalModalFIR(kernels)
    kernels.fill(123)
    samples = np.asarray((2.0, -4.0))
    expected_samples = samples.copy()

    output = fir.step(samples)

    np.testing.assert_array_equal(fir.kernels, expected_kernels)
    np.testing.assert_array_equal(samples, expected_samples)
    assert not np.shares_memory(output, samples)
    assert not fir.kernels.flags.writeable


def test_causal_modal_fir_step_dispatches_through_selected_kernel(monkeypatch):
    calls = []

    class SpyDispatcher:
        def __getitem__(self, signature):
            calls.append(("signature", signature))

            def step(kernels, history, write_index, valid_history, samples, output):
                calls.append(("step", write_index, valid_history, samples.copy()))
                return matched_ports._causal_modal_fir_step_numpy(
                    kernels,
                    history,
                    write_index,
                    valid_history,
                    samples,
                    output,
                )

            return step

    monkeypatch.setattr(matched_ports, "CYTHON_MODAL_FIR_AVAILABLE", True)
    monkeypatch.setattr(matched_ports, "causal_modal_fir_step", SpyDispatcher())
    fir = CausalModalFIR(np.asarray((0.0, 0.75), dtype=np.float64))

    fir.step(2.0)

    assert fir.backend == "cython"
    assert calls[0] == ("signature", "double")
    assert calls[1][0:3] == ("step", 0, 0)


@pytest.mark.parametrize(
    "kernels",
    [
        [1.0, 0.0],
        [[0.0, 1.0], [1e-300, 0.0]],
    ],
)
def test_causal_modal_fir_rejects_nonzero_zero_lag(kernels):
    with pytest.raises(ValueError, match="zero zero-lag coefficient"):
        CausalModalFIR(kernels)

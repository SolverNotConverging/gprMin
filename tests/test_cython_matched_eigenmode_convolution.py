"""Direct tests for the compiled matched-modal circular FIR kernel."""

import numpy as np
import pytest

from gprMax.cython.matched_eigenmode_convolution import causal_modal_fir_step


_SIGNATURES = {
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
    np.dtype(np.complex64): "float complex",
    np.dtype(np.complex128): "double complex",
}


def _random_values(rng, shape, dtype):
    values = rng.standard_normal(shape)
    if dtype.kind == "c":
        values = values + 1j * rng.standard_normal(shape)
    return np.ascontiguousarray(values, dtype=dtype)


def _direct_strictly_causal_outputs(kernels, samples):
    """Independent chronological-history implementation of the FIR sum."""

    outputs = np.zeros_like(samples)
    for sample_index in range(samples.shape[0]):
        available = min(sample_index, kernels.shape[1] - 1)
        for lag in range(1, available + 1):
            outputs[sample_index] += (
                kernels[:, lag] * samples[sample_index - lag]
            )
    return outputs


@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [
        (np.dtype(np.float32), 2e-5, 2e-5),
        (np.dtype(np.float64), 3e-13, 3e-13),
        (np.dtype(np.complex64), 2e-5, 2e-5),
        (np.dtype(np.complex128), 3e-13, 3e-13),
    ],
)
@pytest.mark.parametrize(
    "mode_count,tap_count",
    [(1, 1), (1, 2), (3, 7), (4, 257)],
)
def test_cython_circular_step_matches_direct_convolution_after_multiple_wraps(
    dtype,
    rtol,
    atol,
    mode_count,
    tap_count,
):
    rng = np.random.default_rng(8431 + mode_count + tap_count)
    kernels = _random_values(rng, (mode_count, tap_count), dtype)
    kernels[:, 0] = 0
    step_count = 3 * max(1, tap_count - 1) + 5
    samples = _random_values(rng, (step_count, mode_count), dtype)
    expected = _direct_strictly_causal_outputs(kernels, samples)

    history_length = tap_count - 1
    history = np.zeros((mode_count, history_length), dtype=dtype)
    write_index = 0
    valid_history = 0
    outputs = np.empty_like(samples)
    step = causal_modal_fir_step[_SIGNATURES[dtype]]

    for sample_index, sample in enumerate(samples):
        write_index = step(
            kernels,
            history,
            write_index,
            valid_history,
            sample,
            outputs[sample_index],
        )
        valid_history = min(valid_history + 1, history_length)

    assert outputs.dtype == dtype
    assert outputs.flags.c_contiguous
    np.testing.assert_allclose(outputs, expected, rtol=rtol, atol=atol)
    if history_length:
        assert write_index == step_count % history_length
        newest_first = (
            write_index - 1 - np.arange(history_length)
        ) % history_length
        np.testing.assert_array_equal(
            history[:, newest_first],
            samples[-history_length:][::-1].T,
        )
    else:
        assert write_index == 0
        assert history.shape == (mode_count, 0)


@pytest.mark.parametrize("write_index", [0, 2, 4])
def test_cython_step_mutates_only_write_slot_and_wraps(write_index):
    mode_count = 3
    history_length = 5
    kernels = np.arange(
        mode_count * (history_length + 1), dtype=np.float64
    ).reshape(mode_count, history_length + 1)
    kernels[:, 0] = 0
    history = (
        100 + np.arange(mode_count * history_length, dtype=np.float64)
    ).reshape(mode_count, history_length)
    samples = np.asarray((7.5, -3.25, 11.0))
    output = np.full(mode_count, np.nan)
    kernels_before = kernels.copy()
    history_before = history.copy()
    samples_before = samples.copy()

    expected = np.zeros(mode_count)
    for lag in range(history_length):
        history_index = (write_index - 1 - lag) % history_length
        expected += kernels[:, lag + 1] * history[:, history_index]

    next_write = causal_modal_fir_step["double"](
        kernels,
        history,
        write_index,
        history_length,
        samples,
        output,
    )

    np.testing.assert_array_equal(kernels, kernels_before)
    np.testing.assert_array_equal(samples, samples_before)
    np.testing.assert_allclose(output, expected)
    np.testing.assert_array_equal(history[:, write_index], samples)
    untouched = np.arange(history_length) != write_index
    np.testing.assert_array_equal(
        history[:, untouched], history_before[:, untouched]
    )
    assert next_write == (write_index + 1) % history_length


def test_cython_step_is_strictly_causal_and_supports_zero_history():
    one_delay = np.asarray(((0.0, 1.0),))
    history = np.zeros((1, 1))
    output = np.empty(1)
    step = causal_modal_fir_step["double"]

    write_index = step(one_delay, history, 0, 0, np.asarray((9.0,)), output)
    np.testing.assert_array_equal(output, (0.0,))
    step(one_delay, history, write_index, 1, np.asarray((0.0,)), output)
    np.testing.assert_array_equal(output, (9.0,))

    no_delay = np.zeros((2, 1))
    empty_history = np.zeros((2, 0))
    output = np.full(2, np.nan)
    next_write = step(
        no_delay,
        empty_history,
        0,
        0,
        np.asarray((2.0, -4.0)),
        output,
    )
    np.testing.assert_array_equal(output, np.zeros(2))
    assert next_write == 0


def test_cython_step_supports_aliasing_samples_and_output():
    kernels = np.asarray(
        ((0.0, 0.5, -0.25, 0.125), (0.0, -0.2, 0.3, 0.1))
    )
    history = np.asarray(((1.0, 2.0, 3.0), (-1.0, -2.0, -3.0)))
    write_index = 1
    values = np.asarray((7.0, -4.0))
    current = values.copy()
    history_before = history.copy()
    expected = np.zeros(2)
    for lag in range(3):
        history_index = (write_index - 1 - lag) % 3
        expected += kernels[:, lag + 1] * history[:, history_index]

    next_write = causal_modal_fir_step["double"](
        kernels,
        history,
        write_index,
        3,
        values,
        values,
    )

    np.testing.assert_allclose(values, expected)
    np.testing.assert_array_equal(history[:, write_index], current)
    untouched = np.arange(3) != write_index
    np.testing.assert_array_equal(
        history[:, untouched], history_before[:, untouched]
    )
    assert next_write == 2


def test_cython_step_rejects_inconsistent_shapes_and_state_indices():
    step = causal_modal_fir_step["double"]
    kernels = np.zeros((2, 4))
    history = np.zeros((2, 3))
    samples = np.zeros(2)
    output = np.zeros(2)

    with pytest.raises(ValueError, match="one more column"):
        step(np.zeros((2, 3)), history, 0, 0, samples, output)
    with pytest.raises(ValueError, match="same mode count"):
        step(kernels, history[:1], 0, 0, samples, output)
    with pytest.raises(ValueError, match="one value per mode"):
        step(kernels, history, 0, 0, samples[:1], output)
    with pytest.raises(ValueError, match="one value per mode"):
        step(kernels, history, 0, 0, samples, output[:1])
    with pytest.raises(ValueError, match="valid_history"):
        step(kernels, history, 0, -1, samples, output)
    with pytest.raises(ValueError, match="valid_history"):
        step(kernels, history, 0, 4, samples, output)
    with pytest.raises(ValueError, match="write_index"):
        step(kernels, history, -1, 0, samples, output)
    with pytest.raises(ValueError, match="write_index"):
        step(kernels, history, 3, 0, samples, output)
    with pytest.raises(ValueError, match="write_index must be zero"):
        step(
            np.zeros((2, 1)),
            np.zeros((2, 0)),
            1,
            0,
            samples,
            output,
        )


@pytest.mark.parametrize(
    "write_index,valid_history,message",
    [
        (-1, 0, "write_index"),
        (3, 0, "write_index"),
        (0, -1, "valid_history"),
        (0, 4, "valid_history"),
    ],
)
def test_cython_state_validation_does_not_mutate_history_or_output(
    write_index, valid_history, message
):
    kernels = np.arange(8, dtype=np.float64).reshape(2, 4)
    kernels[:, 0] = 0
    history = np.arange(6, dtype=np.float64).reshape(2, 3)
    samples = np.asarray((2.5, -7.0))
    output = np.asarray((101.0, 202.0))
    history_before = history.copy()
    output_before = output.copy()

    with pytest.raises(ValueError, match=message):
        causal_modal_fir_step["double"](
            kernels,
            history,
            write_index,
            valid_history,
            samples,
            output,
        )

    np.testing.assert_array_equal(history, history_before)
    np.testing.assert_array_equal(output, output_before)

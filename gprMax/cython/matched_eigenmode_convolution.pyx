# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <http://www.gnu.org/licenses/>.

"""Causal circular-history convolution for matched eigenmode boundaries."""

cimport cython


ctypedef fused fir_numeric:
    float
    double
    float complex
    double complex


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef Py_ssize_t causal_modal_fir_step(
    const fir_numeric[:, ::1] kernels,
    fir_numeric[:, ::1] history,
    Py_ssize_t write_index,
    Py_ssize_t valid_history,
    const fir_numeric[::1] samples,
    fir_numeric[::1] output,
):
    """Evaluate one strictly causal FIR step and store the current samples.

    ``kernels[:, 0]`` is the zero-lag coefficient and is deliberately not
    evaluated. ``write_index`` identifies the circular-history slot that will
    receive the current samples only after every output has been accumulated.
    The returned value is the slot for the next call.

    Shape, index, and history-count checks remain in the compiled entry point
    because bounds checking is disabled for the inner loop.
    """

    cdef Py_ssize_t mode_count = kernels.shape[0]
    cdef Py_ssize_t history_length = history.shape[1]
    cdef Py_ssize_t mode, lag, history_index, first_count
    cdef fir_numeric accumulator, current_sample

    if kernels.shape[1] != history_length + 1:
        raise ValueError("kernels must have exactly one more column than history")
    if history.shape[0] != mode_count:
        raise ValueError("kernels and history must have the same mode count")
    if samples.shape[0] != mode_count or output.shape[0] != mode_count:
        raise ValueError("samples and output must have one value per mode")
    if valid_history < 0 or valid_history > history_length:
        raise ValueError("valid_history is outside the circular-history range")
    if history_length == 0:
        if write_index != 0:
            raise ValueError("write_index must be zero for a zero-length history")
        for mode in range(mode_count):
            output[mode] = 0
        return 0
    if write_index < 0 or write_index >= history_length:
        raise ValueError("write_index is outside the circular-history range")

    # The newest samples occupy slots immediately before write_index. Split
    # the reverse traversal at the physical start of the ring so the hot loops
    # contain no modulo operation and no wrap branch.
    first_count = min(valid_history, write_index)
    with nogil:
        for mode in range(mode_count):
            # Capture before writing output so direct callers may safely use
            # the same one-dimensional array for samples and output.
            current_sample = samples[mode]
            accumulator = 0
            for lag in range(first_count):
                history_index = write_index - 1 - lag
                accumulator = accumulator + (
                    kernels[mode, lag + 1] * history[mode, history_index]
                )
            for lag in range(first_count, valid_history):
                history_index = history_length - 1 - (lag - first_count)
                accumulator = accumulator + (
                    kernels[mode, lag + 1] * history[mode, history_index]
                )
            output[mode] = accumulator
            # Modes are independent, so storing this captured sample after
            # its convolution preserves h[0] = 0 without a second mode loop.
            history[mode, write_index] = current_sample

    write_index += 1
    if write_index == history_length:
        write_index = 0
    return write_index

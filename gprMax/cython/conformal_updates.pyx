# cython: language_level=3, boundscheck=False, wraparound=False
"""Sparse integral updates for the internal conformal reference operator.

Array validation and CFL certification belong to the Python compiler. Each
row owns one output DOF; no atomics or cross-row writes are required.
"""

from libc.stdint cimport int64_t

ctypedef fused field_t:
    float
    double


def update_sparse(const int64_t[::1] offsets,
                  const int64_t[::1] indices,
                  const double[::1] coefficients,
                  const double[::1] mass,
                  const field_t[::1] source,
                  field_t[::1] target,
                  double signed_dt):
    cdef Py_ssize_t row, entry
    cdef double circulation
    with nogil:
        for row in range(target.shape[0]):
            circulation = 0
            for entry in range(offsets[row], offsets[row + 1]):
                circulation = circulation + coefficients[entry] * source[indices[entry]]
            target[row] = target[row] + signed_dt * circulation / mass[row]

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

"""Matched modal FDTD boundary conditions and numerical translation kernels.

The scalar translation-operator machinery follows Alimenti et al., *IEEE
Transactions on Microwave Theory and Techniques*, vol. 48, no. 1, 2000. The
runtime boundary couples those kernels to gprMax's solved eigenmode profiles
and CPU time-stepping loop.

The paper's four-node construction is an auxiliary one-dimensional modal
line used to obtain a one-cell impulse response. It is not a four-cell
absorbing layer in the three-dimensional FDTD grid.
"""

from __future__ import annotations

import logging
import numbers

import numpy as np
import numpy.typing as npt

import gprMax.config as config


def _causal_modal_fir_step_numpy(
    kernels,
    history,
    write_index,
    valid_history,
    samples,
    output,
):
    """Source-tree fallback for the compiled circular-history FIR kernel."""

    mode_count, tap_count = kernels.shape
    history_length = history.shape[1]
    if tap_count != history_length + 1:
        raise ValueError("kernels must have exactly one more column than history")
    if history.shape[0] != mode_count:
        raise ValueError("kernels and history must have the same mode count")
    if samples.shape != (mode_count,) or output.shape != (mode_count,):
        raise ValueError("samples and output must have one value per mode")
    if not 0 <= valid_history <= history_length:
        raise ValueError("valid_history is outside the circular-history range")
    if history_length == 0:
        if write_index != 0:
            raise ValueError("write_index must be zero for a zero-length history")
        output.fill(0)
        return 0
    if not 0 <= write_index < history_length:
        raise ValueError("write_index is outside the circular-history range")

    current = np.array(samples, copy=True)
    output.fill(0)
    for lag in range(valid_history):
        history_index = (write_index - 1 - lag) % history_length
        output += kernels[:, lag + 1] * history[:, history_index]
    history[:, write_index] = current
    return (write_index + 1) % history_length


try:
    from gprMax.cython.matched_eigenmode_convolution import causal_modal_fir_step
except ImportError:  # Source-tree fallback before extensions are rebuilt.
    CYTHON_MODAL_FIR_AVAILABLE = False

    class _FallbackDispatcher:
        def __call__(self, *args):
            return _causal_modal_fir_step_numpy(*args)

        def __getitem__(self, unused):
            return _causal_modal_fir_step_numpy

    causal_modal_fir_step = _FallbackDispatcher()
else:
    CYTHON_MODAL_FIR_AVAILABLE = True


__all__ = [
    "CYTHON_MODAL_FIR_AVAILABLE",
    "CausalModalFIR",
    "MatchedEigenmodeBoundary",
    "cascade_impulse_response",
    "cascaded_impulse_responses",
    "initialise_eigenmode_matches",
    "one_cell_impulse_response",
]


logger = logging.getLogger(__name__)
PROFILE_IMAGINARY_TOLERANCE = 1e-8
PROFILE_OVERLAP_TOLERANCE = 0.999
CUTOFF_RELATIVE_SPREAD_TOLERANCE = 5e-2
GRAM_CONDITION_LIMIT = 1e10
PROJECTION_RELATIVE_ERROR_BUDGET = 1e-3
FIR_CYTHON_SIGNATURES = {
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
    np.dtype(np.complex64): "float complex",
    np.dtype(np.complex128): "double complex",
}


def _positive_integer(value, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be an integer greater than zero")
    return int(value)


def _real_scalar(value, name: str, *, allow_zero: bool) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    outside_domain = result < 0 if allow_zero else result <= 0
    if outside_domain:
        qualifier = "non-negative" if allow_zero else "greater than zero"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _inexact_dtype(dtype, *, values=None) -> np.dtype:
    if dtype is None:
        resolved = np.asarray(values).dtype
        if resolved.kind not in "fc":
            resolved = np.dtype(np.float64)
    else:
        resolved = np.dtype(dtype)
    if resolved.kind not in "fc":
        raise ValueError("dtype must be a real or complex floating-point dtype")
    return resolved


def one_cell_impulse_response(
    sample_count: int,
    dt: float,
    spacing: float,
    cutoff_wavenumber: float,
    *,
    wave_speed: float,
    dtype=np.float64,
) -> npt.NDArray[np.inexact]:
    """Generate the paper's numerical one-cell modal impulse response.

    The modal line obeys Eqs. (35)-(36), while the semi-infinite-line
    four-node construction is Eq. (38). ``spacing`` is the longitudinal
    cell size and ``cutoff_wavenumber`` is :math:`k_{c,n}`. ``wave_speed``
    is the propagation speed used by the configured homogeneous guide.

    The returned samples are ``h[0:sample_count]``. In particular,
    ``h[0] == 0`` and ``h[1] == B`` when a second sample is requested, so
    an online convolution never depends on the current input sample.

    Eq. (37) is checked before constructing the response. Inputs exactly at
    the stability limit are accepted.

    Args:
        sample_count: Number of causal impulse-response samples to return.
        dt: FDTD time step.
        spacing: Longitudinal FDTD cell spacing.
        cutoff_wavenumber: Non-negative modal cutoff wavenumber.
        wave_speed: Wave speed in the uniform guide.
        dtype: Real or complex floating-point output dtype.

    Returns:
        A contiguous one-dimensional impulse-response array.

    Raises:
        ValueError: If an input is invalid or Eq. (37) is violated.
    """

    count = _positive_integer(sample_count, "sample_count")
    time_step = _real_scalar(dt, "dt", allow_zero=False)
    cell_spacing = _real_scalar(spacing, "spacing", allow_zero=False)
    cutoff = _real_scalar(
        cutoff_wavenumber, "cutoff_wavenumber", allow_zero=True
    )
    speed = _real_scalar(wave_speed, "wave_speed", allow_zero=False)
    output_dtype = _inexact_dtype(dtype)
    real_dtype = np.empty((), dtype=output_dtype).real.dtype

    stability_limit = 2.0 / (
        speed * np.hypot(cutoff, 2.0 / cell_spacing)
    )
    if not np.isfinite(stability_limit) or time_step > stability_limit:
        raise ValueError(
            "dt violates the modal-line stability limit from paper Eq. (37): "
            f"dt={time_step:g}, limit={stability_limit:g}"
        )

    courant_squared = (speed * time_step / cell_spacing) ** 2
    cutoff_squared = (speed * time_step * cutoff) ** 2
    coefficient_a = np.asarray(
        2.0 - 2.0 * courant_squared - cutoff_squared, dtype=real_dtype
    )[()]
    coefficient_b = np.asarray(courant_squared, dtype=real_dtype)[()]
    if not np.isfinite(coefficient_a) or not np.isfinite(coefficient_b):
        raise ValueError("modal-line coefficients must be finite")

    response = np.zeros(count, dtype=output_dtype)
    if count == 1:
        return np.ascontiguousarray(response)
    response[1] = coefficient_b

    # V_3 and V_4 are the final two nodes of the auxiliary four-node line in
    # paper Eq. (38). Their samples at i=0 and i=1 are zero by construction.
    node_3 = np.zeros(count, dtype=output_dtype)
    node_4 = np.zeros(count, dtype=output_dtype)

    for time_index in range(2, count):
        response[time_index] = (
            coefficient_a * response[time_index - 1]
            - response[time_index - 2]
            + coefficient_b * node_3[time_index - 1]
        )
        node_3[time_index] = (
            coefficient_a * node_3[time_index - 1]
            - node_3[time_index - 2]
            + coefficient_b
            * (node_4[time_index - 1] + response[time_index - 1])
        )
        if time_index > 2:
            # Eq. (38): sum from ell=2 through i-1. Reversing node_3 makes
            # this a dot product with ascending h[1], h[2], ... samples.
            node_4[time_index] = np.dot(
                response[1 : time_index - 1],
                node_3[time_index - 1 : 1 : -1],
            )

    if not (
        np.all(np.isfinite(response))
        and np.all(np.isfinite(node_3))
        and np.all(np.isfinite(node_4))
    ):
        raise FloatingPointError(
            "the modal impulse response became non-finite despite a stable input step"
        )
    return np.ascontiguousarray(response)


def _validated_impulse_response(response, *, dtype=None) -> npt.NDArray[np.inexact]:
    output_dtype = _inexact_dtype(dtype, values=response)
    values = np.asarray(response, dtype=output_dtype)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("impulse_response must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("impulse_response must contain only finite values")
    return np.ascontiguousarray(values)


def cascade_impulse_response(
    impulse_response: npt.ArrayLike,
    depth_cells: int,
    *,
    sample_count: int | None = None,
    dtype=None,
) -> npt.NDArray[np.inexact]:
    """Cascade a causal one-cell FIR through ``depth_cells`` cells.

    Each convolution is truncated to ``sample_count`` causal samples. This is
    exact over the retained interval: discarded later samples cannot affect
    an earlier output of a causal convolution. NumPy convolution is used
    intentionally so this function remains a small executable reference.

    Args:
        impulse_response: One-cell causal FIR coefficients.
        depth_cells: Positive number of one-cell operators to cascade.
        sample_count: Retained output length. The input length is used when
            omitted.
        dtype: Optional real or complex floating-point calculation dtype.

    Returns:
        The causal, truncated impulse response at the requested depth.
    """

    depth = _positive_integer(depth_cells, "depth_cells")
    one_cell = _validated_impulse_response(impulse_response, dtype=dtype)
    if sample_count is None:
        retained_count = int(one_cell.size)
    else:
        retained_count = _positive_integer(sample_count, "sample_count")

    # A zero-cell translation is the convolution identity. Applying the
    # one-cell response `depth` times then constructs P_d from P_Delta-w.
    cascaded = np.zeros(retained_count, dtype=one_cell.dtype)
    cascaded[0] = 1
    for _ in range(depth):
        cascaded = np.convolve(cascaded, one_cell)[:retained_count]
        cascaded = np.asarray(cascaded, dtype=one_cell.dtype)
    return np.ascontiguousarray(cascaded)


def cascaded_impulse_responses(
    impulse_response: npt.ArrayLike,
    depth_cells: int,
    *,
    sample_count: int | None = None,
    dtype=None,
) -> tuple[npt.NDArray[np.inexact], npt.NDArray[np.inexact]]:
    r"""Return translation kernels for ``depth_cells`` and twice that depth.

    The first result represents :math:`\mathcal{P}_{n,d}`. Convolving it
    with itself produces :math:`\mathcal{P}_{n,2d}`, the source-cancellation
    term in paper Eq. (19). Both responses are causally truncated to the same
    number of samples.
    """

    depth_response = cascade_impulse_response(
        impulse_response,
        depth_cells,
        sample_count=sample_count,
        dtype=dtype,
    )
    retained_count = int(depth_response.size)
    double_depth_response = np.convolve(depth_response, depth_response)[
        :retained_count
    ]
    double_depth_response = np.ascontiguousarray(
        double_depth_response, dtype=depth_response.dtype
    )
    return depth_response, double_depth_response


class CausalModalFIR:
    r"""Online causal FIR state for independent scalar modal signals.

    ``kernels`` has shape ``(mode_count, tap_count)``; a one-dimensional
    kernel denotes one mode. At time index ``i``, :meth:`step` returns

    .. math::

        y_n[i] = \sum_{\ell=1}^{L-1} h_n[\ell]x_n[i-\ell].

    The current samples are stored only *after* this output is evaluated.
    Combined with the required zero-lag coefficient ``h[:, 0] == 0``, this
    guarantees that the object cannot create same-step feedback. Supported
    standard NumPy precisions use a Cython circular-history kernel; unusual
    extended precisions retain the NumPy fallback.
    """

    def __init__(self, kernels: npt.ArrayLike, *, dtype=None):
        resolved_dtype = _inexact_dtype(dtype, values=kernels)
        coefficients = np.asarray(kernels, dtype=resolved_dtype)
        if coefficients.ndim == 1:
            coefficients = coefficients[np.newaxis, :]
        if (
            coefficients.ndim != 2
            or coefficients.shape[0] == 0
            or coefficients.shape[1] == 0
        ):
            raise ValueError(
                "kernels must have shape (mode_count, tap_count) with non-zero dimensions"
            )
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("kernels must contain only finite values")
        if np.any(coefficients[:, 0] != 0):
            raise ValueError(
                "every modal kernel must have a zero zero-lag coefficient to avoid same-step feedback"
            )

        self.kernels = np.ascontiguousarray(coefficients).copy()
        self.kernels.setflags(write=False)
        self.dtype = self.kernels.dtype
        self.mode_count = int(self.kernels.shape[0])
        self.tap_count = int(self.kernels.shape[1])
        self._history = np.zeros(
            (self.mode_count, max(0, self.tap_count - 1)), dtype=self.dtype
        )
        signature = FIR_CYTHON_SIGNATURES.get(self.dtype)
        if CYTHON_MODAL_FIR_AVAILABLE and signature is not None:
            self._step_kernel = causal_modal_fir_step[signature]
            self.backend = "cython"
        else:
            self._step_kernel = _causal_modal_fir_step_numpy
            self.backend = "numpy"
        self._write_index = 0
        self._valid_history = 0

    def reset(self) -> None:
        """Reset every modal delay line to its zero initial condition."""

        self._history.fill(0)
        self._write_index = 0
        self._valid_history = 0

    @property
    def history(self) -> npt.NDArray[np.inexact]:
        """Return a copy of past samples, newest first along the last axis."""

        history_length = self._history.shape[1]
        if history_length == 0:
            return self._history.copy()
        newest_first = (
            self._write_index - 1 - np.arange(history_length)
        ) % history_length
        return np.ascontiguousarray(self._history[:, newest_first])

    def step(self, samples: npt.ArrayLike) -> npt.NDArray[np.inexact]:
        """Translate one scalar sample per mode and advance the FIR state."""

        raw_samples = np.asarray(samples)
        if raw_samples.ndim == 0 and self.mode_count == 1:
            raw_samples = raw_samples.reshape(1)
        if raw_samples.shape != (self.mode_count,):
            raise ValueError(
                f"samples must have shape ({self.mode_count},), got {raw_samples.shape}"
            )
        if self.dtype.kind != "c" and np.iscomplexobj(raw_samples):
            raise ValueError("complex samples require a complex FIR dtype")
        current = np.ascontiguousarray(raw_samples, dtype=self.dtype)
        if not np.all(np.isfinite(current)):
            raise ValueError("samples must contain only finite values")

        if self.tap_count == 1:
            return np.zeros(self.mode_count, dtype=self.dtype)

        translated = np.empty(self.mode_count, dtype=self.dtype)
        self._write_index = self._step_kernel(
            self.kernels,
            self._history,
            self._write_index,
            self._valid_history,
            current,
            translated,
        )
        self._valid_history = min(
            self._valid_history + 1, self._history.shape[1]
        )
        return translated


class MatchedEigenmodeBoundary:
    """Paper-based matched modal boundary attached to one eigenmode port.

    The eigenmode port remains the interior expansion/reference plane. The
    matched boundary is the corresponding outer domain face, separated by
    ``depth_cells`` ordinary cells of longitudinally uniform guide. At every
    integer electric-field time the total tangential field on the expansion
    plane is projected onto a fixed modal basis, translated causally, and
    imposed on the outer plane. For the active port, paper Eq. (19) adds the
    source term ``s - P_2d(s)``.

    This first implementation intentionally accepts only a homogeneous,
    lossless, nondispersive fill and effectively real, frequency-independent
    tangential modal profiles. Those restrictions keep the paper's scalar
    translation operator valid and prevent a frequency-dependent projection
    from being treated as an instantaneous operation.
    """

    formulation = "Alimenti2000NumericalModalTranslation"

    def __init__(self, owner, grid):
        self.owner = owner
        self.depth_cells = _positive_integer(
            owner.match_depth_cells, "match_depth_cells"
        )
        self.normal_axis = int(owner.normal_axis)
        self.direction_sign = 1 if owner.direction == "+" else -1
        self.boundary_index = (
            0 if self.direction_sign > 0 else int(grid.size[self.normal_axis])
        )
        self.expansion_plane_index = int(owner.plane_index)
        self.mode_indices = tuple(int(value) for value in owner.mode_indices)
        self.real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
        self._validate_location(grid)
        self.fill_material = self._validate_uniform_section(grid)
        self.relative_permittivity = float(self.fill_material.er)
        self.relative_permeability = float(self.fill_material.mr)
        self.wave_speed = config.c / np.sqrt(
            self.relative_permittivity * self.relative_permeability
        )
        self.basis, self.dual_basis = self._prepare_fixed_modal_basis(grid)
        self.cutoff_wavenumbers = self._prepare_cutoff_wavenumbers()
        self.source_mode_position = None
        if self.owner.port_monitor.is_source:
            self.source_mode_position = self.mode_indices.index(
                int(self.owner.mode_index)
            )

        self.translation_filter = None
        self.source_cancellation_filter = None
        if not getattr(config.sim_config, "geometry_only", False):
            self._prepare_translation_filters(grid)

    def _validate_location(self, grid):
        axis = "xyz"[self.normal_axis]
        face = f"{axis}0" if self.direction_sign > 0 else f"{axis}max"
        opposite_face = f"{axis}max" if self.direction_sign > 0 else f"{axis}0"
        normal_size = int(grid.size[self.normal_axis])
        actual_depth = (
            self.expansion_plane_index
            if self.direction_sign > 0
            else normal_size - self.expansion_plane_index
        )
        if actual_depth != self.depth_cells:
            raise ValueError(
                f"Eigenmode match on port {self.owner.port_index} declares "
                f"depth_cells={self.depth_cells}, but its reference plane is "
                f"{actual_depth} cell(s) from domain face {face}. The matched "
                "boundary must terminate at that domain face."
            )
        if not 0 < self.expansion_plane_index < normal_size:
            raise ValueError(
                f"Eigenmode match on port {self.owner.port_index} requires its "
                "modal reference plane to be strictly inside the domain, with "
                "at least one device-side cell beyond it."
            )
        if grid.pmls["thickness"][face] != 0:
            raise ValueError(
                f"Eigenmode match on port {self.owner.port_index} replaces domain "
                f"face {face}; set that face's PML thickness to zero."
            )
        if face in grid.symmetry_boundaries:
            raise ValueError(
                f"Eigenmode match on port {self.owner.port_index} cannot share "
                f"domain face {face} with a symmetry boundary."
            )
        opposite_pml = int(grid.pmls["thickness"][opposite_face])
        opposite_interface = (
            normal_size - opposite_pml
            if self.direction_sign > 0
            else opposite_pml
        )
        intersects_opposite_pml = (
            self.direction_sign > 0
            and self.expansion_plane_index > opposite_interface
        ) or (
            self.direction_sign < 0
            and self.expansion_plane_index < opposite_interface
        )
        if intersects_opposite_pml:
            raise ValueError(
                f"Eigenmode match on port {self.owner.port_index} extends to "
                f"normal index {self.expansion_plane_index}, beyond the "
                f"{opposite_face} PML interface at index {opposite_interface}. "
                "The matched buffer and reference plane cannot intersect the "
                "opposite longitudinal PML slab."
            )
        for local_axis, transverse_axis in enumerate(self.owner.transverse_axes):
            transverse_name = "xyz"[transverse_axis]
            lower_face = f"{transverse_name}0"
            upper_face = f"{transverse_name}max"
            lower = int(self.owner.transverse_start[local_axis])
            upper = int(self.owner.transverse_stop[local_axis])
            lower_pml = int(grid.pmls["thickness"][lower_face])
            upper_pml = int(grid.pmls["thickness"][upper_face])
            upper_interface = int(grid.size[transverse_axis]) - upper_pml
            if lower < lower_pml or upper > upper_interface:
                raise ValueError(
                    f"Eigenmode match on port {self.owner.port_index} has an "
                    f"aperture from {lower} to {upper} on the {transverse_name} "
                    "axis that intersects a transverse PML slab. Move the "
                    "aperture inside the PML interfaces or set the intersected "
                    "transverse PML thickness to zero."
                )

    def _section_slices(self, grid):
        slices = [slice(None), slice(None), slice(None)]
        if self.direction_sign > 0:
            slices[self.normal_axis] = slice(0, self.expansion_plane_index)
        else:
            slices[self.normal_axis] = slice(
                self.expansion_plane_index, int(grid.size[self.normal_axis])
            )
        for local_axis, global_axis in enumerate(self.owner.transverse_axes):
            slices[global_axis] = slice(
                int(self.owner.transverse_start[local_axis]),
                int(self.owner.transverse_stop[local_axis]),
            )
        return tuple(slices)

    def _validate_uniform_section(self, grid):
        section = np.asarray(grid.solid[self._section_slices(grid)])
        material_ids = np.unique(section)
        if section.size == 0 or material_ids.size != 1:
            raise ValueError(
                f"Eigenmode match on port {self.owner.port_index} requires one "
                "homogeneous material throughout its longitudinal buffer."
            )
        material_by_id = {int(material.numID): material for material in grid.materials}
        material = material_by_id[int(material_ids[0])]
        dispersive = getattr(material, "poles", 0) or any(
            name in str(getattr(material, "type", "")).lower()
            for name in ("debye", "lorentz", "drude")
        )
        values = (material.er, material.mr, material.se, material.sm)
        if (
            material.ID in ("pec", "pmc")
            or dispersive
            or not all(np.isscalar(value) and np.isfinite(value) for value in values)
            or float(material.er) <= 0
            or float(material.mr) <= 0
            or float(material.se) != 0
            or float(material.sm) != 0
        ):
            raise ValueError(
                f"Eigenmode match on port {self.owner.port_index} requires a "
                "finite, positive, lossless, nondispersive homogeneous fill; "
                f"material {material.ID!r} is unsupported."
            )

        # Compare every component-resolved Yee material slice. The modal solve
        # consumes all six E/H tensors and constraint masks, so checking only
        # tangential E would miss longitudinal wires, PMC changes, and other
        # special component updates inside the buffer.
        first_plane = self.boundary_index
        final_plane = self.expansion_plane_index
        step = 1 if first_plane <= final_plane else -1
        boundary_plane_indices = tuple(
            range(first_plane, final_plane + step, step)
        )
        # Normal E and tangential H occupy longitudinal half-cells, so their
        # high-coordinate outer array slot is padding. Tangential E and normal
        # H occupy integer planes and include the outer boundary itself.
        buffer_component_indices = (
            tuple(range(first_plane, min(final_plane + 1, int(grid.size[self.normal_axis]))))
            if self.direction_sign > 0
            else tuple(range(final_plane, first_plane))
        )
        local_to_global = (*self.owner.transverse_axes, self.normal_axis)
        component_material_ids = set()
        for field_kind, component_offset in (("E", 0), ("H", 3)):
            for local_axis, global_axis in enumerate(local_to_global):
                component = global_axis + component_offset
                reference = None
                is_integer_plane_component = (
                    field_kind == "E" and global_axis in self.owner.transverse_axes
                ) or (
                    field_kind == "H" and global_axis == self.normal_axis
                )
                plane_indices = (
                    boundary_plane_indices
                    if is_integer_plane_component
                    else buffer_component_indices
                )
                for plane_index in plane_indices:
                    component_ids = self._local_component_view(
                        grid.ID[component], local_axis, field_kind, plane_index
                    )
                    component_material_ids.update(
                        int(value) for value in np.unique(component_ids)
                    )
                    if reference is None:
                        reference = np.array(component_ids, copy=True)
                    elif not np.array_equal(component_ids, reference):
                        raise ValueError(
                            f"Eigenmode match on port {self.owner.port_index} "
                            "requires identical six-component Yee material and "
                            "constraint slices from the domain boundary to the "
                            f"modal reference plane; E/H component {component} "
                            f"changes at normal index {plane_index}."
                        )

        special_stencil_materials = sorted(
            {
                material_by_id[material_id].ID
                for material_id in component_material_ids
                if str(getattr(material_by_id[material_id], "type", "")).lower()
                == "thin-wire"
                or getattr(material_by_id[material_id], "thin_wire_axis", None)
                is not None
            }
        )
        if special_stencil_materials:
            raise ValueError(
                f"Eigenmode match on port {self.owner.port_index} does not "
                "support thin-wire custom update stencils in its "
                "longitudinal buffer; found material(s) "
                f"{', '.join(repr(value) for value in special_stencil_materials)}."
            )
        return material

    def _local_component_view(self, array, local_axis, field_kind, plane_index):
        u_slice, v_slice = self.owner._local_component_ranges(local_axis, field_kind)
        slices = [slice(None), slice(None), slice(None)]
        slices[self.normal_axis] = int(plane_index)
        slices[self.owner.transverse_axes[0]] = u_slice
        slices[self.owner.transverse_axes[1]] = v_slice
        return array[tuple(slices)]

    def electric_target_regions(self):
        """Return component IDs and half-open index boxes written at the boundary."""

        u_axis, v_axis = self.owner.transverse_axes
        u0, v0 = (int(value) for value in self.owner.transverse_start)
        u1, v1 = (int(value) for value in self.owner.transverse_stop)
        regions = []
        for component_axis in self.owner.transverse_axes:
            bounds = [(0, 0), (0, 0), (0, 0)]
            bounds[self.normal_axis] = (
                self.boundary_index,
                self.boundary_index + 1,
            )
            if component_axis == u_axis:
                bounds[u_axis] = (u0, u1)
                bounds[v_axis] = (v0, v1 + 1)
            else:
                bounds[u_axis] = (u0, u1 + 1)
                bounds[v_axis] = (v0, v1)
            regions.append((component_axis, tuple(bounds)))
        return tuple(regions)

    def _tangential_component_view(self, array, component_axis, plane_index):
        u_axis, v_axis = self.owner.transverse_axes
        u0, v0 = (int(value) for value in self.owner.transverse_start)
        u1, v1 = (int(value) for value in self.owner.transverse_stop)
        slices = [slice(None), slice(None), slice(None)]
        slices[self.normal_axis] = int(plane_index)
        if component_axis == u_axis:
            slices[u_axis] = slice(u0, u1)
            slices[v_axis] = slice(v0, v1 + 1)
        elif component_axis == v_axis:
            slices[u_axis] = slice(u0, u1 + 1)
            slices[v_axis] = slice(v0, v1)
        else:
            raise ValueError("matched modal boundaries use only tangential E")
        return array[tuple(slices)]

    def _flatten_tangential_fields(self, fields):
        return np.concatenate(
            [np.asarray(fields[axis]).ravel() for axis in self.owner.transverse_axes]
        )

    def _prepare_fixed_modal_basis(self, grid):
        monitor = self.owner.port_monitor
        anchor_count = len(monitor.anchor_e)
        if anchor_count == 0 or not self.mode_indices:
            raise ValueError("an eigenmode match requires solved modal anchors")
        midpoint = 0.5 * (float(self.owner.dft_start) + float(self.owner.dft_stop))
        representative = int(
            np.argmin(np.abs(np.asarray(monitor.anchor_frequencies) - midpoint))
        )
        rows = []
        for mode_position, mode_index in enumerate(self.mode_indices):
            representative_vector = self._flatten_tangential_fields(
                monitor.anchor_e[representative][mode_position]
            ).astype(np.complex128, copy=False)
            norm = float(np.linalg.norm(representative_vector))
            if not np.isfinite(norm) or norm <= 1e-300:
                raise ValueError(
                    f"Eigenmode match port {self.owner.port_index}, mode {mode_index} "
                    "has a zero or invalid tangential electric profile."
                )
            phase = -0.5 * np.angle(np.sum(representative_vector**2))
            representative_vector = representative_vector * np.exp(1j * phase)
            imaginary_residual = float(
                np.linalg.norm(np.imag(representative_vector))
                / np.linalg.norm(representative_vector)
            )
            if imaginary_residual > PROFILE_IMAGINARY_TOLERANCE:
                raise ValueError(
                    f"Eigenmode match port {self.owner.port_index}, mode {mode_index} "
                    f"has complex-profile residual {imaginary_residual:.3e}; the "
                    "first implementation requires an effectively real basis."
                )

            reference_unit = representative_vector / np.linalg.norm(
                representative_vector
            )
            for anchor_index in range(anchor_count):
                candidate = self._flatten_tangential_fields(
                    monitor.anchor_e[anchor_index][mode_position]
                ).astype(np.complex128, copy=False)
                candidate_norm = float(np.linalg.norm(candidate))
                if not np.isfinite(candidate_norm) or candidate_norm <= 1e-300:
                    raise ValueError(
                        f"Eigenmode match port {self.owner.port_index}, mode "
                        f"{mode_index} has an invalid anchor profile."
                    )
                candidate_unit = candidate / candidate_norm
                overlap = np.vdot(reference_unit, candidate_unit)
                overlap_magnitude = float(abs(overlap))
                if (
                    not np.isfinite(overlap_magnitude)
                    or overlap_magnitude < PROFILE_OVERLAP_TOLERANCE
                ):
                    raise ValueError(
                        f"Eigenmode match port {self.owner.port_index}, mode "
                        f"{mode_index} changes profile across frequency "
                        f"(minimum overlap {overlap_magnitude:.6f}); a causal "
                        "frequency-dependent projection is not implemented."
                    )
                candidate_unit = candidate_unit * np.exp(-1j * np.angle(overlap))
                candidate_imaginary_residual = float(
                    np.linalg.norm(np.imag(candidate_unit))
                    / np.linalg.norm(candidate_unit)
                )
                if candidate_imaginary_residual > PROFILE_IMAGINARY_TOLERANCE:
                    raise ValueError(
                        f"Eigenmode match port {self.owner.port_index}, mode "
                        f"{mode_index} has complex-profile residual "
                        f"{candidate_imaginary_residual:.3e} at anchor "
                        f"{monitor.anchor_frequencies[anchor_index]:g} Hz; the "
                        "first implementation requires an effectively real "
                        "basis at every anchor."
                    )
            rows.append(np.real(representative_vector))

        # Keep the small spatial projection in float64 even for a float32 FDTD
        # grid. Row normalization makes the condition estimate measure modal
        # linear independence rather than arbitrary power-normalized amplitude.
        basis = np.ascontiguousarray(np.vstack(rows), dtype=np.float64)
        expected_size = sum(
            self._tangential_component_view(
                (grid.Ex, grid.Ey, grid.Ez)[axis], axis, self.expansion_plane_index
            ).size
            for axis in self.owner.transverse_axes
        )
        if basis.shape[1] != expected_size:
            raise ValueError(
                "matched modal profile shapes do not match the Yee boundary aperture"
            )
        row_norms = np.linalg.norm(basis, axis=1)
        if np.any(~np.isfinite(row_norms)) or np.any(row_norms <= 1e-300):
            raise ValueError("matched modal basis contains an invalid row norm")
        normalized_basis = basis / row_norms[:, np.newaxis]
        gram = normalized_basis @ normalized_basis.T
        condition = float(np.linalg.cond(gram))
        condition_limit = min(
            GRAM_CONDITION_LIMIT,
            PROJECTION_RELATIVE_ERROR_BUDGET / np.finfo(self.real_dtype).eps,
        )
        if not np.isfinite(condition) or condition >= condition_limit:
            raise ValueError(
                f"Eigenmode match port {self.owner.port_index} electric modal "
                f"Gram matrix is ill-conditioned ({condition:.3e}; limit "
                f"{condition_limit:.3e} for {self.real_dtype.name})."
            )
        normalized_dual = np.linalg.solve(gram, normalized_basis)
        dual = normalized_dual / row_norms[:, np.newaxis]
        identity_error = float(
            np.max(np.abs(dual @ basis.T - np.eye(len(self.mode_indices))))
        )
        if (
            not np.isfinite(identity_error)
            or identity_error > PROJECTION_RELATIVE_ERROR_BUDGET
        ):
            raise ValueError(
                f"Eigenmode match port {self.owner.port_index} modal dual failed "
                f"its identity check ({identity_error:.3e})."
            )
        return basis, np.ascontiguousarray(dual, dtype=np.float64)

    def _prepare_cutoff_wavenumbers(self):
        monitor = self.owner.port_monitor
        frequencies = np.asarray(monitor.anchor_frequencies, dtype=np.float64)
        neff = np.asarray(monitor.anchor_neff, dtype=np.complex128)
        cutoff = []
        for mode_position, mode_index in enumerate(self.mode_indices):
            omega = 2 * np.pi * frequencies
            beta = omega * neff[:, mode_position] / config.c
            cutoff_squared = (omega / self.wave_speed) ** 2 - beta**2
            scale = np.maximum((omega / self.wave_speed) ** 2, 1.0)
            if np.any(np.abs(np.imag(cutoff_squared)) > 1e-6 * scale):
                raise ValueError(
                    f"Eigenmode match port {self.owner.port_index}, mode {mode_index} "
                    "has a complex cutoff relation; lossy/complex propagation is "
                    "not supported by the scalar matched operator."
                )
            values = np.real(cutoff_squared)
            negative_tolerance = 1e-6 * np.max(scale)
            if np.any(values < -negative_tolerance):
                raise ValueError(
                    f"Eigenmode match port {self.owner.port_index}, mode {mode_index} "
                    "does not yield a physical non-negative cutoff wavenumber."
                )
            values = np.maximum(values, 0.0)
            representative = float(np.median(values))
            spread_scale = max(
                representative, 1e-8 * float(np.max(scale)), 1e-300
            )
            relative_spread = float(
                np.max(np.abs(values - representative)) / spread_scale
            )
            if relative_spread > CUTOFF_RELATIVE_SPREAD_TOLERANCE:
                raise ValueError(
                    f"Eigenmode match port {self.owner.port_index}, mode {mode_index} "
                    f"has cutoff-squared relative spread {relative_spread:.3e}; "
                    "the mode is not described by one fixed scalar cutoff."
                )
            cutoff.append(np.sqrt(representative))
        return np.ascontiguousarray(cutoff, dtype=np.float64)

    def _prepare_translation_filters(self, grid):
        sample_count = int(grid.iterations) + 1
        depth_kernels = []
        double_depth_kernels = []
        for cutoff in self.cutoff_wavenumbers:
            one_cell = one_cell_impulse_response(
                sample_count,
                grid.dt,
                grid.dl[self.normal_axis],
                cutoff,
                wave_speed=self.wave_speed,
                dtype=self.real_dtype,
            )
            depth, double_depth = cascaded_impulse_responses(
                one_cell,
                self.depth_cells,
                sample_count=sample_count,
                dtype=self.real_dtype,
            )
            depth_kernels.append(depth)
            double_depth_kernels.append(double_depth)
        self.translation_filter = CausalModalFIR(
            np.vstack(depth_kernels), dtype=self.real_dtype
        )
        if self.source_mode_position is not None:
            self.source_cancellation_filter = CausalModalFIR(
                double_depth_kernels[self.source_mode_position],
                dtype=self.real_dtype,
            )

    def _read_expansion_field(self, grid):
        fields = (grid.Ex, grid.Ey, grid.Ez)
        return np.concatenate(
            [
                np.asarray(
                    self._tangential_component_view(
                        fields[axis], axis, self.expansion_plane_index
                    )
                ).ravel()
                for axis in self.owner.transverse_axes
            ]
        ).astype(np.float64, copy=False)

    def _write_boundary_field(self, grid, flattened):
        fields = (grid.Ex, grid.Ey, grid.Ez)
        offset = 0
        for axis in self.owner.transverse_axes:
            target = self._tangential_component_view(
                fields[axis], axis, self.boundary_index
            )
            size = target.size
            target[...] = flattened[offset : offset + size].reshape(target.shape)
            offset += size

    def reset(self):
        """Reset causal histories and restore the time-zero boundary field."""

        if self.translation_filter is not None:
            self.translation_filter.reset()
        if self.source_cancellation_filter is not None:
            self.source_cancellation_filter.reset()

    def update_electric_boundary(self, sample_index, grid):
        """Impose the matched boundary at electric time ``sample_index * dt``."""

        if self.translation_filter is None:
            return
        modal_voltage = self.dual_basis @ self._read_expansion_field(grid)
        translated_voltage = self.translation_filter.step(modal_voltage)
        boundary_field = translated_voltage @ self.basis
        if self.source_mode_position is not None:
            time = float(sample_index) * grid.dt
            source_value = (
                self.owner._waveform_value(time, grid)
                if self.owner._source_is_active(time)
                else 0.0
            )
            twice_translated = self.source_cancellation_filter.step(
                np.asarray((source_value,), dtype=self.real_dtype)
            )[0]
            boundary_field = boundary_field + (
                source_value - twice_translated
            ) * self.basis[self.source_mode_position]
        self._write_boundary_field(grid, boundary_field)


def _validate_nonoverlapping_boundaries(boundaries):
    for first_index, first in enumerate(boundaries):
        for second in boundaries[first_index + 1 :]:
            for first_component, first_bounds in first.electric_target_regions():
                for second_component, second_bounds in second.electric_target_regions():
                    if first_component != second_component:
                        continue
                    intersects = all(
                        max(first_axis[0], second_axis[0])
                        < min(first_axis[1], second_axis[1])
                        for first_axis, second_axis in zip(
                            first_bounds, second_bounds
                        )
                    )
                    if intersects:
                        raise ValueError(
                            f"Eigenmode matches on ports {first.owner.port_index} "
                            f"and {second.owner.port_index} both write "
                            f"E{'xyz'[first_component]} boundary nodes. Matched "
                            "port apertures must not overlap, including shared "
                            "Yee edges on adjacent domain faces."
                        )


def initialise_eigenmode_matches(grid):
    """Create and initialise matched boundaries after modal port solves."""

    owners = [*grid.eigenmodesources, *grid.eigenmodereceivers]
    matched_owners = [owner for owner in owners if owner.match_depth_cells is not None]
    if not matched_owners:
        return
    grid.eigenmodematches.clear()
    boundaries = [MatchedEigenmodeBoundary(owner, grid) for owner in matched_owners]
    _validate_nonoverlapping_boundaries(boundaries)
    for boundary in boundaries:
        owner = boundary.owner
        owner.matched_boundary = boundary
        # The matched source has no TF/SF jump at its interior reference plane;
        # sample H on the same upstream side used by an ordinary passive port.
        owner.port_monitor.magnetic_side = -1
    grid.eigenmodematches.extend(boundaries)
    for boundary in grid.eigenmodematches:
        boundary.reset()
        boundary.update_electric_boundary(0, grid)
        logger.info(
            f"Eigenmode port {boundary.owner.port_index} matched at boundary "
            f"index {boundary.boundary_index} with expansion-plane index "
            f"{boundary.expansion_plane_index}, depth {boundary.depth_cells} "
            f"cell(s), and modes {boundary.mode_indices}."
        )

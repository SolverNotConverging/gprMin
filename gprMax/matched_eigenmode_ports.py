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

"""Power-adjoint modal ADE boundary for lossless waveguides.

The boundary terminates a single 3D mode with a raw-Yee power-adjoint E/H
pairing and a passive trapezoidal half-cell state. It permits longitudinally
uniform lossless, nondispersive multi-material guides.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

import gprMax.config as config
from gprMax.modal_admittance import synthesize_passive_admittance
from gprMax.modal_admittance import yee_staggered_characteristic_admittance
from gprMax.modal_admittance_ade import bilinear_prewarp_angular_frequency
from gprMax.modal_admittance_ade import RationalModalAdmittanceADE


__all__ = [
    "FixedBasisAdmittanceSamples",
    "MatchedEigenmodeBoundary",
    "constant_modal_admittance_step",
    "fixed_basis_admittance_samples",
    "initialise_eigenmode_matches",
]


logger = logging.getLogger(__name__)
PROFILE_IMAGINARY_TOLERANCE = 1e-8
PROFILE_OVERLAP_TOLERANCE = 0.999
PROPAGATION_IMAGINARY_TOLERANCE = 1e-8
MATCHED_BROADBAND_WARNING_FRACTION = 0.25
GRAM_CONDITION_LIMIT = 1e10
SCALAR_FIXED_BASIS_RUNTIME_RESIDUAL_LIMIT = 1e-3
# The rational path is exercised explicitly by experimental integration tests
# until coupled Yee/ADE energy and long-time regressions justify making it the
# production default.
ENABLE_RATIONAL_ADMITTANCE_RUNTIME = False


@dataclass(frozen=True)
class FixedBasisAdmittanceSamples:
    """Scalar modal admittance samples expressed in one fixed power gauge.

    The electric and magnetic anchor arrays are deliberately projected against
    the *centre* magnetic and electric covectors, respectively. Independently
    normalizing every anchor would make every characteristic admittance equal
    to one and remove the frequency dependence that a rational model must fit.
    """

    admittances: np.ndarray
    voltages: np.ndarray
    currents: np.ndarray
    electric_residuals: np.ndarray
    magnetic_residuals: np.ndarray


def fixed_basis_admittance_samples(
    electric_basis,
    magnetic_covector,
    anchor_electric,
    anchor_magnetic_covectors,
) -> FixedBasisAdmittanceSamples:
    """Project anchor E/H pairs into a fixed scalar modal power coordinate.

    All products are unconjugated because anchor phase is part of the complex
    travelling-wave amplitude. ``electric_basis`` and ``magnetic_covector``
    must describe the same centre-frequency forward mode and have positive
    real raw-Yee power pairing. A common complex rescaling of an anchor E/H
    pair cancels exactly from the returned admittance.
    """

    basis = np.asarray(electric_basis, dtype=np.complex128)
    covector = np.asarray(magnetic_covector, dtype=np.complex128)
    electric = np.asarray(anchor_electric, dtype=np.complex128)
    magnetic = np.asarray(anchor_magnetic_covectors, dtype=np.complex128)
    if basis.ndim != 1 or covector.ndim != 1 or basis.shape != covector.shape:
        raise ValueError(
            "fixed modal electric basis and magnetic covector must be equal-length "
            "one-dimensional arrays"
        )
    if electric.ndim != 2 or magnetic.ndim != 2 or electric.shape != magnetic.shape:
        raise ValueError(
            "anchor electric and magnetic arrays must be equal-shape two-dimensional "
            "arrays"
        )
    if electric.shape[1] != basis.size or electric.shape[0] == 0:
        raise ValueError("anchor modal arrays are incompatible with the fixed basis")
    if not all(
        np.all(np.isfinite(values))
        for values in (basis, covector, electric, magnetic)
    ):
        raise ValueError("fixed-basis modal admittance inputs must be finite")

    power = complex(np.dot(basis, covector))
    basis_norm = float(np.linalg.norm(basis))
    covector_norm = float(np.linalg.norm(covector))
    power_scale = basis_norm * covector_norm
    if (
        not np.isfinite(power)
        or not np.isfinite(power_scale)
        or power_scale <= 1e-300
        or power.real <= 64 * np.finfo(np.float64).eps * power_scale
        or abs(power.imag) > 64 * np.finfo(np.float64).eps * power_scale
    ):
        raise ValueError(
            "fixed modal electric/magnetic basis must have positive real power pairing"
        )
    power = float(power.real)

    electric_norms = np.linalg.norm(electric, axis=1)
    magnetic_norms = np.linalg.norm(magnetic, axis=1)
    if np.any(electric_norms <= 1e-300) or np.any(magnetic_norms <= 1e-300):
        raise ValueError("anchor modal electric and magnetic fields must be nonzero")
    voltage_numerators = electric @ covector
    voltage_tolerances = (
        64 * np.finfo(np.float64).eps * electric_norms * covector_norm
    )
    if np.any(np.abs(voltage_numerators) <= voltage_tolerances):
        raise ValueError("an anchor has zero fixed-basis modal voltage")
    voltages = voltage_numerators / power
    currents = magnetic @ basis / power
    admittances = currents / voltages

    electric_residuals = np.linalg.norm(
        electric - voltages[:, None] * basis[None, :], axis=1
    ) / electric_norms
    magnetic_residuals = np.linalg.norm(
        magnetic - currents[:, None] * covector[None, :], axis=1
    ) / magnetic_norms
    if not all(
        np.all(np.isfinite(values))
        for values in (
            voltages,
            currents,
            admittances,
            electric_residuals,
            magnetic_residuals,
        )
    ):
        raise ValueError("fixed-basis modal admittance projection produced non-finite data")

    return FixedBasisAdmittanceSamples(
        admittances=np.ascontiguousarray(admittances),
        voltages=np.ascontiguousarray(voltages),
        currents=np.ascontiguousarray(currents),
        electric_residuals=np.ascontiguousarray(electric_residuals),
        magnetic_residuals=np.ascontiguousarray(magnetic_residuals),
    )


def _positive_integer(value, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be an integer greater than zero")
    return int(value)


def constant_modal_admittance_step(
    previous_voltage,
    magnetic_coefficient,
    incident_voltage,
    tau_over_dt,
):
    """Advance the trapezoidal constant-admittance modal load by one sample.

    All inputs are generalized modal coordinates with normalized positive
    characteristic admittance. ``magnetic_coefficient`` is direction-normalized
    so that it equals ``a - b`` while electric voltage equals ``a + b``.
    ``tau_over_dt`` must be positive; this represents the terminal half-cell's
    electric storage and removes the algebraic load's Nyquist computational
    mode.
    """

    previous = np.asarray(previous_voltage)
    magnetic = np.asarray(magnetic_coefficient)
    incident = np.asarray(incident_voltage)
    ratio = np.asarray(tau_over_dt)
    try:
        previous, magnetic, incident, ratio = np.broadcast_arrays(
            previous, magnetic, incident, ratio
        )
    except ValueError as exc:
        raise ValueError("modal admittance step inputs must be broadcast-compatible") from exc
    if not all(
        np.all(np.isfinite(values))
        for values in (previous, magnetic, incident, ratio)
    ):
        raise ValueError("modal admittance step inputs must contain only finite values")
    if np.any(ratio <= 0):
        raise ValueError("tau_over_dt must be positive")
    return (
        (ratio - 0.5) * previous + 2.0 * incident - magnetic
    ) / (ratio + 0.5)


class _MatchedBoundaryGeometry:
    """Geometry and material validation helpers for the modal ADE boundary."""

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
        if section.size == 0:
            raise ValueError(
                f"Eigenmode match on port {self.owner.port_index} has an empty "
                "longitudinal buffer aperture."
            )
        normal_planes = np.moveaxis(section, self.normal_axis, 0)
        reference_section = normal_planes[0]
        for local_index, candidate_section in enumerate(normal_planes[1:], start=1):
            if not np.array_equal(candidate_section, reference_section):
                normal_index = (
                    local_index
                    if self.direction_sign > 0
                    else self.expansion_plane_index + local_index
                )
                raise ValueError(
                    f"Eigenmode match on port {self.owner.port_index} requires "
                    "one longitudinally uniform material cross-section from the "
                    "domain boundary to the modal reference plane; the cell "
                    f"material section changes at normal index {normal_index}."
                )

        material_ids = {int(value) for value in np.unique(section)}
        material_by_id = {int(material.numID): material for material in grid.materials}

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
            tuple(range(first_plane, min(final_plane, int(grid.size[self.normal_axis]))))
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

        material_ids.update(component_material_ids)
        section_materials = []
        for material_id in sorted(material_ids):
            material = material_by_id[material_id]
            section_materials.append(material)
            if material.ID in ("pec", "pmc"):
                continue
            dispersive = getattr(material, "poles", 0) or any(
                name in str(getattr(material, "type", "")).lower()
                for name in ("debye", "lorentz", "drude")
            )
            values = (material.er, material.mr, material.se, material.sm)
            if (
                dispersive
                or not all(
                    np.isscalar(value) and np.isfinite(value) for value in values
                )
                or float(material.er) <= 0
                or float(material.mr) <= 0
                or float(material.se) != 0
                or float(material.sm) != 0
            ):
                raise ValueError(
                    f"Eigenmode match on port {self.owner.port_index} requires "
                    "positive, lossless, nondispersive section materials; ideal "
                    f"PEC/PMC constraints are allowed, but material {material.ID!r} is unsupported."
                )
        return tuple(section_materials)

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


class MatchedEigenmodeBoundary(_MatchedBoundaryGeometry):
    """Power-adjoint, constant-modal-admittance termination.

    The ordinary FDTD cells between the eigenmode reference plane and the
    outer face provide the propagation delay. At the outer face, the solved
    centre-frequency electric and magnetic mode pair defines generalized
    voltage and current coordinates. In those coordinates the normalized
    characteristic admittance is one, independent of the arbitrary scaling
    used by the eigensolver.

    A positive half-cell storage time constant completes the Yee boundary in
    a trapezoidal, energy-passive form. For a passive mode its update is

    ``tau * (V[n+1] - V[n]) / dt + (V[n+1] + V[n]) / 2 = -Q[n+1/2]``.

    An active matched generator replaces the right-hand side by ``2a - Q``.
    Modal magnetic extraction and electric reconstruction are a discrete
    power-adjoint pair, which is the essential distinction from an electric
    least-squares feedback boundary.
    """

    formulation = "PowerAdjointModalAdmittanceADE"

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
        if len(self.mode_indices) != 1:
            raise ValueError(
                "The modal ADE matched boundary supports exactly one retained "
                "mode; use an ordinary eigenmode port followed by PML for a "
                "multimode termination."
            )
        if self.owner.invariant_axis is not None:
            raise ValueError(
                "The modal ADE matched boundary supports 3D ports only; use an "
                "ordinary eigenmode port followed by PML for a 2D model."
            )
        self._validate_location(grid)
        self._validate_uniform_section(grid)

        (
            self.basis,
            self.modal_hu,
            self.modal_hv,
        ) = self._prepare_power_modal_basis(grid)
        self.power_gram = self._prepare_power_gram(grid)
        (
            self.group_velocities,
            self.modal_time_constants,
        ) = self._prepare_modal_time_constants(grid)
        self.normalized_admittances = np.ones(
            len(self.mode_indices), dtype=np.float64
        )
        self.fixed_basis_admittance_samples = None
        self.staggered_characteristic_admittance_samples = np.empty(
            0, dtype=np.complex128
        )
        self.rational_admittance_fit = None
        self.rational_admittance_fit_error = None
        self.rational_training_frequencies = np.empty(0, dtype=np.float64)
        self.rational_validation_frequencies = np.empty(0, dtype=np.float64)
        self.rational_discrete_poles = np.empty(0, dtype=np.complex128)
        self.rational_ade = None
        self.rational_runtime_enabled = False
        self.rational_runtime_rejection = (
            "experimental rational runtime is disabled"
        )
        try:
            self.fixed_basis_admittance_samples = (
                self._prepare_fixed_basis_admittance_samples(grid)
            )
            if ENABLE_RATIONAL_ADMITTANCE_RUNTIME:
                largest_profile_residual = max(
                    float(
                        np.max(
                            self.fixed_basis_admittance_samples.electric_residuals,
                            initial=0.0,
                        )
                    ),
                    float(
                        np.max(
                            self.fixed_basis_admittance_samples.magnetic_residuals,
                            initial=0.0,
                        )
                    ),
                )
                if (
                    largest_profile_residual
                    > SCALAR_FIXED_BASIS_RUNTIME_RESIDUAL_LIMIT
                ):
                    raise ValueError(
                        "experimental scalar rational matched runtime requires "
                        "fixed-basis E/H residual no larger than "
                        f"{SCALAR_FIXED_BASIS_RUNTIME_RESIDUAL_LIMIT:.3e}; "
                        f"measured {largest_profile_residual:.3e}. The section "
                        "requires a higher-dimensional discrete DtN model."
                    )
            self.rational_admittance_fit = self._prepare_shadow_rational_fit(grid)
            if ENABLE_RATIONAL_ADMITTANCE_RUNTIME:
                if self.rational_admittance_fit is None:
                    raise ValueError(
                        "experimental rational runtime was requested but no "
                        "certified fitted model is available"
                    )
                self.rational_ade = RationalModalAdmittanceADE(
                    self.rational_admittance_fit.model,
                    dt=float(grid.dt),
                    half_cell_storage=float(self.modal_time_constants[0]),
                )
                self.rational_runtime_enabled = True
                self.rational_runtime_rejection = ""
        except Exception as exc:
            if ENABLE_RATIONAL_ADMITTANCE_RUNTIME:
                raise
            logger.warning(
                f"Eigenmode admittance match port {self.owner.port_index} shadow "
                f"rational diagnostics failed without affecting the production "
                f"boundary: {exc}"
            )
            self.fixed_basis_admittance_samples = None
            self.staggered_characteristic_admittance_samples = np.empty(
                0, dtype=np.complex128
            )
            self.rational_admittance_fit = None
            self.rational_admittance_fit_error = str(exc)
            self.rational_ade = None
        self.modal_voltage_state = np.zeros(
            len(self.mode_indices), dtype=np.float64
        )

        self.source_mode_position = None
        if self.owner.port_monitor.is_source:
            self.source_mode_position = self.mode_indices.index(
                int(self.owner.mode_index)
            )

    @staticmethod
    def _common_real_phase(electric, magnetic, transverse_axes):
        impedance = float(config.sim_config.em_consts["z0"])
        unconjugated_energy = 0.0j
        total_energy = 0.0
        for axis in transverse_axes:
            efield = np.asarray(electric[axis], dtype=np.complex128)
            hfield = impedance * np.asarray(magnetic[axis], dtype=np.complex128)
            unconjugated_energy += np.sum(efield * efield)
            unconjugated_energy += np.sum(hfield * hfield)
            total_energy += float(np.vdot(efield, efield).real)
            total_energy += float(np.vdot(hfield, hfield).real)
        if (
            not np.isfinite(total_energy)
            or total_energy <= 1e-300
            or not np.isfinite(unconjugated_energy)
        ):
            raise ValueError("matched modal E/H basis has zero or invalid energy")
        return np.exp(-0.5j * np.angle(unconjugated_energy))

    @staticmethod
    def _real_profile_residual(electric, magnetic, transverse_axes):
        impedance = float(config.sim_config.em_consts["z0"])
        total_energy = 0.0
        imaginary_energy = 0.0
        for axis in transverse_axes:
            efield = np.asarray(electric[axis], dtype=np.complex128)
            hfield = impedance * np.asarray(magnetic[axis], dtype=np.complex128)
            total_energy += float(np.vdot(efield, efield).real)
            total_energy += float(np.vdot(hfield, hfield).real)
            imaginary_energy += float(np.vdot(np.imag(efield), np.imag(efield)).real)
            imaginary_energy += float(
                np.vdot(np.imag(hfield), np.imag(hfield)).real
            )
        if not np.isfinite(total_energy) or total_energy <= 1e-300:
            return np.inf
        return float(np.sqrt(imaginary_energy / total_energy))

    def _centre_anchor_index(self):
        monitor = self.owner.port_monitor
        midpoint = 0.5 * (float(self.owner.dft_start) + float(self.owner.dft_stop))
        frequencies = np.asarray(monitor.anchor_frequencies, dtype=np.float64)
        matches = np.flatnonzero(
            np.isclose(
                frequencies,
                midpoint,
                rtol=8 * np.finfo(float).eps,
                atol=0.0,
            )
        )
        if matches.size != 1:
            raise ValueError(
                f"Eigenmode admittance match port {self.owner.port_index} requires "
                f"exactly one solved modal basis at the band centre {midpoint:g} Hz."
            )
        self.basis_frequency = float(frequencies[int(matches[0])])
        return int(matches[0])

    def _prepare_power_modal_basis(self, grid):
        monitor = self.owner.port_monitor
        if not self.mode_indices or not monitor.anchor_e:
            raise ValueError("an eigenmode admittance match requires solved modal anchors")
        representative = self._centre_anchor_index()
        anchor_count = len(monitor.anchor_e)
        u_axis, v_axis = self.owner.transverse_axes
        electric_rows = []
        modal_hu = []
        modal_hv = []
        minimum_overlaps = []

        for mode_position, mode_index in enumerate(self.mode_indices):
            centre_e = [
                np.asarray(field, dtype=np.complex128)
                for field in monitor.anchor_e[representative][mode_position]
            ]
            centre_h = [
                np.asarray(field, dtype=np.complex128)
                for field in monitor.anchor_h[representative][mode_position]
            ]
            phase = self._common_real_phase(
                centre_e, centre_h, self.owner.transverse_axes
            )
            centre_e = [field * phase for field in centre_e]
            centre_h = [field * phase for field in centre_h]
            residual = self._real_profile_residual(
                centre_e, centre_h, self.owner.transverse_axes
            )
            if residual > PROFILE_IMAGINARY_TOLERANCE:
                raise ValueError(
                    f"Eigenmode admittance match port {self.owner.port_index}, "
                    f"mode {mode_index} has centre E/H complex-profile residual "
                    f"{residual:.3e}; an effectively real modal pair is required."
                )

            power = float(np.real(self.owner._modal_cross_power(centre_e, centre_h, grid)))
            if not np.isfinite(power) or power <= 1e-12:
                raise ValueError(
                    f"Eigenmode admittance match port {self.owner.port_index}, "
                    f"mode {mode_index} has invalid forward modal power {power:g}."
                )
            scale = 1.0 / np.sqrt(power)
            centre_e = [field * scale for field in centre_e]
            centre_h = [field * scale for field in centre_h]

            mode_minimum_overlap = 1.0
            for anchor_index in range(anchor_count):
                candidate_e = [
                    np.asarray(field, dtype=np.complex128)
                    for field in monitor.anchor_e[anchor_index][mode_position]
                ]
                candidate_h = [
                    np.asarray(field, dtype=np.complex128)
                    for field in monitor.anchor_h[anchor_index][mode_position]
                ]
                overlap = self.owner._modal_overlap(
                    centre_e, centre_h, candidate_e, candidate_h
                )
                overlap_magnitude = float(abs(overlap))
                mode_minimum_overlap = min(
                    mode_minimum_overlap, overlap_magnitude
                )
                if (
                    not np.isfinite(overlap_magnitude)
                    or overlap_magnitude < PROFILE_OVERLAP_TOLERANCE
                ):
                    raise ValueError(
                        f"Eigenmode admittance match port {self.owner.port_index}, "
                        f"mode {mode_index} changes E/H profile across frequency "
                        f"(minimum overlap {overlap_magnitude:.6f}); narrow the "
                        "band or use an ordinary eigenmode port followed by PML."
                    )
                # ``centre_e/h`` are already phase-rotated. The overlap phase
                # is therefore the complete candidate-to-centre correction;
                # applying ``phase`` again would rotate the centre anchor twice.
                align = np.exp(-1j * np.angle(overlap))
                candidate_e = [field * align for field in candidate_e]
                candidate_h = [field * align for field in candidate_h]
                candidate_residual = self._real_profile_residual(
                    candidate_e, candidate_h, self.owner.transverse_axes
                )
                if candidate_residual > PROFILE_IMAGINARY_TOLERANCE:
                    raise ValueError(
                        f"Eigenmode admittance match port {self.owner.port_index}, "
                        f"mode {mode_index} has E/H complex-profile residual "
                        f"{candidate_residual:.3e} at anchor "
                        f"{monitor.anchor_frequencies[anchor_index]:g} Hz."
                    )

            electric_rows.append(self._flatten_tangential_fields(centre_e))
            modal_hu.append(centre_h[u_axis])
            modal_hv.append(centre_h[v_axis])
            minimum_overlaps.append(mode_minimum_overlap)

        self.minimum_profile_overlaps = np.asarray(
            minimum_overlaps, dtype=np.float64
        )
        basis = np.ascontiguousarray(np.real(np.vstack(electric_rows)), dtype=np.float64)
        return (
            basis,
            np.ascontiguousarray(np.real(modal_hu), dtype=np.float64),
            np.ascontiguousarray(np.real(modal_hv), dtype=np.float64),
        )

    def _port_measure(self, grid):
        u_axis, v_axis = self.owner.transverse_axes
        # The admittance MVP is 3D-only. Native Yee edge pairing uses one
        # transverse cell area; the 2D monitor's TE factor-of-two correction
        # belongs only to its cell-averaged DFT quadrature.
        return float(grid.dl[u_axis] * grid.dl[v_axis])

    def _prepare_power_gram(self, grid):
        handedness = self.owner._modal_basis_handedness()
        measure = self._port_measure(grid)
        # The Yee summation-by-parts boundary term pairs raw E_u nodes with
        # raw H_v nodes and raw E_v nodes with raw H_u nodes. Cell averaging,
        # although appropriate for the output DFT quadrature, would introduce
        # cross terms and break the extraction/reconstruction adjoint identity.
        outward_covectors = np.concatenate(
            (
                handedness * self.modal_hv.reshape(len(self.mode_indices), -1),
                -handedness * self.modal_hu.reshape(len(self.mode_indices), -1),
            ),
            axis=1,
        )
        outward_covectors *= measure
        gram = self.basis @ outward_covectors.T
        reciprocity_scale = max(float(np.linalg.norm(gram)), 1e-300)
        reciprocity_residual = float(np.linalg.norm(gram - gram.T) / reciprocity_scale)
        if reciprocity_residual > 1e-8:
            raise ValueError(
                f"Eigenmode admittance match port {self.owner.port_index} modal "
                f"power Gram reciprocity residual is {reciprocity_residual:.3e}."
            )
        symmetric = 0.5 * (gram + gram.T)
        eigenvalues = np.linalg.eigvalsh(symmetric)
        scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
        tolerance = 64 * np.finfo(np.float64).eps * scale
        if np.any(eigenvalues <= tolerance):
            raise ValueError(
                f"Eigenmode admittance match port {self.owner.port_index} has "
                "a non-positive modal power Gram matrix."
            )
        condition = float(np.linalg.cond(symmetric))
        if not np.isfinite(condition) or condition >= GRAM_CONDITION_LIMIT:
            raise ValueError(
                f"Eigenmode admittance match port {self.owner.port_index} modal "
                f"power Gram matrix is ill-conditioned ({condition:.3e})."
            )
        return np.ascontiguousarray(symmetric, dtype=np.float64)

    def _prepare_fixed_basis_admittance_samples(self, grid):
        """Measure anchor admittance in the centre-frequency power gauge.

        This is initially diagnostic-only: the production timestep continues
        to use the order-zero normalized admittance until the rational fitter
        and positive-real enforcement are independently certified.
        """

        monitor = self.owner.port_monitor
        u_axis, v_axis = self.owner.transverse_axes
        handedness = self.owner._modal_basis_handedness()
        measure = self._port_measure(grid)
        centre_covector = measure * np.concatenate(
            (
                handedness * self.modal_hv[0].ravel(),
                -handedness * self.modal_hu[0].ravel(),
            )
        )
        anchor_electric = []
        anchor_magnetic_covectors = []
        for anchor_index in range(len(monitor.anchor_e)):
            electric = monitor.anchor_e[anchor_index][0]
            magnetic = monitor.anchor_h[anchor_index][0]
            anchor_electric.append(self._flatten_tangential_fields(electric))
            anchor_magnetic_covectors.append(
                measure
                * np.concatenate(
                    (
                        handedness * np.asarray(magnetic[v_axis]).ravel(),
                        -handedness * np.asarray(magnetic[u_axis]).ravel(),
                    )
                )
            )

        samples = fixed_basis_admittance_samples(
            self.basis[0],
            centre_covector,
            np.asarray(anchor_electric),
            np.asarray(anchor_magnetic_covectors),
        )
        centre_index = self._centre_anchor_index()
        if not np.isclose(
            samples.admittances[centre_index],
            1.0 + 0.0j,
            rtol=2e-10,
            atol=2e-12,
        ):
            raise ValueError(
                f"Eigenmode admittance match port {self.owner.port_index} centre "
                "anchor does not produce unit admittance in the fixed power gauge."
            )
        logger.info(
            f"Eigenmode admittance match port {self.owner.port_index} fixed-basis "
            f"anchor admittance spans {np.min(np.abs(samples.admittances)):g} to "
            f"{np.max(np.abs(samples.admittances)):g}; rational synthesis is "
            "evaluated in shadow mode and remains disabled for runtime use."
        )
        return samples

    def _prepare_shadow_rational_fit(self, grid):
        """Fit and certify a rational load for shadow or experimental use.

        The shadow fit exercises sample extraction, vector fitting, validation,
        and global passivity enforcement on real modal data. Runtime use is
        controlled separately by ``ENABLE_RATIONAL_ADMITTANCE_RUNTIME``.
        """

        frequencies = np.asarray(
            self.owner.port_monitor.anchor_frequencies, dtype=np.float64
        )
        physical_omega = 2 * np.pi * frequencies
        modal_neff = np.asarray(
            self.owner.port_monitor.anchor_neff[:, 0], dtype=np.complex128
        )
        beta = physical_omega * np.real(modal_neff) / config.c
        values = yee_staggered_characteristic_admittance(
            physical_omega,
            beta,
            self.fixed_basis_admittance_samples.admittances,
            normal_spacing=float(grid.dl[self.normal_axis]),
            dt=float(grid.dt),
            half_cell_storage=float(self.modal_time_constants[0]),
        )
        self.staggered_characteristic_admittance_samples = values
        centre_index = self._centre_anchor_index()
        validation_indices = []
        candidate_orders = (0,)
        if frequencies.size >= 9:
            maximum_supported_order = min(
                8,
                2 * max(0, (frequencies.size - 5) // 4),
            )
            candidate_orders = tuple(
                range(0, maximum_supported_order + 1, 2)
            )
            largest_order = candidate_orders[-1]
            required_training = (
                1 if largest_order == 0 else 2 * largest_order + 4
            )
            maximum_holdout = frequencies.size - required_training
            candidates = [
                index
                for index in range(1, frequencies.size - 1)
                if index != centre_index
            ]
            if maximum_holdout > 0 and candidates:
                positions = np.linspace(
                    0,
                    len(candidates) - 1,
                    min(maximum_holdout, len(candidates)),
                    dtype=int,
                )
                validation_indices = sorted({candidates[int(value)] for value in positions})
        training_mask = np.ones(frequencies.size, dtype=bool)
        training_mask[validation_indices] = False
        self.rational_training_frequencies = np.ascontiguousarray(
            frequencies[training_mask], dtype=np.float64
        )
        self.rational_validation_frequencies = np.ascontiguousarray(
            frequencies[validation_indices], dtype=np.float64
        )
        mapped_omega = bilinear_prewarp_angular_frequency(physical_omega, float(grid.dt))
        validation_kwargs = {}
        if validation_indices:
            validation_kwargs = {
                "validation_angular_frequencies": mapped_omega[validation_indices],
                "validation_admittances": values[validation_indices],
            }
        try:
            result = synthesize_passive_admittance(
                mapped_omega[training_mask],
                values[training_mask],
                candidate_orders=candidate_orders,
                direct=1.0,
                maximum_relative_error=1e-3,
                maximum_reflection_error=5e-4,
                passivity_margin=1e-8,
                **validation_kwargs,
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            logger.warning(
                f"Eigenmode admittance match port {self.owner.port_index} shadow "
                f"rational synthesis failed and remains disabled: {exc}"
            )
            self.rational_admittance_fit_error = str(exc)
            return None
        self.rational_admittance_fit_error = None
        timestep = float(grid.dt)
        self.rational_discrete_poles = np.ascontiguousarray(
            (1.0 + 0.5 * timestep * result.model.poles)
            / (1.0 - 0.5 * timestep * result.model.poles),
            dtype=np.complex128,
        )
        runtime_status = (
            "experimental runtime use was requested and awaits the final profile gate"
            if ENABLE_RATIONAL_ADMITTANCE_RUNTIME
            else "runtime use remains disabled"
        )
        logger.info(
            f"Eigenmode admittance match port {self.owner.port_index} shadow "
            f"rational synthesis selected {result.model.order} pole(s), maximum "
            f"validation relative error {result.validation_maximum_relative_error:.3e}, "
            f"and passed the scalar polynomial passivity search; {runtime_status}."
        )
        return result

    def _prepare_modal_time_constants(self, grid):
        monitor = self.owner.port_monitor
        frequencies = np.asarray(monitor.anchor_frequencies, dtype=np.float64)
        neff = np.asarray(monitor.anchor_neff, dtype=np.complex128)
        centre = self.basis_frequency
        verification_fractional_span = (
            float(frequencies[-1] - frequencies[0]) / centre
            if frequencies.size > 1
            else 0.0
        )
        requested_fractional_span = (
            float(self.owner.dft_stop - self.owner.dft_start) / centre
        )
        fractional_span = max(
            verification_fractional_span,
            requested_fractional_span,
        )
        if fractional_span > MATCHED_BROADBAND_WARNING_FRACTION:
            logger.warning(
                f"Eigenmode admittance match port {self.owner.port_index} "
                "requested band or verification-anchor span reaches "
                f"{100 * fractional_span:.1f}% of the centre frequency. "
                "The centre-frequency E/H pair supplies a constant modal "
                "admittance; prefer a narrower band when the application permits."
            )
        spacing = float(grid.dl[self.normal_axis])
        group_velocities = []
        time_constants = []
        for mode_position, mode_index in enumerate(self.mode_indices):
            modal_neff = neff[:, mode_position]
            if np.any(
                np.abs(np.imag(modal_neff))
                > PROPAGATION_IMAGINARY_TOLERANCE
                * np.maximum(1.0, np.abs(np.real(modal_neff)))
            ):
                raise ValueError(
                    f"Eigenmode admittance match port {self.owner.port_index}, "
                    f"mode {mode_index} requires real lossless propagation constants."
                )
            beta = 2 * np.pi * frequencies * np.real(modal_neff) / config.c
            if frequencies.size >= 2:
                nearest = np.argsort(np.abs(frequencies - centre))[
                    : min(3, frequencies.size)
                ]
                omega = 2 * np.pi * frequencies[nearest]
                slope = float(np.polyfit(omega, beta[nearest], 1)[0])
                group_velocity = 1.0 / slope if slope > 0 else np.nan
            else:
                centre_neff = float(np.real(modal_neff[0]))
                group_velocity = config.c / centre_neff if centre_neff > 0 else np.nan
                logger.warning(
                    f"Eigenmode admittance match port {self.owner.port_index}, "
                    f"mode {mode_index} has only one anchor. Its half-cell "
                    "storage uses centre phase velocity because group delay "
                    "cannot be verified; use auto or multiple explicit anchors."
                )
            if not np.isfinite(group_velocity) or group_velocity <= 0:
                raise ValueError(
                    f"Eigenmode admittance match port {self.owner.port_index}, "
                    f"mode {mode_index} has invalid estimated group velocity "
                    f"{group_velocity:g} m/s."
                )
            tau = 0.5 * spacing / group_velocity
            group_velocities.append(group_velocity)
            time_constants.append(tau)
            logger.info(
                f"Eigenmode admittance match port {self.owner.port_index}, mode "
                f"{mode_index} uses centre E/H profile {centre:g} Hz, normalized "
                f"admittance 1, estimated group velocity {group_velocity:g} m/s, "
                f"and boundary half-cell time constant {tau:g} s."
            )
        return (
            np.asarray(group_velocities, dtype=np.float64),
            np.asarray(time_constants, dtype=np.float64),
        )

    def _read_boundary_magnetic_coefficients(self, grid):
        fields = (grid.Hx, grid.Hy, grid.Hz)
        u_axis, v_axis = self.owner.transverse_axes
        hplane = self.boundary_index if self.direction_sign > 0 else self.boundary_index - 1
        raw_hu = self._local_component_view(fields[u_axis], 0, "H", hplane)
        raw_hv = self._local_component_view(fields[v_axis], 1, "H", hplane)
        handedness = self.owner._modal_basis_handedness()
        measure = self._port_measure(grid)
        outward_covector = np.concatenate(
            (
                (-self.direction_sign * handedness * raw_hv).ravel(),
                (self.direction_sign * handedness * raw_hu).ravel(),
            )
        )
        outward_overlap = self.basis @ (measure * outward_covector)
        # The outward coefficient is b-a. The public recurrence uses the
        # direction-normalized magnetic coefficient Q=a-b.
        coefficients = -np.linalg.solve(self.power_gram, outward_overlap)
        if not np.all(np.isfinite(coefficients)):
            raise ValueError(
                f"Eigenmode admittance match port {self.owner.port_index} "
                "produced non-finite modal magnetic coefficients."
            )
        return coefficients

    def reset(self):
        """Reset the modal voltage and any fitted characteristic-admittance state."""

        self.modal_voltage_state.fill(0)
        if self.rational_ade is not None:
            self.rational_ade.reset()

    def update_electric_boundary(self, sample_index, grid):
        """Advance the passive modal-load boundary to one integer E time."""

        if sample_index == 0:
            self.modal_voltage_state.fill(0)
            if self.rational_ade is not None:
                self.rational_ade.reset()
            self._write_boundary_field(grid, self.modal_voltage_state @ self.basis)
            return

        old_voltage = self.modal_voltage_state
        magnetic_voltage = self._read_boundary_magnetic_coefficients(grid)
        incident = np.zeros(len(self.mode_indices), dtype=np.float64)
        if self.source_mode_position is not None:
            time = (float(sample_index) - 0.5) * grid.dt
            source_value = (
                self.owner._waveform_value(time, grid)
                if self.owner._source_is_active(time)
                else 0.0
            )
            incident[self.source_mode_position] = source_value

        if self.rational_runtime_enabled:
            new_voltage = np.asarray(
                (
                    self.rational_ade.step(
                        -float(magnetic_voltage[0]),
                        float(incident[0]),
                    ),
                ),
                dtype=np.float64,
            )
        else:
            alpha = self.modal_time_constants / float(grid.dt)
            new_voltage = constant_modal_admittance_step(
                old_voltage,
                magnetic_voltage,
                incident,
                alpha,
            )
        if not np.all(np.isfinite(new_voltage)):
            raise ValueError(
                f"Eigenmode admittance match port {self.owner.port_index} "
                "produced non-finite modal electric coefficients."
            )
        self.modal_voltage_state = np.asarray(new_voltage, dtype=np.float64)
        self._write_boundary_field(grid, self.modal_voltage_state @ self.basis)


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

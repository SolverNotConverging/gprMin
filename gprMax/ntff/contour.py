"""Frequency-domain 2-D Kirchhoff contour transform, with exp(+i omega t).

The returned scalar is sqrt(r) exp(+ikr) times Ez (TMz) or Hz (TEz),
and likewise for other invariant axes. All powers are per invariant length.
"""

from itertools import product
from types import SimpleNamespace

import numpy as np
from scipy.constants import epsilon_0, mu_0

from gprMax.mode2d import mode2d_geometry
from gprMax.ntff.frequency_domain import (
    _ComponentDFTAccumulator,
    _frequencies,
    _window,
    validate_nyquist_frequencies,
)
from gprMax.ntff.evaluator import evaluate_far_zone_patches
from gprMax.ntff.equivalent_currents import EquivalentCurrentPhasors, evaluate_love_current_phasors
from gprMax.ntff.surfaces import COMPONENT_OFFSETS


def contour_directions(angles, axis):
    """Angles run from +u toward +v, with (u, v, invariant) right-handed."""
    angles = np.deg2rad(angles)
    directions = np.zeros((angles.size, 3))
    directions[:, (axis + 1) % 3] = np.cos(angles)
    directions[:, (axis + 2) % 3] = np.sin(angles)
    return directions


def evaluate_contour(points, normals, lengths, scalar, derivative, wavenumbers, directions, origin, nthreads=1):
    """Evaluate the outgoing 2-D Green identity from collocated phasors.

    G = -i H0^(2)(kr)/4; F = exp(-i*pi/4)/sqrt(8*pi*k)
    integral exp(ik*s.r') [ik*(n.s)*psi - dpsi/dn] dl.
    Inputs scalar/derivative have shape (frequency, contour sample).
    """
    result = evaluate_far_zone_patches(
        points,
        normals,
        lengths,
        wavenumbers / (2 * np.pi),
        directions,
        scalar,
        derivative,
        wave_speed=1.0,
        origin=origin,
        nthreads=nthreads,
    )
    return result * (4 * np.pi * np.exp(-0.25j * np.pi) / np.sqrt(8 * np.pi * wavenumbers))[:, None]


class NTFFContourMonitor:
    """Streaming Yee-field DFT using the shared CPU/device NTFF collectors."""

    def __init__(
        self,
        grid,
        mode,
        lower,
        upper,
        frequencies,
        angles,
        name,
        window="rectangular",
        origin=None,
        formulation="equivalent_current",
        real_dtype=np.float64,
        nthreads=1,
        solver_backend="cpu",
    ):
        self.grid = grid
        self.geometry = mode2d_geometry(mode)
        if self.geometry is None:
            raise ValueError("NTFFContour requires a 2-D TE or TM model")
        self.name = name
        self.formulation = "ksir" if formulation == "ramahi" else formulation
        if self.formulation not in ("ksir", "equivalent_current"):
            raise ValueError("NTFFContour formulation must be 'ksir' (or 'ramahi') or 'equivalent_current'")
        self.real_dtype = np.dtype(real_dtype)
        self.complex_dtype = np.dtype(np.complex64 if self.real_dtype == np.dtype(np.float32) else np.complex128)
        self.nthreads, self.solver_backend = nthreads, solver_backend
        self.mpi_comm = getattr(grid, "comm", None)
        self.dt, self.iterations = grid.dt, grid.iterations
        self.frequencies = _frequencies(frequencies, np.float64)
        if np.any(self.frequencies <= 0):
            raise ValueError("NTFFContour far fields require strictly positive frequencies")
        validate_nyquist_frequencies(self.frequencies, self.dt)
        self.angles = np.atleast_1d(np.asarray(angles, dtype=float))
        if self.angles.ndim != 1 or not self.angles.size or not np.all(np.isfinite(self.angles)):
            raise ValueError("NTFFContour angles must be a nonempty finite one-dimensional array")
        self.window_name = window
        self.window_values = _window(window, self.iterations, np.float64)
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.axis = self.geometry.invariant_axis
        self.plane = ((self.axis + 1) % 3, (self.axis + 2) % 3)
        shape = np.asarray(getattr(grid, "global_size", (grid.nx, grid.ny, grid.nz))) + 1
        spacing = np.asarray(grid.dl)
        logical, normals, lengths = [], [], []
        for normal_axis, tangent_axis in (self.plane, self.plane[::-1]):
            if self.upper[normal_axis] <= self.lower[normal_axis]:
                raise ValueError("NTFFContour requires positive extent on both in-plane axes")
            for side, sign in ((self.lower[normal_axis], -1), (self.upper[normal_axis], 1)):
                for along in range(self.lower[tangent_axis], self.upper[tangent_axis]):
                    point = np.zeros(3)
                    point[self.axis] = self.geometry.live_index
                    point[normal_axis], point[tangent_axis] = side, along + 0.5
                    normal = np.zeros(3)
                    normal[normal_axis] = sign
                    logical.append(point)
                    normals.append(normal)
                    lengths.append(spacing[tangent_axis])
        logical = np.array(logical)
        self.points = logical * spacing
        self.normals, self.lengths = np.array(normals), np.array(lengths)
        self.origin = (self.lower + self.upper) * spacing / 2 if origin is None else np.asarray(origin, dtype=float)
        if self.origin.shape != (3,) or not np.all(np.isfinite(self.origin)):
            raise ValueError("NTFFContour origin must contain three finite coordinates")
        self.scalar_component = ("E" if self.geometry.polarisation == "TM" else "H") + "xyz"[self.axis]
        self.components = (
            (self.scalar_component,)
            if self.formulation == "ksir"
            else self.geometry.active_electric + self.geometry.active_magnetic
        )
        self.stencils = {}
        self.derivative_stencil = []
        self.phasors = {}
        for component in self.components:
            indices, weights = [], []
            # Ramahi uses same-component inside/outside samples for the
            # normal derivative; it never derives it from the other field.
            shifts = (-0.5, 0.5) if self.formulation == "ksir" else (0,)
            normal_spacing = np.sum(abs(self.normals) * spacing, axis=1)
            for shift in shifts:
                coordinates = logical + shift * self.normals - np.array(COMPONENT_OFFSETS[component])
                coordinates[:, self.axis] = self.geometry.live_index
                base = np.floor(coordinates).astype(int)
                fraction = coordinates - base
                for bits in product((0, 1), repeat=2):
                    index = base.copy()
                    weight = np.ones(len(base))
                    for axis, bit in zip(self.plane, bits):
                        index[:, axis] += bit
                        weight *= fraction[:, axis] if bit else 1 - fraction[:, axis]
                    if np.any(index < 0) or np.any(index >= shape):
                        raise ValueError("NTFFContour interpolation stencil leaves the model")
                    indices.append(tuple(index.T))
                    weights.append(weight / len(shifts))
                    if self.formulation == "ksir":
                        self.derivative_stencil.append((tuple(index.T), weight * (2 * shift) / normal_spacing))
            self.stencils[component] = list(zip(indices, weights))
            self.phasors[component] = np.zeros((len(self.frequencies), len(logical)), dtype=self.complex_dtype)
        self._prepare_collectors()
        self._result = None

    def _prepare_collectors(self):
        """Collect unique Yee nodes; interpolate only after the DFT completes.

        Identical inside/outside indices adapt a node to the existing paired
        sample kernel. MPI assigns each node exactly once, with no timestep
        communication beyond the solver's normal halo exchanges.
        """
        self.sampling_surfaces, self._accumulators, self._node_maps = {}, {}, {}
        local_shape = tuple(np.array((self.grid.nx, self.grid.ny, self.grid.nz)) + 1)
        for component, stencil in self.stencils.items():
            nodes = np.unique(np.concatenate([np.column_stack(index) for index, _ in stencil]), axis=0)
            if self.mpi_comm is None:
                selected = np.arange(len(nodes))
                local = nodes
            else:
                owners = np.array([self.grid.get_rank_from_coordinate(node) for node in nodes])
                selected = np.flatnonzero(owners == self.grid.rank)
                local = nodes[selected] - self.grid.lower_extent
            flat = np.ravel_multi_index(local.T, local_shape).astype(np.int64)
            surface = SimpleNamespace(
                component=component,
                npatches=len(flat),
                field_shape=local_shape,
                faces=(SimpleNamespace(inside_flat_indices=flat, outside_flat_indices=flat),),
            )
            self.sampling_surfaces[component] = surface
            self._accumulators[component] = _ComponentDFTAccumulator(
                surface,
                self.frequencies,
                self.dt,
                self.iterations,
                0 if component[0] == "E" else 0.5,
                self.window_values,
                self.window_name,
                "2d_closed_contour",
                self.real_dtype,
                self.complex_dtype,
                self.nthreads,
            )
            lookup = {tuple(node): i for i, node in enumerate(nodes)}
            maps = [np.array([lookup[tuple(node)] for node in np.column_stack(index)]) for index, _ in stencil]
            self._node_maps[component] = (nodes, selected, maps)

    def validate_materials(self, material_ids, id_lookup):
        """Require a homogeneous exterior and keep every stencil out of PML."""
        grid = self.grid
        from gprMax.ntff.interface import _surface_pml_limits

        limits = _surface_pml_limits(grid)
        ids = set()
        for component, stencil in self.stencils.items():
            for indices, weights in stencil:
                keep = weights > 0
                samples = np.column_stack(indices)[keep]
                for axis in self.plane:
                    if np.any(samples[:, axis] < limits[axis][0]) or np.any(samples[:, axis] > limits[axis][1]):
                        raise ValueError("NTFFContour sampling stencil enters PML")
                if self.mpi_comm is not None:
                    samples = (
                        np.array(
                            [node for node in samples if self.grid.get_rank_from_coordinate(node) == grid.rank],
                            dtype=int,
                        ).reshape(-1, 3)
                        - grid.lower_extent
                    )
                ids.update(np.unique(material_ids[(id_lookup[component],) + tuple(samples.T)]))
        if self.mpi_comm is not None:
            ids = set().union(*self.mpi_comm.allgather(ids))
        if len(ids) != 1:
            raise ValueError("NTFFContour requires one homogeneous material at all contour samples")
        self.surface_material_id = int(next(iter(ids)))
        # Inspect the exterior, including the interpolation neighbourhood, on
        # the live plane. Interior structures may have arbitrary materials.
        bad_exterior = False
        for component in self.geometry.active_electric + self.geometry.active_magnetic:
            offset = np.zeros(3, dtype=int) if self.mpi_comm is None else grid.lower_extent
            values = np.take(
                material_ids[id_lookup[component]], self.geometry.live_index - offset[self.axis], axis=self.axis
            )
            axes = [axis for axis in range(3) if axis != self.axis]
            coordinates = np.indices(values.shape)
            exterior = np.zeros(values.shape, dtype=bool)
            for local, axis in enumerate(axes):
                position = coordinates[local] + offset[axis] + COMPONENT_OFFSETS[component][axis]
                exterior |= (position <= self.lower[axis]) | (position >= self.upper[axis])
            # Last grid planes contain unused Yee slots; exclude domain edges.
            interior = np.ones(values.shape, dtype=bool)
            size = np.asarray(getattr(grid, "global_size", (grid.nx, grid.ny, grid.nz)))
            for local, axis in enumerate(axes):
                global_index = coordinates[local] + offset[axis]
                interior &= (global_index > 0) & (global_index < size[axis] - 1)
            bad_exterior |= np.any(values[exterior & interior] != self.surface_material_id)
        if self.mpi_comm is not None:
            bad_exterior = any(self.mpi_comm.allgather(bool(bad_exterior)))
        if bad_exterior:
            raise ValueError("NTFFContour requires a homogeneous exterior outside the contour")
        bad_source = False
        for collection in (
            "hertziandipoles",
            "magneticdipoles",
            "voltagesources",
            "transmissionlines",
            "magneticfrillsources",
        ):
            for source in getattr(grid, collection, ()):
                prefix = "H" if collection == "magneticdipoles" else "E"
                position = np.asarray(source.coord) + COMPONENT_OFFSETS[prefix + source.polarisation]
                if self.mpi_comm is not None:
                    position = grid.local_to_global_coordinate(position)
                if not all(self.lower[a] < position[a] < self.upper[a] for a in self.plane):
                    bad_source = True
        # A matched auxiliary guide is outside the physical domain. Its whole
        # coupling aperture, including the adjacent Yee stencil, must be
        # enclosed so that the contour sees only the radiated physical field.
        for modal_monitor in getattr(grid, "eigenmodeports", ()):
            port = modal_monitor.owner
            normal = int(port.normal_axis)
            if normal not in self.plane:
                bad_source = True
                continue
            plane = getattr(port, "global_plane_index", port.plane_index)
            bad_source |= not (self.lower[normal] + 1 < plane < self.upper[normal] - 1)
            starts = getattr(port, "global_transverse_start", port.transverse_start)
            stops = getattr(port, "global_transverse_stop", port.transverse_stop)
            for axis, start, stop in zip(port.transverse_axes, starts, stops):
                if axis in self.plane:
                    bad_source |= not (self.lower[axis] + 1 < start and stop < self.upper[axis] - 1)
        unsupported = bool(getattr(grid, "discreteplanewaves", ()))
        if self.mpi_comm is not None:
            bad_source = any(self.mpi_comm.allgather(bad_source))
            unsupported = any(self.mpi_comm.allgather(unsupported))
        if bad_source:
            raise ValueError(
                "NTFFContour must enclose every impressed source and modal aperture with one-cell clearance"
            )
        if unsupported:
            raise ValueError("NTFFContour does not support plane-wave excitation")
        return self.surface_material_id

    def configure_background(self, materials):
        material = next(item for item in materials if item.numID == self.surface_material_id)
        if (
            not np.isfinite(material.er)
            or not np.isfinite(material.mr)
            or material.er <= 0
            or material.mr <= 0
            or material.se != 0
            or material.sm != 0
            or getattr(material, "poles", 0)
        ):
            raise ValueError("NTFFContour requires a lossless, nondispersive background")
        self.epsilon, self.mu = epsilon_0 * material.er, mu_0 * material.mr
        self.impedance = np.sqrt(self.mu / self.epsilon)
        self.wavenumbers = 2 * np.pi * self.frequencies * np.sqrt(self.mu * self.epsilon)

    def _observe(self, iteration, prefix, fields):
        if self._result is not None:
            raise ValueError("cannot sample a finalised NTFFContour")
        for component in self.components:
            if component[0] == prefix:
                field = fields["xyz".index(component[1].lower())]
                self._accumulators[component].observe(iteration, field)

    def device_sampling_multiplier(self, component, iteration):
        return self._accumulators[component].sampling_multiplier(iteration)

    def load_device_component_dfts(self, component, inside, outside):
        self._accumulators[component].load_device_dfts(inside, outside)

    def observe_electric(self, iteration, Ex, Ey, Ez):
        self._observe(iteration, "E", (Ex, Ey, Ez))

    def observe_magnetic(self, iteration, Hx, Hy, Hz):
        self._observe(iteration, "H", (Hx, Hy, Hz))

    def finalise(self):
        if self._result is not None:
            return
        derivative = None
        for component, accumulator in self._accumulators.items():
            if accumulator._next_iteration != self.iterations:
                raise ValueError("NTFFContour cannot finalise an incomplete record")
            nodes, selected, maps = self._node_maps[component]
            if self.mpi_comm is None:
                raw = accumulator.inside_dft
            else:
                chunks = self.mpi_comm.gather((selected, accumulator.inside_dft), root=0)
                if self.grid.rank != 0:
                    continue
                raw = np.empty((len(self.frequencies), len(nodes)), dtype=self.complex_dtype)
                for indices, values in chunks:
                    raw[:, indices] = values
            self.phasors[component] = sum(
                raw[:, indices] * weights for indices, (_, weights) in zip(maps, self.stencils[component])
            )
            if self.formulation == "ksir":
                derivative = sum(
                    raw[:, indices] * weights for indices, (_, weights) in zip(maps, self.derivative_stencil)
                )
        if self.mpi_comm is not None and self.grid.rank != 0:
            return
        self._evaluate_phasors(derivative)

    def _evaluate_phasors(self, derivative=None):
        """Evaluate each formulation independently from its sampled fields."""
        axis_vector = np.eye(3)[self.axis]
        self.normal_derivative = derivative
        tm = self.geometry.polarisation == "TM"
        if self.formulation == "equivalent_current":
            electric = np.zeros((len(self.frequencies), len(self.points), 3), dtype=self.complex_dtype)
            magnetic = np.zeros_like(electric)
            for component, phasors in self.phasors.items():
                (electric if component[0] == "E" else magnetic)[:, :, "xyz".index(component[1])] = phasors
            currents = EquivalentCurrentPhasors(
                self.points.astype(self.real_dtype),
                self.normals.astype(self.real_dtype),
                self.lengths.astype(self.real_dtype),
                np.ascontiguousarray(np.cross(self.normals, magnetic), dtype=self.complex_dtype),
                np.ascontiguousarray(-np.cross(self.normals, electric), dtype=self.complex_dtype),
            )
        elif derivative is None:
            raise ValueError("KSIR requires independently sampled normal derivatives")

        def evaluate(angles):
            directions = contour_directions(angles, self.axis)
            if self.formulation == "ksir":
                return evaluate_contour(
                    self.points,
                    self.normals,
                    self.lengths,
                    self.phasors[self.scalar_component],
                    derivative,
                    self.wavenumbers,
                    directions,
                    self.origin,
                    self.nthreads,
                )
            field = evaluate_love_current_phasors(
                currents,
                self.frequencies,
                directions,
                origin=self.origin,
                wave_speed=1 / np.sqrt(self.mu * self.epsilon),
                impedance=self.impedance,
                nthreads=self.nthreads,
            )
            field = field * (4 * np.pi * np.exp(-0.25j * np.pi) / np.sqrt(8 * np.pi * self.wavenumbers))[:, None, None]
            return field[:, :, self.axis] if tm else np.cross(directions, field)[:, :, self.axis] / self.impedance

        far = evaluate(self.angles)
        intensity_scale = 1 / (2 * self.impedance) if tm else self.impedance / 2
        intensity = intensity_scale * abs(far) ** 2
        radius = np.max(np.linalg.norm((self.points - self.origin)[:, self.plane], axis=1))
        count = max(360, int(np.ceil(8 * np.max(self.wavenumbers) * radius)))
        circle = evaluate(np.arange(count) * 360 / count)
        power = intensity_scale * np.mean(abs(circle) ** 2, axis=1) * 2 * np.pi
        directivity = np.divide(
            2 * np.pi * intensity, power[:, None], out=np.full_like(intensity, np.nan), where=power[:, None] > 0
        )
        directions = contour_directions(self.angles, self.axis)
        angular_tangent = np.cross(axis_vector, directions)
        electric = far[:, :, None] * (axis_vector if tm else self.impedance * angular_tangent)
        magnetic = far[:, :, None] * (-angular_tangent / self.impedance if tm else axis_vector)
        self._result = dict(
            scalar=far,
            electric=electric,
            magnetic=magnetic,
            radiation_intensity=intensity,
            radiated_power_per_length=power,
            directivity=directivity,
        )

    @property
    def result(self):
        if self._result is None:
            raise RuntimeError("NTFFContour result is available after the simulation finishes")
        self._add_modal_power()
        return self._result

    def _add_modal_power(self):
        """Use the modal power Gram matrix, including cross-mode terms.

        Port phasors use the rectangular, dt-weighted DFT. Frequencies must
        match exactly; interpolation of power waves is not a normalization.
        Power remains spectral power per invariant length (W s^2 / m).
        """
        monitors = getattr(self.grid, "eigenmodeports", ())
        if not monitors or "accepted_power_per_length" in self._result:
            return
        if any(port.result is None for port in monitors):
            return
        incident, reflected, net_power, masks = [], [], [], []
        for port in monitors:
            pin = np.full(len(self.frequencies), np.nan)
            pout = pin.copy()
            pnet = pin.copy()
            valid = np.zeros(len(pin), dtype=bool)
            for i, frequency in enumerate(self.frequencies):
                # Modal frequencies follow solver precision. Compare with
                # their stored representation rather than rejecting a rounded
                # float32 frequency requested identically by both outputs.
                represented = np.asarray(frequency, dtype=port.result.frequency.dtype)
                matches = np.flatnonzero(port.result.frequency == represented)
                if self.window_name != "rectangular" or len(matches) != 1:
                    continue
                j = matches[0]
                if not (np.all(port.result.power_wave_valid[:, j]) and port.power_matrix_valid[j]):
                    continue
                a, b = port.result.incident[:, j], port.result.outgoing[:, j]
                w = port.power_matrix[j]
                pin[i] = np.real(np.vdot(a, w @ a))
                pout[i] = np.real(np.vdot(b, w @ b))
                # E is proportional to a+b and H to a-b at the port plane.
                # A lossy modal impedance can add a forward/backward cross
                # term to net flux, even when each travelling-wave power is
                # represented by the Hermitian power matrix.
                g = port.electric_gram[j]
                pnet[i] = np.real(np.vdot(a - b, g @ (a + b)))
                valid[i] = (
                    np.isfinite(pin[i])
                    and np.isfinite(pout[i])
                    and np.isfinite(pnet[i])
                    and pin[i] >= 0
                    and pout[i] >= 0
                )
            incident.append(pin)
            reflected.append(pout)
            net_power.append(pnet)
            masks.append(valid)
        incident, reflected = np.asarray(incident), np.asarray(reflected)
        net_power = np.asarray(net_power)
        accepted = np.sum(net_power, axis=0)
        valid = np.all(masks, axis=0)
        if any(
            getattr(self.grid, name, ())
            for name in (
                "hertziandipoles",
                "magneticdipoles",
                "voltagesources",
                "transmissionlines",
                "magneticfrillsources",
            )
        ):
            # A second impressed drive contributes radiation that cannot be
            # normalized by the eigenmode ports' accepted power alone.
            valid[:] = False
        scale = max(float(np.nanmax(np.where(valid, np.sum(incident, axis=0), 0))), np.finfo(float).tiny)
        gain_valid = valid & (accepted > 100 * np.finfo(self.real_dtype).eps * scale)
        self._result.update(
            port_ids=np.asarray([port.owner.port_index for port in monitors], dtype=int),
            incident_power_per_port_per_length=incident,
            reflected_power_per_port_per_length=reflected,
            accepted_power_per_port_per_length=net_power,
            accepted_power_per_length=accepted,
            port_power_valid=valid,
            gain_valid=gain_valid,
            gain=np.divide(
                2 * np.pi * self._result["radiation_intensity"],
                accepted[:, None],
                out=np.full_like(self._result["radiation_intensity"], np.nan),
                where=gain_valid[:, None],
            ),
            radiation_efficiency=np.divide(
                self._result["radiated_power_per_length"],
                accepted,
                out=np.full_like(accepted, np.nan),
                where=gain_valid,
            ),
        )

    def write_hdf5(self, base_group):
        group = base_group.require_group("ntff_contours").create_group(self.name)
        group.attrs["formulation"] = self.formulation
        group.attrs["solver_backend"] = self.solver_backend
        group.attrs["precision"] = self.real_dtype.name
        group.attrs["mpi_ranks"] = 1 if self.mpi_comm is None else self.mpi_comm.size
        group.attrs["mode"] = self.geometry.mode
        group.attrs["phasor_time_dependence"] = "exp(+i omega t)"
        group.attrs["field_normalization"] = "sqrt(r) exp(+i k r) field(r)"
        group.attrs["angle_convention"] = "degrees from +u towards +v; (u,v,invariant) right-handed"
        group.attrs["in_plane_axes"] = "".join("xyz"[axis] for axis in self.plane)
        group.attrs["window"] = self.window_name
        group.attrs["dt"] = self.dt
        group.attrs["iterations"] = self.iterations
        group.attrs["epsilon"] = self.epsilon
        group.attrs["mu"] = self.mu
        for name, values in dict(
            frequencies=self.frequencies,
            angles=self.angles,
            origin=self.origin,
            points=self.points,
            normals=self.normals,
            lengths=self.lengths,
        ).items():
            group.create_dataset(name, data=values)
        for name, values in self.result.items():
            group.create_dataset(name, data=values)
        surface = group.create_group("surface_dft")
        for name, values in self.phasors.items():
            surface.create_dataset(name, data=values)
        if self.normal_derivative is not None:
            surface.create_dataset("normal_derivative", data=self.normal_derivative)
        group.attrs["spectral_normalization"] = "dt sum(field exp(-i omega t)); no source normalization"

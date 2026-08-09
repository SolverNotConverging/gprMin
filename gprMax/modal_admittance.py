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

"""Scalar pole-residue models and deterministic vector fitting.

This module is deliberately independent of the FDTD grid. It fits the proper
characteristic modal admittance only; the known Yee terminal half-cell term is
kept outside the fit and is realized by :mod:`gprMax.modal_admittance_ade`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


__all__ = [
    "PassiveRationalFitResult",
    "PoleResidueAdmittance",
    "RationalFitDiagnostics",
    "RationalFitResult",
    "fit_fixed_poles",
    "reflection_equivalent_error",
    "seed_stable_poles",
    "synthesize_passive_admittance",
    "vector_fit_scalar",
    "yee_staggered_characteristic_admittance",
]


_STRUCTURE_TOLERANCE = 2e-8


@dataclass(frozen=True)
class PoleResidueAdmittance:
    """A scalar proper admittance ``D + sum(r_k / (s - p_k))``."""

    poles: np.ndarray
    residues: np.ndarray
    direct: float

    def __post_init__(self):
        poles_values = np.ascontiguousarray(
            self.poles, dtype=np.complex128
        ).reshape(-1)
        residue_values = np.ascontiguousarray(
            self.residues, dtype=np.complex128
        ).reshape(-1)
        # Back coefficient arrays with immutable bytes. ``write=False`` on an
        # owning ndarray can be reversed by a caller; a bytes-backed view
        # cannot be made writable and keeps a certified model immutable.
        poles = np.frombuffer(poles_values.tobytes(), dtype=np.complex128)
        residues = np.frombuffer(residue_values.tobytes(), dtype=np.complex128)
        direct = complex(self.direct)
        if poles.shape != residues.shape:
            raise ValueError("poles and residues must have identical shapes")
        if not np.all(np.isfinite(poles)) or not np.all(np.isfinite(residues)):
            raise ValueError("poles and residues must contain only finite values")
        direct_scale = max(1.0, abs(direct))
        if not np.isfinite(direct) or abs(direct.imag) > (
            _STRUCTURE_TOLERANCE * direct_scale
        ):
            raise ValueError("direct admittance must be finite and effectively real")
        object.__setattr__(self, "poles", poles)
        object.__setattr__(self, "residues", residues)
        object.__setattr__(self, "direct", float(direct.real))

    @property
    def order(self) -> int:
        return int(self.poles.size)

    def evaluate(self, s):
        """Evaluate the scalar admittance at one or more complex points."""

        values = np.asarray(s, dtype=np.complex128)
        if not np.all(np.isfinite(values)):
            raise ValueError("evaluation points must contain only finite values")
        flattened = values.reshape(-1)
        response = np.full(flattened.shape, self.direct, dtype=np.complex128)
        if self.order:
            response += np.sum(
                self.residues[None, :]
                / (flattened[:, None] - self.poles[None, :]),
                axis=1,
            )
        response = response.reshape(values.shape)
        if np.ndim(s) == 0:
            return complex(response)
        return response


@dataclass(frozen=True)
class RationalFitDiagnostics:
    iterations: int
    converged: bool
    pole_movement: float
    rms_relative_error: float
    maximum_relative_error: float
    maximum_reflection_error: float
    frequency_scale: float
    admittance_scale: float


@dataclass(frozen=True)
class RationalFitResult:
    model: PoleResidueAdmittance
    diagnostics: RationalFitDiagnostics


@dataclass(frozen=True)
class PassiveRationalFitResult:
    """Selected fitted model together with passivity and validation results."""

    model: PoleResidueAdmittance
    fit_diagnostics: RationalFitDiagnostics
    raw_passivity_certificate: object
    final_passivity_certificate: object
    repair_iterations: int
    relative_parameter_change: float
    validation_maximum_relative_error: float
    validation_maximum_reflection_error: float
    attempted_orders: tuple[int, ...]


@dataclass(frozen=True)
class _PoleGroup:
    kind: str
    first: int
    second: int | None = None


def _as_fit_data(angular_frequencies, admittances, weights=None):
    omega = np.asarray(angular_frequencies, dtype=np.float64)
    values = np.asarray(admittances, dtype=np.complex128)
    if omega.ndim != 1 or values.ndim != 1 or omega.shape != values.shape:
        raise ValueError(
            "angular frequencies and admittances must be equal-length one-dimensional arrays"
        )
    if omega.size == 0:
        raise ValueError("at least one admittance sample is required")
    if not np.all(np.isfinite(omega)) or not np.all(np.isfinite(values)):
        raise ValueError("admittance fit samples must be finite")
    if np.any(omega < 0) or np.any(np.diff(omega) <= 0):
        raise ValueError("angular frequencies must be strictly increasing and nonnegative")
    if weights is None:
        fit_weights = np.ones(omega.size, dtype=np.float64)
    else:
        fit_weights = np.asarray(weights, dtype=np.float64)
        if fit_weights.shape != omega.shape:
            raise ValueError("fit weights must have one value per sample")
        if not np.all(np.isfinite(fit_weights)) or np.any(fit_weights <= 0):
            raise ValueError("fit weights must be finite and positive")
    return omega, values, fit_weights


def _relative_close(first, second, tolerance=_STRUCTURE_TOLERANCE):
    return abs(first - second) <= tolerance * max(1.0, abs(first), abs(second))


def _pole_groups(poles):
    poles = np.asarray(poles, dtype=np.complex128).reshape(-1)
    unused = set(range(poles.size))
    groups = []
    while unused:
        first = min(unused)
        pole = poles[first]
        if abs(pole.imag) <= _STRUCTURE_TOLERANCE * max(1.0, abs(pole)):
            groups.append(_PoleGroup("real", first))
            unused.remove(first)
            continue
        candidates = [
            index
            for index in unused
            if index != first and _relative_close(poles[index], pole.conjugate())
        ]
        if not candidates:
            raise ValueError("complex poles must occur in conjugate pairs")
        second = min(candidates, key=lambda index: abs(poles[index] - pole.conjugate()))
        if poles[first].imag < 0:
            first, second = second, first
        groups.append(_PoleGroup("pair", first, second))
        unused.remove(first)
        unused.remove(second)
    return tuple(groups)


def _canonicalize_conjugate_poles(poles):
    """Return exact real/conjugate poles for an accepted approximate pairing."""

    values = np.asarray(poles, dtype=np.complex128).reshape(-1)
    groups = _pole_groups(values)
    canonical = np.array(values, dtype=np.complex128, copy=True)
    for group in groups:
        if group.kind == "real":
            canonical[group.first] = complex(values[group.first].real, 0.0)
            continue
        first = values[group.first]
        second = values[group.second]
        real_part = 0.5 * (first.real + second.real)
        imaginary_part = 0.5 * (abs(first.imag) + abs(second.imag))
        canonical[group.first] = complex(real_part, imaginary_part)
        canonical[group.second] = complex(real_part, -imaginary_part)
    return canonical


def _response_basis(s, poles, groups):
    columns = []
    for group in groups:
        first = 1.0 / (s - poles[group.first])
        if group.kind == "real":
            columns.append(first)
            continue
        second = 1.0 / (s - poles[group.second])
        columns.append(first + second)
        columns.append(1j * first - 1j * second)
    if not columns:
        return np.empty((s.size, 0), dtype=np.complex128)
    return np.column_stack(columns)


def _parameters_to_residues(parameters, poles, groups):
    parameters = np.asarray(parameters, dtype=np.float64)
    residues = np.zeros(len(poles), dtype=np.complex128)
    cursor = 0
    for group in groups:
        if group.kind == "real":
            residues[group.first] = parameters[cursor]
            cursor += 1
            continue
        residue = parameters[cursor] + 1j * parameters[cursor + 1]
        residues[group.first] = residue
        residues[group.second] = residue.conjugate()
        cursor += 2
    if cursor != parameters.size:
        raise ValueError("residue parameter count does not match the pole structure")
    return residues


def _solve_real_least_squares(matrix, right_hand_side, weights):
    weighted = np.sqrt(weights)
    complex_matrix = matrix * weighted[:, None]
    complex_rhs = right_hand_side * weighted
    real_matrix = np.vstack((complex_matrix.real, complex_matrix.imag))
    real_rhs = np.concatenate((complex_rhs.real, complex_rhs.imag))
    column_norms = np.linalg.norm(real_matrix, axis=0)
    largest_column = float(np.max(column_norms, initial=0.0))
    if largest_column <= 0 or np.any(
        column_norms <= 64 * np.finfo(float).eps * largest_column
    ):
        raise ValueError("rational-admittance least-squares system is rank deficient")
    scaled_matrix = real_matrix / column_norms[None, :]
    scaled_parameters, _, rank, singular_values = np.linalg.lstsq(
        scaled_matrix, real_rhs, rcond=None
    )
    if rank < scaled_matrix.shape[1]:
        raise ValueError("rational-admittance least-squares system is rank deficient")
    if (
        singular_values.size
        and singular_values[-1]
        <= 1e-12 * singular_values[0]
    ):
        raise ValueError(
            "rational-admittance least-squares system is ill-conditioned"
        )
    parameters = scaled_parameters / column_norms
    if not np.all(np.isfinite(parameters)) or not np.all(np.isfinite(singular_values)):
        raise ValueError("rational-admittance least-squares solve failed")
    return parameters


def seed_stable_poles(
    order: int,
    minimum_angular_frequency: float,
    maximum_angular_frequency: float,
    *,
    damping_fraction: float = 0.05,
) -> np.ndarray:
    """Create deterministic left-half-plane real/conjugate starting poles."""

    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)):
        raise ValueError("order must be a nonnegative integer")
    if order < 0:
        raise ValueError("order must be a nonnegative integer")
    low = float(minimum_angular_frequency)
    high = float(maximum_angular_frequency)
    damping = float(damping_fraction)
    if not np.isfinite(low) or not np.isfinite(high) or low < 0 or high <= low:
        raise ValueError("pole-seed frequency range must satisfy 0 <= low < high")
    if not np.isfinite(damping) or damping <= 0:
        raise ValueError("damping_fraction must be finite and positive")
    if order == 0:
        return np.empty(0, dtype=np.complex128)

    pair_count = order // 2
    real_count = order % 2
    effective_low = max(low, high * 1e-3, np.finfo(float).tiny)
    frequencies = (
        np.geomspace(effective_low, high, pair_count)
        if pair_count
        else np.empty(0)
    )
    poles = []
    if real_count:
        poles.append(-np.sqrt(effective_low * high))
    for frequency in frequencies:
        poles.extend(
            (
                -damping * frequency + 1j * frequency,
                -damping * frequency - 1j * frequency,
            )
        )
    return np.asarray(poles, dtype=np.complex128)


def _canonical_stable_poles(roots, minimum_real_part):
    roots = np.asarray(roots, dtype=np.complex128).reshape(-1)
    if not np.all(np.isfinite(roots)):
        raise ValueError("vector fitting produced non-finite relocated poles")
    if roots.size == 0:
        return roots
    polynomial = np.poly(roots)
    coefficient_scale = max(float(np.max(np.abs(polynomial), initial=0.0)), 1.0)
    if np.max(np.abs(polynomial.imag), initial=0.0) > 1e-6 * coefficient_scale:
        raise ValueError("vector fitting failed to preserve a real pole polynomial")
    roots = np.roots(np.asarray(polynomial.real, dtype=np.float64))
    roots = np.asarray(
        [
            complex(-max(abs(root.real), minimum_real_part), root.imag)
            for root in roots
        ],
        dtype=np.complex128,
    )

    real_roots = []
    positive = []
    negative = list(np.flatnonzero(np.imag(roots) < 0))
    used_negative = set()
    for root in roots:
        if abs(root.imag) <= 1e-8 * max(1.0, abs(root)):
            real_roots.append(complex(root.real, 0.0))
        elif root.imag > 0:
            candidates = [index for index in negative if index not in used_negative]
            if not candidates:
                raise ValueError("relocated poles could not be paired by conjugacy")
            partner_index = min(
                candidates, key=lambda index: abs(roots[index] - root.conjugate())
            )
            used_negative.add(partner_index)
            partner = roots[partner_index]
            real_part = 0.5 * (root.real + partner.real)
            imag_part = 0.5 * (abs(root.imag) + abs(partner.imag))
            positive.append(complex(real_part, imag_part))
    if len(real_roots) + 2 * len(positive) != roots.size:
        raise ValueError("relocated pole count changed during conjugate pairing")
    real_roots.sort(key=lambda value: value.real)
    positive.sort(key=lambda value: (value.imag, value.real))
    result = list(real_roots)
    for root in positive:
        result.extend((root, root.conjugate()))
    return np.asarray(result, dtype=np.complex128)


def fit_fixed_poles(
    angular_frequencies,
    admittances,
    poles,
    *,
    weights=None,
    direct: float | None = None,
) -> PoleResidueAdmittance:
    """Fit real/conjugate residues and an optional real direct term."""

    omega, values, fit_weights = _as_fit_data(
        angular_frequencies, admittances, weights
    )
    fitted_poles = np.asarray(poles, dtype=np.complex128).reshape(-1)
    if not np.all(np.isfinite(fitted_poles)):
        raise ValueError("fixed poles must be finite")
    fitted_poles = _canonicalize_conjugate_poles(fitted_poles)
    groups = _pole_groups(fitted_poles)
    basis = _response_basis(1j * omega, fitted_poles, groups)
    if direct is None:
        matrix = np.column_stack((basis, np.ones(omega.size, dtype=np.complex128)))
        parameters = _solve_real_least_squares(matrix, values, fit_weights)
        residue_parameters = parameters[:-1]
        fitted_direct = float(parameters[-1])
    else:
        fitted_direct = float(direct)
        if not np.isfinite(fitted_direct):
            raise ValueError("fixed direct admittance must be finite")
        if basis.shape[1] == 0:
            return PoleResidueAdmittance(fitted_poles, np.empty(0), fitted_direct)
        residue_parameters = _solve_real_least_squares(
            basis, values - fitted_direct, fit_weights
        )
    residues = _parameters_to_residues(residue_parameters, fitted_poles, groups)
    return PoleResidueAdmittance(fitted_poles, residues, fitted_direct)


def reflection_equivalent_error(reference, approximation):
    """Return ``|(Yfit-Y)/(Yfit+Y)|`` with an explicit singular guard."""

    reference = np.asarray(reference, dtype=np.complex128)
    approximation = np.asarray(approximation, dtype=np.complex128)
    reference, approximation = np.broadcast_arrays(reference, approximation)
    denominator = approximation + reference
    scale = np.maximum(1.0, np.maximum(np.abs(reference), np.abs(approximation)))
    result = np.full(reference.shape, np.inf, dtype=np.float64)
    usable = np.abs(denominator) > 64 * np.finfo(float).eps * scale
    result[usable] = np.abs(
        (approximation[usable] - reference[usable]) / denominator[usable]
    )
    if result.ndim == 0:
        return float(result)
    return result


def yee_staggered_characteristic_admittance(
    angular_frequencies,
    propagation_constants,
    colocated_admittances,
    *,
    normal_spacing: float,
    dt: float,
    half_cell_storage: float,
):
    """Remove the analytic half-cell term from the exact Yee terminal target.

    For an outward mode, the adjacent raw-H plane is one normal half-cell from
    the boundary E plane. Expressed against midpoint E voltage, its terminal
    admittance is ``Y * exp(j*beta*dw/2) / cos(omega*dt/2)``. The returned
    samples subtract the analytic Tustin half-cell term ``j*Omega*Eh`` and are
    therefore the characteristic samples to be represented by the rational
    model in the runtime recurrence.
    """

    omega = np.asarray(angular_frequencies, dtype=np.float64)
    beta = np.asarray(propagation_constants, dtype=np.float64)
    admittance = np.asarray(colocated_admittances, dtype=np.complex128)
    try:
        omega, beta, admittance = np.broadcast_arrays(omega, beta, admittance)
    except ValueError as exc:
        raise ValueError(
            "Yee characteristic-admittance inputs must be broadcast-compatible"
        ) from exc
    spacing = float(normal_spacing)
    timestep = float(dt)
    storage = float(half_cell_storage)
    if not all(np.all(np.isfinite(values)) for values in (omega, beta, admittance)):
        raise ValueError("Yee characteristic-admittance inputs must be finite")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("normal_spacing must be finite and positive")
    if not np.isfinite(timestep) or timestep <= 0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(storage) or storage <= 0:
        raise ValueError("half_cell_storage must be finite and positive")
    theta = 0.5 * omega * timestep
    if np.any(np.abs(theta) >= 0.5 * np.pi):
        raise ValueError("angular frequencies must lie strictly below Nyquist")
    centred_terminal = (
        admittance
        * np.exp(0.5j * beta * spacing)
        / np.cos(theta)
    )
    mapped = (2.0 / timestep) * np.tan(theta)
    result = centred_terminal - 1j * mapped * storage
    if result.ndim == 0:
        return complex(result)
    return np.ascontiguousarray(result)


def _fit_diagnostics(
    model,
    omega,
    values,
    *,
    iterations,
    converged,
    pole_movement,
    frequency_scale,
    admittance_scale,
):
    fitted = model.evaluate(1j * omega)
    value_floor = max(float(np.max(np.abs(values), initial=0.0)) * 1e-12, 1e-15)
    relative = np.abs(fitted - values) / np.maximum(np.abs(values), value_floor)
    return RationalFitDiagnostics(
        iterations=int(iterations),
        converged=bool(converged),
        pole_movement=float(pole_movement),
        rms_relative_error=float(np.sqrt(np.mean(relative**2))),
        maximum_relative_error=float(np.max(relative)),
        maximum_reflection_error=float(
            np.max(reflection_equivalent_error(values, fitted))
        ),
        frequency_scale=float(frequency_scale),
        admittance_scale=float(admittance_scale),
    )


def vector_fit_scalar(
    angular_frequencies,
    admittances,
    order: int,
    *,
    weights=None,
    direct: float | None = None,
    initial_poles=None,
    maximum_iterations: int = 50,
    pole_tolerance: float = 1e-7,
    damping_fraction: float = 0.05,
) -> RationalFitResult:
    """Fit a deterministic stable real-rational scalar admittance.

    This implements the classical scalar vector-fitting relocation equation.
    Positive-real enforcement is intentionally a separate mandatory stage in
    :mod:`gprMax.modal_admittance_passivity`.
    """

    omega, values, fit_weights = _as_fit_data(
        angular_frequencies, admittances, weights
    )
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)):
        raise ValueError("order must be a nonnegative integer")
    order = int(order)
    if order < 0:
        raise ValueError("order must be a nonnegative integer")
    if isinstance(maximum_iterations, (bool, np.bool_)) or not isinstance(
        maximum_iterations, (int, np.integer)
    ) or maximum_iterations < 1:
        raise ValueError("maximum_iterations must be an integer greater than zero")
    if not np.isfinite(pole_tolerance) or pole_tolerance <= 0:
        raise ValueError("pole_tolerance must be finite and positive")
    if omega.size < max(2, 2 * order + 4) and order:
        raise ValueError(
            f"order {order} requires at least {2 * order + 4} admittance samples"
        )

    positive = omega[omega > 0]
    if positive.size:
        frequency_scale = float(
            np.exp(0.5 * (np.log(np.min(positive)) + np.log(np.max(positive))))
        )
    else:
        frequency_scale = 1.0
    admittance_scale = max(float(np.median(np.abs(values))), 1e-15)
    scaled_omega = omega / frequency_scale
    scaled_values = values / admittance_scale
    scaled_direct = None if direct is None else float(direct) / admittance_scale

    if order == 0:
        model = fit_fixed_poles(
            scaled_omega,
            scaled_values,
            np.empty(0),
            weights=fit_weights,
            direct=scaled_direct,
        )
        physical = PoleResidueAdmittance(
            np.empty(0), np.empty(0), model.direct * admittance_scale
        )
        return RationalFitResult(
            physical,
            _fit_diagnostics(
                physical,
                omega,
                values,
                iterations=0,
                converged=True,
                pole_movement=0.0,
                frequency_scale=frequency_scale,
                admittance_scale=admittance_scale,
            ),
        )

    if initial_poles is None:
        scaled_poles = seed_stable_poles(
            order,
            float(np.min(scaled_omega)),
            float(np.max(scaled_omega)),
            damping_fraction=damping_fraction,
        )
    else:
        physical_initial = np.asarray(initial_poles, dtype=np.complex128).reshape(-1)
        if physical_initial.size != order:
            raise ValueError("initial_poles must contain exactly order values")
        scaled_poles = _canonicalize_conjugate_poles(
            physical_initial / frequency_scale
        )
    minimum_real_part = 1e-8
    s = 1j * scaled_omega
    converged = False
    movement = np.inf
    iterations = 0
    for iterations in range(1, int(maximum_iterations) + 1):
        groups = _pole_groups(scaled_poles)
        basis = _response_basis(s, scaled_poles, groups)
        sigma_columns = -scaled_values[:, None] * basis
        if scaled_direct is None:
            matrix = np.column_stack(
                (basis, np.ones(omega.size, dtype=np.complex128), sigma_columns)
            )
            right_hand_side = scaled_values
            sigma_parameters = _solve_real_least_squares(
                matrix, right_hand_side, fit_weights
            )[-basis.shape[1] :]
        else:
            matrix = np.column_stack((basis, sigma_columns))
            right_hand_side = scaled_values - scaled_direct
            sigma_parameters = _solve_real_least_squares(
                matrix, right_hand_side, fit_weights
            )[-basis.shape[1] :]
        sigma_residues = _parameters_to_residues(
            sigma_parameters, scaled_poles, groups
        )
        relocated = np.linalg.eigvals(
            np.diag(scaled_poles)
            - np.outer(np.ones(order, dtype=np.complex128), sigma_residues)
        )
        new_poles = _canonical_stable_poles(relocated, minimum_real_part)
        old_sorted = np.sort_complex(scaled_poles)
        new_sorted = np.sort_complex(new_poles)
        movement = float(
            np.max(np.abs(new_sorted - old_sorted))
            / max(1.0, float(np.max(np.abs(old_sorted), initial=0.0)))
        )
        scaled_poles = new_poles
        if movement <= pole_tolerance:
            converged = True
            break

    scaled_model = fit_fixed_poles(
        scaled_omega,
        scaled_values,
        scaled_poles,
        weights=fit_weights,
        direct=scaled_direct,
    )
    model = PoleResidueAdmittance(
        poles=scaled_model.poles * frequency_scale,
        residues=scaled_model.residues * frequency_scale * admittance_scale,
        direct=scaled_model.direct * admittance_scale,
    )
    diagnostics = _fit_diagnostics(
        model,
        omega,
        values,
        iterations=iterations,
        converged=converged,
        pole_movement=movement,
        frequency_scale=frequency_scale,
        admittance_scale=admittance_scale,
    )
    return RationalFitResult(model, diagnostics)


def synthesize_passive_admittance(
    angular_frequencies,
    admittances,
    *,
    candidate_orders=(0, 2, 4, 6, 8),
    weights=None,
    direct: float | None = 1.0,
    validation_angular_frequencies=None,
    validation_admittances=None,
    maximum_relative_error: float = 5e-3,
    maximum_reflection_error: float = 2.5e-3,
    passivity_margin: float = 1e-8,
    maximum_passivity_parameter_change: float = 0.1,
) -> PassiveRationalFitResult:
    """Select, repair, and globally certify a scalar rational admittance.

    Candidate order and anchor count are deliberately independent. The
    smallest sufficiently overdetermined order that passes validation and the
    global scalar positive-real certificate is returned. Passivity repair is
    mandatory and cannot be bypassed by this API.
    """

    # Lazy import avoids a module cycle: the passivity layer consumes the
    # pole-residue model defined above.
    from gprMax.modal_admittance_passivity import certify_scalar_positive_real
    from gprMax.modal_admittance_passivity import repair_scalar_positive_real

    omega, values, fit_weights = _as_fit_data(
        angular_frequencies, admittances, weights
    )
    requested_orders = tuple(candidate_orders)
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in requested_orders
    ):
        raise ValueError("candidate_orders must contain nonnegative integers")
    orders = tuple(int(value) for value in requested_orders)
    if not orders or any(value < 0 for value in orders):
        raise ValueError("candidate_orders must contain nonnegative integers")
    if tuple(sorted(set(orders))) != orders:
        raise ValueError("candidate_orders must be strictly increasing and unique")
    for name, value in (
        ("maximum_relative_error", maximum_relative_error),
        ("maximum_reflection_error", maximum_reflection_error),
        ("passivity_margin", passivity_margin),
        ("maximum_passivity_parameter_change", maximum_passivity_parameter_change),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")

    if validation_angular_frequencies is None and validation_admittances is None:
        validation_omega = omega
        validation_values = values
    elif validation_angular_frequencies is None or validation_admittances is None:
        raise ValueError(
            "validation frequencies and admittances must either both be supplied or both omitted"
        )
    else:
        validation_omega, validation_values, _ = _as_fit_data(
            validation_angular_frequencies,
            validation_admittances,
        )

    attempted = []
    failure_details = []
    for order in orders:
        minimum_samples = 1 if order == 0 else 2 * order + 4
        if omega.size < minimum_samples:
            failure_details.append(
                f"order {order}: needs at least {minimum_samples} fitting samples"
            )
            continue
        attempted.append(order)
        try:
            fit = vector_fit_scalar(
                omega,
                values,
                order,
                weights=fit_weights,
                direct=direct,
            )
            if not fit.diagnostics.converged:
                failure_details.append(
                    f"order {order}: vector fitting did not converge; final "
                    f"relative pole movement was {fit.diagnostics.pole_movement:.3e}"
                )
                continue
            raw_certificate = certify_scalar_positive_real(
                fit.model,
                margin=passivity_margin,
            )
            if raw_certificate.is_passive:
                final_model = fit.model
                final_certificate = raw_certificate
                repair_iterations = 0
                parameter_change = 0.0
            else:
                repair = repair_scalar_positive_real(
                    fit.model,
                    margin=passivity_margin,
                    fit_angular_frequencies=omega,
                    fit_weights=fit_weights,
                )
                final_model = repair.model
                final_certificate = repair.certificate
                repair_iterations = repair.iterations
                parameter_change = repair.relative_parameter_change
            if parameter_change > maximum_passivity_parameter_change:
                failure_details.append(
                    f"order {order}: passivity repair changed parameters by "
                    f"{parameter_change:.3e}"
                )
                continue

            validation_fit = final_model.evaluate(1j * validation_omega)
            value_floor = max(
                float(np.max(np.abs(validation_values), initial=0.0)) * 1e-12,
                1e-15,
            )
            relative = np.abs(validation_fit - validation_values) / np.maximum(
                np.abs(validation_values), value_floor
            )
            validation_relative = float(np.max(relative))
            validation_reflection = float(
                np.max(
                    reflection_equivalent_error(
                        validation_values,
                        validation_fit,
                    )
                )
            )
            if validation_relative > maximum_relative_error:
                failure_details.append(
                    f"order {order}: validation relative error "
                    f"{validation_relative:.3e}"
                )
                continue
            if validation_reflection > maximum_reflection_error:
                failure_details.append(
                    f"order {order}: validation reflection-equivalent error "
                    f"{validation_reflection:.3e}"
                )
                continue
            return PassiveRationalFitResult(
                model=final_model,
                fit_diagnostics=fit.diagnostics,
                raw_passivity_certificate=raw_certificate,
                final_passivity_certificate=final_certificate,
                repair_iterations=repair_iterations,
                relative_parameter_change=parameter_change,
                validation_maximum_relative_error=validation_relative,
                validation_maximum_reflection_error=validation_reflection,
                attempted_orders=tuple(attempted),
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            failure_details.append(f"order {order}: {exc}")

    details = "; ".join(failure_details) if failure_details else "no order was attempted"
    raise ValueError(f"no passive rational admittance model met the fit criteria ({details})")

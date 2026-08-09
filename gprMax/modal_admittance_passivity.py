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

"""Scalar all-frequency positive-real search and fixed-pole passivity repair.

The checker does not rely on a prescribed frequency grid. It constructs a
floating-point polynomial representation of ``Re(Y(j*w))`` in ``x = w**2``
and examines its computed stationary points and sign intervals relative to the
requested passivity margin. This catches violations much narrower than a
practical dense scan. It is intentionally paired with adversarial regressions,
but an independent Hamiltonian/KYP check is still required before fitted
models are enabled in production.

Passivity repair keeps the poles fixed and changes only the direct term and
the real degrees of freedom of the conjugate-symmetric residues. A cutting-
plane sequence of convex quadratic programs adds constraints at every global
violation returned by the polynomial certificate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from gprMax.modal_admittance import PoleResidueAdmittance


__all__ = [
    "ScalarPassivityCertificate",
    "ScalarPassivityRepair",
    "certify_scalar_positive_real",
    "enforce_scalar_positive_real",
    "repair_scalar_positive_real",
]


_REAL_STRUCTURE_TOLERANCE = 2e-8
_POLYNOMIAL_TOLERANCE = 2e-11
_ROOT_TOLERANCE = 2e-7


@dataclass(frozen=True)
class ScalarPassivityCertificate:
    """Result of a global scalar positive-realness check.

    Frequencies are angular frequencies in rad/s. ``critical_frequencies``
    contains the finite frequencies used to establish the global minimum;
    they arise from polynomial roots, not a prescribed sampling grid.
    """

    is_passive: bool
    is_stable: bool
    is_real_rational: bool
    direct_is_passive: bool
    minimum_real_admittance: float
    minimum_angular_frequency: float
    requested_margin: float
    critical_frequencies: tuple[float, ...]
    violating_frequencies: tuple[float, ...]
    message: str


@dataclass(frozen=True)
class ScalarPassivityRepair:
    """A repaired model together with its global certificate."""

    model: PoleResidueAdmittance
    certificate: ScalarPassivityCertificate
    iterations: int
    relative_parameter_change: float


@dataclass(frozen=True)
class _ResidueGroup:
    kind: str
    first: int
    second: int | None = None


def _as_model_arrays(model):
    poles = np.atleast_1d(np.asarray(model.poles, dtype=np.complex128))
    residues = np.atleast_1d(np.asarray(model.residues, dtype=np.complex128))
    direct = np.asarray(model.direct, dtype=np.complex128)
    if poles.ndim != 1 or residues.ndim != 1 or poles.shape != residues.shape:
        raise ValueError("poles and residues must be one-dimensional arrays of equal length")
    if direct.ndim != 0:
        raise ValueError("direct must be a scalar")
    if not (
        np.all(np.isfinite(poles))
        and np.all(np.isfinite(residues))
        and np.isfinite(direct)
    ):
        raise ValueError("pole-residue admittance coefficients must be finite")
    return poles, residues, complex(direct)


def _relative_close(first, second, tolerance=_REAL_STRUCTURE_TOLERANCE):
    return abs(first - second) <= tolerance * max(1.0, abs(first), abs(second))


def _conjugate_residue_groups(poles, residues):
    """Return independent real residue degrees of freedom, or ``None``."""

    unused = set(range(poles.size))
    groups = []
    while unused:
        first = min(unused)
        pole = poles[first]
        residue = residues[first]
        if abs(pole.imag) <= _REAL_STRUCTURE_TOLERANCE * max(1.0, abs(pole)):
            if abs(residue.imag) > _REAL_STRUCTURE_TOLERANCE * max(1.0, abs(residue)):
                return None
            groups.append(_ResidueGroup("real", first))
            unused.remove(first)
            continue

        candidates = [
            index
            for index in unused
            if index != first
            and _relative_close(poles[index], pole.conjugate())
            and _relative_close(residues[index], residue.conjugate())
        ]
        if not candidates:
            return None
        second = min(
            candidates,
            key=lambda index: abs(poles[index] - pole.conjugate())
            + abs(residues[index] - residue.conjugate()),
        )
        # Store the positive-imaginary member first for deterministic packing.
        if poles[first].imag < 0:
            first, second = second, first
        groups.append(_ResidueGroup("pair", first, second))
        unused.remove(first)
        unused.remove(second)
    return tuple(groups)


def _frequency_scale(poles):
    magnitudes = np.abs(poles)
    positive = magnitudes[magnitudes > np.finfo(float).tiny]
    if positive.size == 0:
        return 1.0
    return float(np.exp(np.mean(np.log(positive))))


def _real_coefficients(coefficients, name):
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    scale = max(float(np.max(np.abs(coefficients), initial=0.0)), 1.0)
    if np.max(np.abs(coefficients.imag), initial=0.0) > _POLYNOMIAL_TOLERANCE * scale:
        raise ValueError(f"{name} does not have numerically real coefficients")
    return np.asarray(coefficients.real, dtype=np.float64)


def _common_rational_polynomials(poles, residues, direct):
    """Return real numerator/denominator polynomials in a scaled ``s``."""

    scale = _frequency_scale(poles)
    scaled_poles = poles / scale
    scaled_residues = residues / scale
    denominator = np.asarray(np.poly(scaled_poles), dtype=np.complex128)
    numerator = direct * denominator
    for pole, residue in zip(scaled_poles, scaled_residues):
        quotient, remainder = np.polydiv(denominator, np.asarray((1.0, -pole)))
        remainder_scale = max(float(np.linalg.norm(denominator)), 1.0)
        if np.linalg.norm(remainder) > 1e-8 * remainder_scale:
            raise ValueError("failed to construct the common pole polynomial")
        numerator = np.polyadd(numerator, residue * quotient)
    # np.polyadd may trim leading zeros. Restore the common degree so that the
    # subsequent j*w substitutions remain aligned.
    if numerator.size < denominator.size:
        numerator = np.pad(numerator, (denominator.size - numerator.size, 0))
    return (
        _real_coefficients(numerator, "admittance numerator"),
        _real_coefficients(denominator, "admittance denominator"),
        scale,
    )


def _even_polynomial_in_squared_frequency(coefficients, degree):
    """Map an even polynomial in ``w`` to a polynomial in ``x=w**2``."""

    coefficients = _real_coefficients(coefficients, "real-part polynomial")
    if coefficients.size < degree + 1:
        coefficients = np.pad(coefficients, (degree + 1 - coefficients.size, 0))
    scale = max(float(np.max(np.abs(coefficients), initial=0.0)), 1.0)
    result = np.zeros(degree // 2 + 1, dtype=np.float64)
    for index, coefficient in enumerate(coefficients):
        exponent = degree - index
        if exponent % 2:
            if abs(coefficient) > _POLYNOMIAL_TOLERANCE * scale:
                raise ValueError("real admittance on the imaginary axis is not even")
            continue
        result[result.size - 1 - exponent // 2] = coefficient
    return result


def _real_part_polynomials(poles, residues, direct):
    numerator_s, denominator_s, frequency_scale = _common_rational_polynomials(
        poles, residues, direct
    )
    numerator_degree = numerator_s.size - 1
    denominator_degree = denominator_s.size - 1
    numerator_j = numerator_s * (1j ** np.arange(numerator_degree, -1, -1))
    numerator_minus_j = numerator_s * ((-1j) ** np.arange(numerator_degree, -1, -1))
    denominator_j = denominator_s * (1j ** np.arange(denominator_degree, -1, -1))
    denominator_minus_j = denominator_s * (
        (-1j) ** np.arange(denominator_degree, -1, -1)
    )
    real_numerator_w = 0.5 * (
        np.polymul(numerator_j, denominator_minus_j)
        + np.polymul(numerator_minus_j, denominator_j)
    )
    positive_denominator_w = np.polymul(denominator_j, denominator_minus_j)
    common_degree = numerator_degree + denominator_degree
    return (
        _even_polynomial_in_squared_frequency(real_numerator_w, common_degree),
        _even_polynomial_in_squared_frequency(positive_denominator_w, common_degree),
        frequency_scale,
    )


def _trim_polynomial(coefficients):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    scale = float(np.max(np.abs(coefficients), initial=0.0))
    if scale == 0:
        return np.asarray((0.0,))
    threshold = 64 * np.finfo(float).eps * scale
    nonzero = np.flatnonzero(np.abs(coefficients) > threshold)
    if nonzero.size == 0:
        return np.asarray((0.0,))
    return coefficients[int(nonzero[0]) :]


def _nonnegative_real_roots(coefficients):
    coefficients = _trim_polynomial(coefficients)
    if coefficients.size <= 1:
        return np.empty(0, dtype=np.float64)
    roots = np.roots(coefficients / np.max(np.abs(coefficients)))
    real_roots = []
    for root in roots:
        tolerance = _ROOT_TOLERANCE * max(1.0, abs(root.real))
        if abs(root.imag) <= tolerance and root.real >= -tolerance:
            real_roots.append(max(0.0, float(root.real)))
    if not real_roots:
        return np.empty(0, dtype=np.float64)
    real_roots.sort()
    unique = [real_roots[0]]
    for root in real_roots[1:]:
        if abs(root - unique[-1]) > _ROOT_TOLERANCE * max(1.0, abs(root)):
            unique.append(root)
    return np.asarray(unique)


def _evaluate_real_admittance(poles, residues, direct, angular_frequencies):
    frequencies = np.asarray(angular_frequencies, dtype=np.float64)
    values = np.full(frequencies.shape, direct, dtype=np.complex128)
    finite = np.isfinite(frequencies)
    if poles.size and np.any(finite):
        values[finite] += np.sum(
            residues[None, :]
            / (1j * frequencies[finite, None] - poles[None, :]),
            axis=1,
        )
    return np.asarray(values.real, dtype=np.float64)


def _candidate_squared_frequencies(real_numerator, positive_denominator, margin):
    # Extrema of P/Q are roots of P'Q-PQ'.
    derivative_numerator = np.polyder(real_numerator)
    derivative_denominator = np.polyder(positive_denominator)
    stationary = np.polysub(
        np.polymul(derivative_numerator, positive_denominator),
        np.polymul(real_numerator, derivative_denominator),
    )
    stationary_roots = _nonnegative_real_roots(stationary)

    # Roots and sign intervals of P-margin*Q provide an independent guard
    # against a poorly conditioned nearly-multiple stationary root.
    margin_polynomial = np.polysub(real_numerator, margin * positive_denominator)
    crossing_roots = _nonnegative_real_roots(margin_polynomial)
    interval_points = []
    boundaries = np.concatenate((np.asarray((0.0,)), crossing_roots))
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        interval_points.append(lower + 0.5 * (upper - lower))
    if crossing_roots.size:
        interval_points.append(max(1.0, 2.0 * crossing_roots[-1] + 1.0))

    candidates = np.concatenate(
        (
            np.asarray((0.0,)),
            stationary_roots,
            crossing_roots,
            np.asarray(interval_points, dtype=np.float64),
        )
    )
    candidates.sort()
    if candidates.size == 0:
        return np.asarray((0.0,))
    keep = np.concatenate(
        (
            np.asarray((True,)),
            np.diff(candidates)
            > _ROOT_TOLERANCE * np.maximum(1.0, np.abs(candidates[1:])),
        )
    )
    return candidates[keep]


def certify_scalar_positive_real(
    model: PoleResidueAdmittance,
    *,
    margin: float = 0.0,
    stability_margin: float = 0.0,
) -> ScalarPassivityCertificate:
    """Search all computed critical points for scalar positive realness.

    ``margin`` is the required lower bound on ``Re(Y(j*w))``. Poles must lie
    strictly to the left of ``-stability_margin``; imaginary-axis poles are
    intentionally rejected because they are unsafe for the finite-precision
    FDTD recurrence.
    """

    margin = float(margin)
    stability_margin = float(stability_margin)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    if not np.isfinite(stability_margin) or stability_margin < 0:
        raise ValueError("stability_margin must be finite and nonnegative")

    poles, residues, direct_complex = _as_model_arrays(model)
    direct_real = float(direct_complex.real)
    groups = _conjugate_residue_groups(poles, residues)
    real_rational = groups is not None and abs(direct_complex.imag) <= (
        _REAL_STRUCTURE_TOLERANCE * max(1.0, abs(direct_complex))
    )
    stable = bool(np.all(np.real(poles) < -stability_margin))
    direct_ok = bool(real_rational and direct_real >= margin)

    if not real_rational:
        return ScalarPassivityCertificate(
            False,
            stable,
            False,
            False,
            np.nan,
            np.nan,
            margin,
            (),
            (),
            "the pole-residue model is not real-rational",
        )
    if not stable:
        return ScalarPassivityCertificate(
            False,
            False,
            True,
            direct_ok,
            np.nan,
            np.nan,
            margin,
            (),
            (),
            "all poles must lie strictly in the open left half-plane",
        )

    real_numerator, positive_denominator, frequency_scale = _real_part_polynomials(
        poles, residues, direct_real
    )
    squared = _candidate_squared_frequencies(
        real_numerator, positive_denominator, margin
    )
    finite_frequencies = frequency_scale * np.sqrt(np.maximum(0.0, squared))
    # Resonant pole frequencies are cheap extra candidates and improve the
    # reported minimum for extremely narrow, lightly damped violations.
    pole_frequencies = np.abs(np.imag(poles))
    finite_frequencies = np.unique(
        np.concatenate((finite_frequencies, pole_frequencies))
    )
    finite_values = _evaluate_real_admittance(
        poles, residues, direct_real, finite_frequencies
    )
    all_values = np.concatenate((finite_values, np.asarray((direct_real,))))
    minimum_index = int(np.argmin(all_values))
    minimum_value = float(all_values[minimum_index])
    minimum_frequency = (
        float(finite_frequencies[minimum_index])
        if minimum_index < finite_frequencies.size
        else np.inf
    )
    # This tolerance must not scale with the largest response value. A very
    # large, lightly damped positive resonance is unrelated to the accuracy of
    # the minimum and could otherwise mask an order-one negative value at a
    # different frequency. Be deliberately conservative: uncertain minima are
    # rejected and can be revisited with a better-conditioned certificate.
    numerical_tolerance = 256 * np.finfo(float).eps * max(1.0, abs(margin))
    passive = bool(direct_ok and minimum_value >= margin - numerical_tolerance)
    violating = tuple(
        float(frequency)
        for frequency, value in zip(finite_frequencies, finite_values)
        if value < margin - numerical_tolerance
    )
    message = (
        "the model is globally scalar positive-real"
        if passive
        else "the real admittance falls below the requested passivity margin"
    )
    return ScalarPassivityCertificate(
        passive,
        True,
        True,
        direct_ok,
        minimum_value,
        minimum_frequency,
        margin,
        tuple(float(value) for value in finite_frequencies),
        violating,
        message,
    )


def _pack_real_parameters(direct, residues, groups):
    parameters = [float(np.real(direct))]
    for group in groups:
        residue = residues[group.first]
        parameters.append(float(residue.real))
        if group.kind == "pair":
            parameters.append(float(residue.imag))
    return np.asarray(parameters, dtype=np.float64)


def _unpack_real_parameters(parameters, poles, groups):
    residues = np.zeros(poles.size, dtype=np.complex128)
    cursor = 1
    for group in groups:
        if group.kind == "real":
            residues[group.first] = parameters[cursor]
            cursor += 1
        else:
            residue = parameters[cursor] + 1j * parameters[cursor + 1]
            residues[group.first] = residue
            residues[group.second] = residue.conjugate()
            cursor += 2
    return residues


def _linear_response_basis(poles, groups, angular_frequencies):
    frequencies = np.atleast_1d(np.asarray(angular_frequencies, dtype=np.float64))
    if frequencies.ndim != 1 or not np.all(np.isfinite(frequencies)):
        raise ValueError("fit angular frequencies must be a finite one-dimensional array")
    columns = [np.ones(frequencies.size, dtype=np.complex128)]
    for group in groups:
        first_response = 1.0 / (1j * frequencies - poles[group.first])
        if group.kind == "real":
            columns.append(first_response)
        else:
            second_response = 1.0 / (1j * frequencies - poles[group.second])
            columns.append(first_response + second_response)
            columns.append(1j * first_response - 1j * second_response)
    return np.column_stack(columns)


def _default_fit_frequencies(poles):
    if poles.size == 0:
        return np.asarray((0.0, 1.0))
    magnitudes = np.abs(poles)
    positive = magnitudes[magnitudes > np.finfo(float).tiny]
    if positive.size == 0:
        return np.asarray((0.0, 1.0))
    lower = max(float(np.min(positive)) * 1e-3, np.finfo(float).tiny)
    upper = max(float(np.max(positive)) * 1e3, lower * 10.0)
    return np.unique(
        np.concatenate(
            (
                np.asarray((0.0,)),
                np.geomspace(lower, upper, 257),
                np.abs(np.imag(poles)),
            )
        )
    )


def _make_model(poles, residues, direct):
    return PoleResidueAdmittance(
        poles=np.asarray(poles, dtype=np.complex128),
        residues=np.asarray(residues, dtype=np.complex128),
        direct=float(direct),
    )


def repair_scalar_positive_real(
    model: PoleResidueAdmittance,
    *,
    margin: float = 0.0,
    stability_margin: float = 0.0,
    fit_angular_frequencies=None,
    fit_weights=None,
    max_iterations: int = 12,
) -> ScalarPassivityRepair:
    """Minimally repair scalar passivity while keeping all poles fixed.

    The least-squares objective preserves the original complex response at
    ``fit_angular_frequencies``. Global passivity is enforced by iteratively
    adding linear constraints at violations found by
    :func:`certify_scalar_positive_real`.
    """

    initial_certificate = certify_scalar_positive_real(
        model, margin=margin, stability_margin=stability_margin
    )
    if not initial_certificate.is_real_rational:
        raise ValueError("passivity repair requires a real-rational model")
    if not initial_certificate.is_stable:
        raise ValueError("passivity repair does not relocate unstable poles")
    if initial_certificate.is_passive:
        return ScalarPassivityRepair(model, initial_certificate, 0, 0.0)
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be an integer greater than zero")

    poles, residues, direct_complex = _as_model_arrays(model)
    groups = _conjugate_residue_groups(poles, residues)
    original = _pack_real_parameters(direct_complex.real, residues, groups)
    fit_frequencies = (
        _default_fit_frequencies(poles)
        if fit_angular_frequencies is None
        else np.atleast_1d(np.asarray(fit_angular_frequencies, dtype=np.float64))
    )
    response_basis = _linear_response_basis(poles, groups, fit_frequencies)
    if fit_weights is None:
        weights = np.ones(fit_frequencies.size, dtype=np.float64)
    else:
        weights = np.broadcast_to(
            np.asarray(fit_weights, dtype=np.float64), fit_frequencies.shape
        )
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("fit_weights must contain finite positive values")
    weighted_basis = response_basis * np.sqrt(weights)[:, None]
    objective_matrix = np.vstack((weighted_basis.real, weighted_basis.imag))
    column_norms = np.linalg.norm(objective_matrix, axis=0)
    parameter_scales = 1.0 / np.maximum(column_norms, 1e-12)
    scaled_objective = objective_matrix * parameter_scales[None, :]
    hessian = scaled_objective.T @ scaled_objective
    hessian += 1e-12 * max(float(np.trace(hessian)), 1.0) * np.eye(hessian.shape[0])

    constraint_frequencies = {0.0, np.inf}
    constraint_frequencies.update(initial_certificate.violating_frequencies)
    if np.isfinite(initial_certificate.minimum_angular_frequency):
        constraint_frequencies.add(initial_certificate.minimum_angular_frequency)

    latest_model = model
    latest_certificate = initial_certificate
    latest_parameters = original
    enforcement_margin = margin + 1e-10 * max(
        1.0,
        abs(float(direct_complex.real)),
        float(np.max(np.abs(residues), initial=0.0) / _frequency_scale(poles)),
    )
    for iteration in range(1, int(max_iterations) + 1):
        ordered_frequencies = np.asarray(sorted(constraint_frequencies))
        finite = np.isfinite(ordered_frequencies)
        constraint_basis = np.zeros(
            (ordered_frequencies.size, original.size), dtype=np.float64
        )
        if np.any(finite):
            constraint_basis[finite] = _linear_response_basis(
                poles, groups, ordered_frequencies[finite]
            ).real
        constraint_basis[~finite, 0] = 1.0
        scaled_constraints = constraint_basis * parameter_scales[None, :]
        # Solve a hair inside the passive cone so that optimizer and storage
        # roundoff cannot leave an apparently repaired model just outside it.
        lower_bounds = enforcement_margin - constraint_basis @ original

        def objective(delta):
            return 0.5 * float(delta @ hessian @ delta)

        def gradient(delta):
            return hessian @ delta

        result = minimize(
            objective,
            np.zeros(original.size, dtype=np.float64),
            jac=gradient,
            method="SLSQP",
            constraints=(
                LinearConstraint(
                    scaled_constraints,
                    lower_bounds,
                    np.full(lower_bounds.shape, np.inf),
                ),
            ),
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not result.success:
            raise RuntimeError(f"scalar passivity repair failed: {result.message}")
        latest_parameters = original + parameter_scales * result.x
        latest_model = _make_model(
            poles,
            _unpack_real_parameters(latest_parameters, poles, groups),
            latest_parameters[0],
        )
        latest_certificate = certify_scalar_positive_real(
            latest_model, margin=margin, stability_margin=stability_margin
        )
        if latest_certificate.is_passive:
            # ``result.x`` and ``original / parameter_scales`` live in the
            # response-weighted scaled coordinate used by the optimizer. This
            # makes the reported correction dimensionless and invariant to a
            # change of frequency units; a raw norm mixing D with dimensional
            # residues is not meaningful.
            scaled_original = original / parameter_scales
            change = np.linalg.norm(result.x) / max(
                np.linalg.norm(scaled_original), 1.0
            )
            return ScalarPassivityRepair(
                latest_model, latest_certificate, iteration, float(change)
            )
        previous_count = len(constraint_frequencies)
        constraint_frequencies.update(latest_certificate.violating_frequencies)
        if np.isfinite(latest_certificate.minimum_angular_frequency):
            constraint_frequencies.add(latest_certificate.minimum_angular_frequency)
        if len(constraint_frequencies) == previous_count:
            break

    raise RuntimeError(
        "scalar passivity repair did not obtain a global positive-real certificate "
        f"after {max_iterations} iterations; minimum real admittance is "
        f"{latest_certificate.minimum_real_admittance:g}"
    )


def enforce_scalar_positive_real(
    model: PoleResidueAdmittance,
    **kwargs,
) -> PoleResidueAdmittance:
    """Return the globally certified model produced by passivity repair."""

    return repair_scalar_positive_real(model, **kwargs).model

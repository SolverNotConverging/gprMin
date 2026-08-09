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

"""Real-state ADE for a rational, power-normalized modal admittance.

The fitted characteristic admittance is kept separate from the Yee terminal
half-cell storage,

``Yc(s) = D + sum(residue[k] / (s - pole[k]))``

and the coupled boundary equation is

``Eh * dV/dt + Yc(s) (V - 2a) = Iout``.

Here ``a`` is the incident modal voltage and ``Iout`` is the outward modal
current.  Trapezoidal integration is used throughout, so a continuous stable,
positive-real fit retains the bilinear transform's passivity properties.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gprMax.modal_admittance import PoleResidueAdmittance


__all__ = [
    "RealStateSpaceAdmittance",
    "RationalModalAdmittanceADE",
    "bilinear_prewarp_angular_frequency",
    "pole_residue_to_real_state_space",
]


_REALIZATION_TOLERANCE = 1e-10


def _finite_real_scalar(value, name: str, *, tolerance: float = _REALIZATION_TOLERANCE) -> float:
    """Return a finite effectively-real scalar."""

    scalar = complex(value)
    scale = max(1.0, abs(scalar.real), abs(scalar.imag))
    if not np.isfinite(scalar) or abs(scalar.imag) > tolerance * scale:
        raise ValueError(f"{name} must be finite and effectively real")
    return float(scalar.real)


def bilinear_prewarp_angular_frequency(angular_frequency, dt: float):
    """Map physical angular frequency through the trapezoidal bilinear map.

    The returned value is

    ``omega_a = (2 / dt) * tan(omega_d * dt / 2)``.

    Scalar input produces a scalar float. Array-like input produces a NumPy
    array of the same broadcast shape. Frequencies must lie strictly below the
    discrete Nyquist angular frequency in magnitude.
    """

    dt = _finite_real_scalar(dt, "dt")
    if dt <= 0:
        raise ValueError("dt must be positive")
    omega = np.asarray(angular_frequency, dtype=np.float64)
    if not np.all(np.isfinite(omega)):
        raise ValueError("angular_frequency must contain only finite values")
    nyquist = np.pi / dt
    if np.any(np.abs(omega) >= nyquist):
        raise ValueError("angular_frequency must lie strictly below the Nyquist frequency")
    result = (2.0 / dt) * np.tan(0.5 * dt * omega)
    if np.ndim(angular_frequency) == 0:
        return float(result)
    return result


@dataclass(frozen=True)
class RealStateSpaceAdmittance:
    """A real, proper SISO realization ``D + C (sI - A)^-1 B``."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    direct: float

    def __post_init__(self):
        A = np.ascontiguousarray(self.A, dtype=np.float64)
        B = np.ascontiguousarray(self.B, dtype=np.float64).reshape(-1)
        C = np.ascontiguousarray(self.C, dtype=np.float64).reshape(-1)
        direct = _finite_real_scalar(self.direct, "direct")
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix")
        if B.shape != (A.shape[0],) or C.shape != (A.shape[0],):
            raise ValueError("B and C must contain one value per state")
        if not all(np.all(np.isfinite(values)) for values in (A, B, C)):
            raise ValueError("state-space arrays must contain only finite values")
        A = np.frombuffer(A.tobytes(), dtype=np.float64).reshape(A.shape)
        B = np.frombuffer(B.tobytes(), dtype=np.float64)
        C = np.frombuffer(C.tobytes(), dtype=np.float64)
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "C", C)
        object.__setattr__(self, "direct", direct)

    @property
    def state_count(self) -> int:
        return int(self.A.shape[0])

    def evaluate(self, s):
        """Evaluate the continuous characteristic admittance at complex ``s``."""

        values = np.asarray(s, dtype=np.complex128)
        flattened = values.reshape(-1)
        response = np.full(flattened.shape, complex(self.direct), dtype=np.complex128)
        if self.state_count:
            identity = np.eye(self.state_count, dtype=np.complex128)
            for index, point in enumerate(flattened):
                response[index] += self.C @ np.linalg.solve(point * identity - self.A, self.B)
        response = response.reshape(values.shape)
        if np.ndim(s) == 0:
            return complex(response.item())
        return response


def pole_residue_to_real_state_space(
    model: PoleResidueAdmittance,
    *,
    tolerance: float = _REALIZATION_TOLERANCE,
) -> RealStateSpaceAdmittance:
    """Convert explicit real/conjugate pole-residue data to real state space.

    Every non-real pole and residue must have its conjugate explicitly present
    in ``model``. A conjugate pair becomes one real 2-by-2 state block, while a
    real pole becomes one scalar state. The represented transfer function is
    unchanged; a pair is not counted twice after conversion.
    """

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    poles = np.asarray(model.poles, dtype=np.complex128).reshape(-1)
    residues = np.asarray(model.residues, dtype=np.complex128).reshape(-1)
    if poles.shape != residues.shape:
        raise ValueError("poles and residues must have identical shapes")
    if not np.all(np.isfinite(poles)) or not np.all(np.isfinite(residues)):
        raise ValueError("poles and residues must contain only finite values")
    if np.any(np.real(poles) >= 0):
        raise ValueError("all rational-admittance poles must lie in the open left half-plane")
    direct_value = getattr(model, "direct", getattr(model, "D", None))
    direct = _finite_real_scalar(direct_value, "direct", tolerance=tolerance)

    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    used = np.zeros(poles.size, dtype=bool)
    for index, (pole, residue) in enumerate(zip(poles, residues)):
        if used[index]:
            continue
        pole_scale = max(1.0, abs(pole))
        residue_scale = max(1.0, abs(residue))
        if abs(pole.imag) <= tolerance * pole_scale:
            if abs(residue.imag) > tolerance * residue_scale:
                raise ValueError("a real pole must have an effectively real residue")
            blocks.append(
                (
                    np.asarray(((pole.real,),), dtype=np.float64),
                    np.asarray((1.0,), dtype=np.float64),
                    np.asarray((residue.real,), dtype=np.float64),
                )
            )
            used[index] = True
            continue

        candidates = [
            candidate
            for candidate in range(index + 1, poles.size)
            if not used[candidate]
            and np.isclose(poles[candidate], np.conj(pole), rtol=tolerance, atol=tolerance)
            and np.isclose(
                residues[candidate],
                np.conj(residue),
                rtol=tolerance,
                atol=tolerance,
            )
        ]
        if not candidates:
            raise ValueError(
                "each complex pole and residue must have one explicit conjugate partner"
            )
        partner = candidates[0]
        used[index] = True
        used[partner] = True
        # Use the positive-imaginary representative so the block convention is
        # deterministic regardless of the input pair's ordering.
        if pole.imag < 0:
            pole = np.conj(pole)
            residue = np.conj(residue)
        alpha = float(pole.real)
        beta = float(pole.imag)
        gamma = float(residue.real)
        delta = float(residue.imag)
        blocks.append(
            (
                np.asarray(((alpha, -beta), (beta, alpha)), dtype=np.float64),
                np.asarray((1.0, 0.0), dtype=np.float64),
                np.asarray((2.0 * gamma, -2.0 * delta), dtype=np.float64),
            )
        )

    state_count = sum(block[0].shape[0] for block in blocks)
    A = np.zeros((state_count, state_count), dtype=np.float64)
    B = np.zeros(state_count, dtype=np.float64)
    C = np.zeros(state_count, dtype=np.float64)
    offset = 0
    for block_a, block_b, block_c in blocks:
        stop = offset + block_a.shape[0]
        A[offset:stop, offset:stop] = block_a
        B[offset:stop] = block_b
        C[offset:stop] = block_c
        offset = stop
    return RealStateSpaceAdmittance(A=A, B=B, C=C, direct=direct)


class RationalModalAdmittanceADE:
    """Trapezoidal real-state update for one rational modal termination."""

    def __init__(
        self,
        model: PoleResidueAdmittance,
        *,
        dt: float,
        half_cell_storage: float,
    ):
        self.dt = _finite_real_scalar(dt, "dt")
        self.half_cell_storage = _finite_real_scalar(
            half_cell_storage, "half_cell_storage"
        )
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.half_cell_storage <= 0:
            raise ValueError("half_cell_storage must be positive")

        # Construction is a safety boundary, not merely a realization helper.
        # Re-certify the exact stored coefficients so a caller cannot bypass
        # mandatory passivity enforcement by supplying a stable but active
        # pole-residue model directly.
        from gprMax.modal_admittance_passivity import certify_scalar_positive_real

        certificate = certify_scalar_positive_real(model)
        if not certificate.is_passive:
            raise ValueError(
                "rational modal-admittance ADE requires a globally "
                f"positive-real model: {certificate.message}"
            )
        self.model = model
        self.passivity_certificate = certificate
        self.realization = pole_residue_to_real_state_space(model)
        state_count = self.realization.state_count
        identity = np.eye(state_count, dtype=np.float64)
        if state_count:
            left = identity - 0.5 * self.dt * self.realization.A
            right = identity + 0.5 * self.dt * self.realization.A
            self.discrete_A = np.ascontiguousarray(
                np.linalg.solve(left, right), dtype=np.float64
            )
            self.discrete_B = np.ascontiguousarray(
                np.linalg.solve(left, self.dt * self.realization.B),
                dtype=np.float64,
            )
        else:
            self.discrete_A = np.empty((0, 0), dtype=np.float64)
            self.discrete_B = np.empty(0, dtype=np.float64)

        cbd = float(self.realization.C @ self.discrete_B)
        storage_over_dt = self.half_cell_storage / self.dt
        direct = self.realization.direct
        self.boundary_denominator = storage_over_dt + 0.5 * direct + 0.25 * cbd
        self.previous_voltage_coefficient = storage_over_dt - 0.5 * direct - 0.25 * cbd
        self.incident_voltage_coefficient = 2.0 * direct + cbd
        self.previous_state_coefficients = np.ascontiguousarray(
            0.5 * self.realization.C @ (self.discrete_A + identity),
            dtype=np.float64,
        )
        if (
            not np.isfinite(self.boundary_denominator)
            or self.boundary_denominator <= 0
        ):
            raise ValueError(
                "the trapezoidal modal-admittance boundary denominator must be positive"
            )
        coefficient_values = (
            self.previous_voltage_coefficient,
            self.incident_voltage_coefficient,
        )
        if not all(np.isfinite(value) for value in coefficient_values) or not np.all(
            np.isfinite(self.previous_state_coefficients)
        ):
            raise ValueError("the trapezoidal modal-admittance coefficients are invalid")

        self._state = np.zeros(state_count, dtype=np.float64)
        self._voltage = 0.0

    @property
    def state(self) -> np.ndarray:
        """Return a copy of the current real ADE state."""

        return self._state.copy()

    @property
    def voltage(self) -> float:
        return float(self._voltage)

    def reset(self, *, voltage: float = 0.0, state=None):
        """Reset the modal voltage and rational states."""

        voltage = _finite_real_scalar(voltage, "voltage")
        if state is None:
            values = np.zeros(self.realization.state_count, dtype=np.float64)
        else:
            values = np.asarray(state, dtype=np.float64)
            if values.shape != (self.realization.state_count,):
                raise ValueError(
                    "state must contain exactly one value per real realization state"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError("state must contain only finite values")
            values = np.array(values, dtype=np.float64, copy=True)
        self._voltage = voltage
        self._state = values

    def characteristic_admittance(self, s):
        """Evaluate the proper fitted characteristic admittance ``Yc(s)``."""

        return self.realization.evaluate(s)

    def load_admittance(self, s):
        """Evaluate ``Eh*s + Yc(s)`` in the continuous Laplace domain."""

        values = np.asarray(s, dtype=np.complex128)
        result = self.half_cell_storage * values + self.characteristic_admittance(values)
        if np.ndim(s) == 0:
            return complex(np.asarray(result).item())
        return result

    def discrete_load_admittance(self, angular_frequency):
        """Evaluate the trapezoidal load at a physical discrete-time frequency."""

        mapped = bilinear_prewarp_angular_frequency(angular_frequency, self.dt)
        return self.load_admittance(1j * np.asarray(mapped))

    def step(self, outward_current: float, incident_voltage: float = 0.0) -> float:
        """Advance one E sample and return the new generalized modal voltage."""

        current = _finite_real_scalar(outward_current, "outward_current")
        incident = _finite_real_scalar(incident_voltage, "incident_voltage")
        old_voltage = self._voltage
        old_state = self._state
        rhs = (
            current
            + self.previous_voltage_coefficient * old_voltage
            + self.incident_voltage_coefficient * incident
            - float(self.previous_state_coefficients @ old_state)
        )
        new_voltage = rhs / self.boundary_denominator
        centred_characteristic_voltage = 0.5 * (new_voltage + old_voltage) - 2.0 * incident
        new_state = (
            self.discrete_A @ old_state
            + self.discrete_B * centred_characteristic_voltage
        )
        if not np.isfinite(new_voltage) or not np.all(np.isfinite(new_state)):
            raise ValueError("the rational modal-admittance update produced non-finite state")
        self._voltage = float(new_voltage)
        self._state = np.asarray(new_state, dtype=np.float64)
        return self._voltage

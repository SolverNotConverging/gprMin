"""Conservative sparse curl reference operator and magnetic-cell aggregation.

This is an internal numerical building block, not a Scene compiler. The
compiler must supply oriented curl integrals, positive electric/magnetic
Hodge masses, and verified open-face adjacency. In particular, positive
masses alone do not certify the accuracy of a geometric discretisation.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags


def _immutable(array, dtype=None):
    array = np.ascontiguousarray(array, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class Aggregation:
    operator: "CurlOperator"
    # Fine H = prolongation @ coarse H. Restriction of magnetic flux is
    # prolongation.T @ (fine magnetic mass * fine H).
    prolongation: csr_matrix


class CurlOperator:
    """M_h dH/dt = -C E, M_e dE/dt = C.T H.

    The transpose pairing is essential: independently modifying E and H
    coefficients destroys the energy identity. Physical units belong in C
    and the masses; dt is in seconds when the inputs use SI units.
    """

    def __init__(self, curl, electric_mass, magnetic_mass):
        self.curl = csr_matrix(curl, dtype=np.float64, copy=True)
        self.curl.check_format(full_check=True)
        self.curl.sum_duplicates()
        self.curl.eliminate_zeros()
        self.curl.sort_indices()
        self.electric_mass = np.array(electric_mass, dtype=np.float64, copy=True)
        self.magnetic_mass = np.array(magnetic_mass, dtype=np.float64, copy=True)
        nh, ne = self.curl.shape
        if self.electric_mass.shape != (ne,) or self.magnetic_mass.shape != (nh,):
            raise ValueError("mass dimensions must match curl columns and rows")
        if not nh or not ne:
            raise ValueError("operator requires electric and magnetic degrees of freedom")
        for mass in (self.electric_mass, self.magnetic_mass):
            if not np.all(np.isfinite(mass)) or np.any(mass <= 0):
                raise ValueError("Hodge masses must be finite and strictly positive")
        if not np.all(np.isfinite(self.curl.data)):
            raise ValueError("curl coefficients must be finite")
        self.transpose = self.curl.T.tocsr()
        self._cfl = self._cfl_bound()
        self.electric_mass = _immutable(self.electric_mass)
        self.magnetic_mass = _immutable(self.magnetic_mass)
        self._packed = []
        for matrix in (self.curl, self.transpose):
            matrix.data = _immutable(matrix.data)
            matrix.indices = _immutable(matrix.indices)
            matrix.indptr = _immutable(matrix.indptr)
            self._packed.append((_immutable(matrix.indptr, np.int64),
                                 _immutable(matrix.indices, np.int64), matrix.data))

    def _cfl_bound(self):
        # ||A||_2 <= sqrt(||A||_1 ||A||_inf), so this is a conservative
        # sufficient bound, not an estimate based solely on cut-face area.
        scaled = diags(1 / np.sqrt(self.magnetic_mass)) @ self.curl @ diags(
            1 / np.sqrt(self.electric_mass))
        absolute = abs(scaled)
        norm1 = float(np.max(np.asarray(absolute.sum(axis=0))))
        norminf = float(np.max(np.asarray(absolute.sum(axis=1))))
        norm = np.sqrt(norm1) * np.sqrt(norminf)
        return np.inf if norm == 0 else 2 / norm

    @property
    def cfl_limit(self):
        """Conservative certified timestep ceiling (equality excluded)."""
        return self._cfl

    def validate_timestep(self, dt):
        if not np.isfinite(dt) or dt <= 0 or dt >= self.cfl_limit:
            raise ValueError("timestep must be positive and strictly below the certified CFL bound")

    def _validate_fields(self, electric, magnetic):
        for field, mass in ((electric, self.electric_mass), (magnetic, self.magnetic_mass)):
            if not isinstance(field, np.ndarray) or field.shape != mass.shape:
                raise ValueError("field dimensions must match the operator")
            if field.dtype not in (np.dtype("float32"), np.dtype("float64")):
                raise ValueError("fields must use float32 or float64")
            if not field.flags.c_contiguous or not field.flags.writeable:
                raise ValueError("fields must be contiguous and writable")
            if not np.all(np.isfinite(field)):
                raise ValueError("fields must be finite")
        if electric.dtype != magnetic.dtype or np.shares_memory(electric, magnetic):
            raise ValueError("E and H must have matching precision and independent storage")

    def step(self, electric, magnetic, dt, *, backend="reference", impedance=None):
        """Advance E(n), H(n-1/2) in place through one leapfrog step."""
        self.validate_timestep(dt)
        self._validate_fields(electric, magnetic)
        if impedance is not None:
            if impedance.operator is not self or impedance.dt != dt:
                raise ValueError("impedance ports must be compiled for this operator and timestep")
            old_electric = electric.copy()
        if backend == "reference":
            magnetic -= dt * (self.curl @ electric) / self.magnetic_mass
            electric += dt * (self.transpose @ magnetic) / self.electric_mass
        elif backend == "cython":
            from gprMax.cython.conformal_updates import update_sparse

            update_sparse(*self._packed[0], self.magnetic_mass, electric, magnetic, -dt)
            update_sparse(*self._packed[1], self.electric_mass, magnetic, electric, dt)
        else:
            raise ValueError("backend must be reference or cython")
        if impedance is not None:
            impedance.correct(old_electric, electric)

    def energy(self, electric, magnetic, dt):
        """Conserved staggered energy at E(n), H(n-1/2), without forcing.

        This is the leapfrog invariant, not the collocated physical energy.
        It is positive definite under the accepted CFL bound.
        """
        self.validate_timestep(dt)
        self._validate_fields(electric, magnetic)
        return float(0.5 * (
            np.dot(self.electric_mass * electric, electric)
            + np.dot(self.magnetic_mass * magnetic, magnetic)
            - dt * np.dot(magnetic, self.curl @ electric)))

    def aggregate_magnetic(self, groups, *, open_adjacency, region_labels):
        """Conservatively enlarge connected magnetic control volumes.

        ``groups`` partitions every magnetic row, including unchanged rows.
        The geometry compiler must certify ``open_adjacency``: an entry
        means an open shared face, never an intervening PEC/SIBC sheet.
        ``region_labels`` distinguishes disconnected field domains. The
        resulting operator must still pass validate_timestep at the target
        dt; aggregation does not promise to cure every small cut cell.
        """
        nh = self.curl.shape[0]
        groups = tuple(tuple(group) for group in groups)
        flat = [row for group in groups for row in group]
        if (not groups or any(not group for group in groups)
                or any(not isinstance(row, (int, np.integer)) for row in flat)
                or sorted(flat) != list(range(nh))):
            raise ValueError("aggregation must partition all magnetic rows exactly once")
        if len(region_labels) != nh:
            raise ValueError("one connected-region label is required per magnetic row")
        adjacency = [set() for _ in range(nh)]
        for a, b in open_adjacency:
            if not (0 <= a < nh and 0 <= b < nh) or a == b:
                raise ValueError("invalid open-face adjacency")
            adjacency[a].add(b)
            adjacency[b].add(a)
        columns = np.empty(nh, dtype=np.int64)
        for column, group in enumerate(groups):
            if any(region_labels[row] != region_labels[group[0]] for row in group):
                raise ValueError("enlargement cannot join disconnected boundary sides")
            allowed = set(group)
            visited, pending = set(), [group[0]]
            while pending:
                row = pending.pop()
                if row not in visited:
                    visited.add(row)
                    pending.extend((adjacency[row] & allowed) - visited)
            if visited != allowed:
                raise ValueError("enlarged control volume must be connected through open faces")
            columns[list(group)] = column
        prolongation = csr_matrix((np.ones(nh), (np.arange(nh), columns)),
                                  shape=(nh, len(groups)))
        curl = prolongation.T @ self.curl
        mass = np.asarray(prolongation.T @ self.magnetic_mass).ravel()
        return Aggregation(CurlOperator(curl, self.electric_mass, mass), prolongation)

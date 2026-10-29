"""Integral Yee operator assembled from already averaged bulk materials.

The geometric measures select PEC/SIBC exterior degrees of freedom. The
material IDs and their averaged constitutive parameters come from the normal
grid builder. No dielectric interface is reclassified by this compiler.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.sparse import coo_matrix

from gprMax.conformal.operator import CurlOperator


@dataclass(frozen=True)
class YeePacking:
    operator: CurlOperator
    electric_indices: np.ndarray
    magnetic_indices: np.ndarray
    electric_loss: np.ndarray
    magnetic_loss: np.ndarray
    magnetic_prolongation: object = None
    fine_magnetic_mass: np.ndarray | None = None

    def _require_unsplit_projection(self):
        for indices in (self.electric_indices, self.magnetic_indices):
            if len(set(map(tuple, indices))) != len(indices):
                raise ValueError("split fields require an explicit side-aware field projection")

    def gather(self, grid):
        self._require_unsplit_projection()
        values = []
        for names, indices in ((('Ex', 'Ey', 'Ez'), self.electric_indices),
                               (('Hx', 'Hy', 'Hz'), self.magnetic_indices)):
            fields = [getattr(grid, name) for name in names]
            values.append(np.asarray([fields[c][i, j, k] for c, i, j, k in indices],
                                     dtype=fields[0].dtype))
        if self.magnetic_prolongation is not None:
            values[1] = np.asarray(self.magnetic_prolongation.T @ (
                self.fine_magnetic_mass * values[1])).ravel() / self.operator.magnetic_mass
            values[1] = values[1].astype(values[0].dtype)
        return tuple(values)

    def scatter(self, grid, electric, magnetic):
        self._require_unsplit_projection()
        if self.magnetic_prolongation is not None:
            magnetic = self.magnetic_prolongation @ magnetic
        for names, indices, values in ((('Ex', 'Ey', 'Ez'), self.electric_indices, electric),
                                      (('Hx', 'Hy', 'Hz'), self.magnetic_indices, magnetic)):
            for component, name in enumerate(names):
                select = indices[:, 0] == component
                getattr(grid, name)[tuple(indices[select, 1:].T)] = values[select]

    def step(self, electric, magnetic, dt, *, backend="cython", magnetic_impedance=None):
        """Paired leapfrog with midpoint bulk electric/magnetic conduction."""
        self.operator.validate_timestep(dt)
        self.operator._validate_fields(electric, magnetic)
        if magnetic_impedance is not None:
            if magnetic_impedance.operator is not self.operator or magnetic_impedance.dt != dt:
                raise ValueError("magnetic ports must match this operator and timestep")
            if not np.array_equal(magnetic_impedance.bulk_loss, self.magnetic_loss):
                raise ValueError("magnetic ports must include this packing's bulk magnetic loss")
        me, mh = self.operator.electric_mass, self.operator.magnetic_mass
        eh = dt * self.electric_loss / (2 * me)
        hh = dt * self.magnetic_loss / (2 * mh)
        old_h = magnetic.copy()
        old_e = electric.copy()
        # The paired updates must use the already damped H half-step.
        if backend == "reference":
            magnetic[:] = old_h - dt * (self.operator.curl @ electric) / mh
            if magnetic_impedance is not None:
                magnetic_impedance.correct(old_h, magnetic)
            else:
                magnetic[:] = (magnetic - hh*old_h) / (1+hh)
            electric[:] = ((1-eh)*old_e + dt * (self.operator.transpose @ magnetic) / me) / (1+eh)
        elif backend == "cython":
            from gprMax.cython.conformal_updates import update_sparse
            update_sparse(*self.operator._packed[0], mh, electric, magnetic, -dt)
            if magnetic_impedance is not None:
                magnetic_impedance.correct(old_h, magnetic)
            else:
                magnetic[:] = (magnetic - hh*old_h) / (1+hh)
            update_sparse(*self.operator._packed[1], me, magnetic, electric, dt)
            electric[:] = (electric - eh*old_e) / (1+eh)
        else:
            raise ValueError("backend must be reference or cython")


def compile_yee(grid, electric_lengths, magnetic_areas, *, epsilon0, mu0):
    """Compile a 3D rectangular Yee topology with physical cut measures.

    Lengths and areas have the common padded component-array shape. This
    entry point represents at most one exterior region per edge/face;
    disconnected cuts must use an expanded topology, not summed measures.
    """
    size = np.array([grid.nx, grid.ny, grid.nz], dtype=int)
    dl = np.asarray(grid.dl, dtype=float)
    shape = (3, *(size+1))
    lengths, areas = np.asarray(electric_lengths, dtype=float), np.asarray(magnetic_areas, dtype=float)
    if lengths.shape != shape or areas.shape != shape:
        raise ValueError("cut measures must match padded Yee component shapes")
    if (not np.all(np.isfinite(lengths)) or not np.all(np.isfinite(areas))
            or np.any(lengths < 0) or np.any(areas < 0)):
        raise ValueError("cut measures must be finite and nonnegative")
    if not np.all(np.isfinite(dl)) or np.any(dl <= 0):
        raise ValueError("grid spacings must be finite and positive")
    if not np.isfinite(epsilon0) or not np.isfinite(mu0) or epsilon0 <= 0 or mu0 <= 0:
        raise ValueError("vacuum constitutive constants must be finite and positive")
    volume = float(np.prod(dl))
    e_indices, h_indices, me, mh, se, sm = [], [], [], [], [], []
    e_map = {}
    for axis in range(3):
        ranges = [range(size[d]) if d == axis else range(1, size[d]) for d in range(3)]
        for index in product(*ranges):
            length = lengths[(axis, *index)]
            material = grid.materials[int(grid.ID[(axis, *index)])]
            if not length or material.is_pec:
                continue
            if length > dl[axis]*(1+1e-12):
                raise ValueError("retained edge length exceeds its Yee edge")
            if getattr(material, "poles", 0):
                raise ValueError("dispersive bulk requires its ADE operator alongside conformal topology")
            mass = epsilon0 * material.er * length * volume / dl[axis]
            loss = material.se * length * volume / dl[axis]
            if not np.isfinite(loss) or loss < 0:
                raise ValueError("electric bulk loss must be finite and nonnegative")
            e_map[(axis, *index)] = len(e_indices)
            e_indices.append((axis, *index))
            me.append(mass)
            se.append(loss)
    rows, cols, values = [], [], []
    for axis in range(3):
        u, v = (axis+1) % 3, (axis+2) % 3
        ranges = [range(size[d]+1) if d == axis else range(size[d]) for d in range(3)]
        for index in product(*ranges):
            area = areas[(axis, *index)]
            material = grid.materials[int(grid.ID[(axis+3, *index)])]
            if not area or material.is_pmc:
                continue
            if area > volume / dl[axis]*(1+1e-12):
                raise ValueError("retained face area exceeds its Yee face")
            # The outer normal face has half of a dual edge in the domain.
            # It has no incident active tangential E edge and remains static.
            dual_length = dl[axis] * (0.5 if index[axis] in (0, size[axis]) else 1)
            mass = mu0 * material.mr * area * dual_length
            loss = material.sm * area * dual_length
            if not np.isfinite(loss) or loss < 0:
                raise ValueError("magnetic bulk loss must be finite and nonnegative")
            row = len(h_indices)
            h_indices.append((axis, *index))
            mh.append(mass)
            sm.append(loss)
            # Positive circulation is E_u(bottom), E_v(right),
            # -E_u(top), -E_v(left) in cyclic (u,v) coordinates.
            for component, shift_axis, shift, sign in ((u, v, 0, 1), (v, u, 1, 1),
                                                       (u, v, 1, -1), (v, u, 0, -1)):
                point = list(index)
                point[shift_axis] += shift
                key = (component, *point)
                if key in e_map:
                    rows.append(row)
                    cols.append(e_map[key])
                    values.append(sign * lengths[key] * dual_length)
    curl = coo_matrix((values, (rows, cols)), shape=(len(h_indices), len(e_indices))).tocsr()
    return YeePacking(CurlOperator(curl, me, mh), np.asarray(e_indices, dtype=np.int64),
                      np.asarray(h_indices, dtype=np.int64), np.asarray(se), np.asarray(sm))


def full_yee_measures(grid):
    shape = (3, grid.nx+1, grid.ny+1, grid.nz+1)
    dl = np.asarray(grid.dl)
    lengths = np.broadcast_to(dl[:, None, None, None], shape).copy()
    areas = np.broadcast_to((np.prod(dl)/dl)[:, None, None, None], shape).copy()
    return lengths, areas

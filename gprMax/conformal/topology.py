"""Compile connected face regions and split edge fields from physical geometry.

The supplied material grid contains the ordinary averaged dielectric
coefficients. Conductors are supplied by the continuous section compiler.
Every disconnected face region and every disconnected edge interval owns a
separate field degree of freedom, including opposite sides of opaque sheets.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.sparse import coo_matrix

from gprMax.conformal.operator import CurlOperator
from gprMax.conformal.yee import YeePacking


@dataclass(frozen=True)
class CutDiagnostic:
    component: int
    index: tuple
    reason: str


@dataclass(frozen=True)
class MagneticBoundaryPort:
    magnetic_row: int
    weight: float
    model_id: str
    geometric_owner: int


class ExpandedTopologyError(ValueError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__(f"{diagnostic.reason}: component {diagnostic.component}, index {diagnostic.index}")


def _shared_length(first, second, tolerance):
    for a, b in zip(first, np.roll(first, -1, axis=0)):
        direction = b-a
        length = np.linalg.norm(direction)
        if length <= tolerance:
            continue
        unit = direction / length
        for c, d in zip(second, np.roll(second, -1, axis=0)):
            if (np.linalg.norm(np.cross(c-a, unit)) <= tolerance
                    and np.linalg.norm(np.cross(d-a, unit)) <= tolerance):
                lo, hi = sorted(((c-a) @ unit, (d-a) @ unit))
                if min(length, hi) - max(0, lo) > tolerance:
                    return True
    return False


def connected_pieces(pieces, tolerance):
    """Join artificial clipping subdivisions, preserving physical sheet sides."""
    parents = list(range(len(pieces)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for i, first in enumerate(pieces):
        for j in range(i):
            second = pieces[j]
            if first.sides == second.sides and _shared_length(first.vertices, second.vertices, tolerance):
                parents[root(i)] = root(j)
    groups = {}
    for i, piece in enumerate(pieces):
        groups.setdefault(root(i), []).append(piece)
    return tuple(tuple(group) for group in groups.values())


def _union_intervals(intervals, tolerance):
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1] + tolerance:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return merged


def _face_edges(axis, index, dl):
    u, v = (axis+1) % 3, (axis+2) % 3
    start = np.asarray(index, dtype=float) * dl
    vertices = [start.copy() for _ in range(4)]
    vertices[1][u] += dl[u]
    vertices[2][u] += dl[u]
    vertices[2][v] += dl[v]
    vertices[3][v] += dl[v]
    return np.asarray(vertices), (
        (u, tuple(index), 1),
        (v, tuple(np.asarray(index) + np.eye(3, dtype=int)[u]), 1),
        (u, tuple(np.asarray(index) + np.eye(3, dtype=int)[v]), -1),
        (v, tuple(index), -1),
    )


@dataclass(frozen=True)
class CutTopology:
    packing: YeePacking
    electric_intervals: np.ndarray
    magnetic_areas: np.ndarray
    magnetic_sides: tuple
    geometric_digest: str
    # Original component identity remains available after enlargement.
    diagnostics: tuple[CutDiagnostic, ...]
    boundary_ports: tuple[MagneticBoundaryPort, ...] = ()

    def impedance_ports(self, models, dt):
        from gprMax.conformal.impedance import MagneticImpedancePorts

        if not self.boundary_ports:
            return None
        count = len(self.boundary_ports)
        trace = coo_matrix((np.ones(count), (np.arange(count),
                           [port.magnetic_row for port in self.boundary_ports])),
                           shape=(count, len(self.packing.magnetic_indices))).tocsr()
        return MagneticImpedancePorts(self.packing.operator, trace,
            [port.weight for port in self.boundary_ports],
            [models[port.model_id] for port in self.boundary_ports], dt,
            bulk_loss=self.packing.magnetic_loss)

    @property
    def extra_electric_dofs(self):
        indices = self.packing.electric_indices
        return len(indices) - len(set(map(tuple, indices)))

    @property
    def extra_magnetic_dofs(self):
        indices = self.packing.magnetic_indices
        return len(indices) - len(set(map(tuple, indices)))


def compile_topology(grid, compiler, *, epsilon0, mu0):
    """Assemble the explicit PEC exterior topology on a 3D Yee lattice.

    SIBC surfaces supply magnetic impedance ports for the cut-face boundary
    circulation. They use the good-conductor approximation of vanishing
    surface-normal H. Finite sheet tips need an expanded local cell complex
    and are reported before publishing an incomplete topology.
    """
    size, dl = np.asarray(grid.size, dtype=int), np.asarray(grid.dl, dtype=float)
    volume = float(np.prod(dl))
    tolerance = float(np.min(dl)) * 1e-10
    e_indices, e_intervals, me, se = [], [], [], []
    edge_map = {}
    for component in range(3):
        ranges = [range(size[d]) if d == component else range(1, size[d]) for d in range(3)]
        for index in product(*ranges):
            start = np.asarray(index) * dl
            stop = start.copy()
            stop[component] += dl[component]
            regions, sheets = compiler.scene.partition_segment(start, stop)
            # Sheet crossings divide edges. Coplanar sheet intervals remove
            # the tangential PEC edge; SIBC supplies its own boundary trace.
            barriers = [span[0] for _, span in sheets if span[0] == span[1]]
            fluid = []
            for region in regions:
                if region.boundary_kind in ("pec", "sibc"):
                    continue
                middle = (region.start + region.stop) / 2
                if any(lo <= middle <= hi for _, (lo, hi) in sheets if hi > lo):
                    continue
                if (fluid and abs(fluid[-1][1] - region.start) < 1e-12
                        and not any(abs(region.start - barrier) < 1e-12 for barrier in barriers)):
                    fluid[-1] = (fluid[-1][0], region.stop)
                else:
                    fluid.append((region.start, region.stop))
            material = grid.materials[int(grid.ID[(component, *index)])]
            if getattr(material, "poles", 0) or material.is_pec:
                if fluid:
                    raise ValueError("topology requires averaged bulk material IDs before conductor constraints")
            for lo, hi in fluid:
                if hi <= lo:
                    continue
                length = (hi-lo) * dl[component]
                number = len(e_indices)
                key = (component, *index)
                edge_map.setdefault(key, []).append(number)
                e_indices.append(key)
                e_intervals.append((lo, hi))
                me.append(epsilon0 * material.er * length * volume / dl[component])
                se.append(material.se * length * volume / dl[component])
    h_indices, areas, sides, mh, sm = [], [], [], [], []
    rows, cols, values, diagnostics, boundary_ports = [], [], [], [], []
    by_owner = {primitive.record.ordinal: primitive for primitive in compiler.scene.primitives}
    for component in range(3):
        ranges = [range(size[d]+1) if d == component else range(size[d]) for d in range(3)]
        for index in product(*ranges):
            vertices, edge_specs = _face_edges(component, index, dl)
            section = compiler.section(vertices)
            fluid = [piece for piece in section.pieces if piece.boundary_kind not in ("pec", "sibc")]
            groups = connected_pieces(fluid, tolerance)
            material = grid.materials[int(grid.ID[(component+3, *index)])]
            dual_length = dl[component] * (0.5 if index[component] in (0, size[component]) else 1)
            for sheet in section.sheets:
                # A finite sheet ending in a face needs internal transparent
                # edges in addition to its wall edges. Do not extend it to
                # the square boundary as an artificial opaque wall.
                u, v = (component+1) % 3, (component+2) % 3
                for endpoint in (sheet.start, sheet.stop):
                    on_rim = any(abs(endpoint[d] - vertices[0, d]) <= tolerance
                                 or abs(endpoint[d] - (vertices[0, d] + dl[d])) <= tolerance
                                 for d in (u, v))
                    if not on_rim:
                        raise ExpandedTopologyError(CutDiagnostic(component, index, "finite_sheet_tip"))
            for group in groups:
                area = sum(piece.area for piece in group)
                if area <= tolerance**2:
                    continue
                row = len(h_indices)
                h_indices.append((component, *index))
                areas.append(area)
                sides.append(group[0].sides)
                mh.append(mu0 * material.mr * area * dual_length)
                sm.append(material.sm * area * dual_length)
                face_normal = np.eye(3)[component]
                for piece in group:
                    for a, b in zip(piece.vertices, np.roll(piece.vertices, -1, axis=0)):
                        length = np.linalg.norm(b-a)
                        if length <= tolerance:
                            continue
                        middle = (a+b)/2
                        outward = np.cross((b-a)/length, face_normal)
                        wall = None
                        wall_normal = None
                        # A sheet has no occupied volume. Find its actual
                        # finite boundary segment, independently on each side.
                        for sheet in section.sheets:
                            direction = sheet.stop-sheet.start
                            span = np.linalg.norm(direction)
                            direction /= span
                            t = (middle-sheet.start) @ direction
                            if (np.linalg.norm(np.cross(middle-sheet.start, direction)) <= tolerance
                                    and -tolerance <= t <= span+tolerance):
                                wall = by_owner[sheet.owner]
                                wall_normal = sheet.normal
                        if wall is None:
                            owner = int(compiler.scene.owners(middle + 8*tolerance*outward))
                            wall = by_owner.get(owner)
                            if wall is None or wall.kind not in ("pec", "sibc"):
                                continue  # An artificial clipping subdivision.
                            normals, offsets = compiler.planes[owner]
                            touching = np.flatnonzero(np.abs(normals @ middle-offsets) <= 4*tolerance)
                            if not len(touching):
                                continue
                            wall_normal = normals[touching[0]]
                        if wall.kind == "sibc":
                            projected = np.sqrt(max(0, 1-wall_normal[component]**2))
                            if projected <= 1e-12:
                                diagnostics.append(CutDiagnostic(component, index, "singular_surface_projection"))
                            else:
                                boundary_ports.append(MagneticBoundaryPort(row, length*dual_length/projected,
                                    wall.material_id, wall.record.ordinal))
                incident = 0
                for edge_component, edge_index, sign in edge_specs:
                    edge_start = np.asarray(edge_index) * dl
                    perpendicular = [d for d in range(3) if d != edge_component]
                    intervals = []
                    for piece in group:
                        for a, b in zip(piece.vertices, np.roll(piece.vertices, -1, axis=0)):
                            if (np.max(np.abs(a[perpendicular] - edge_start[perpendicular])) <= tolerance
                                    and np.max(np.abs(b[perpendicular] - edge_start[perpendicular])) <= tolerance):
                                lo, hi = sorted(((a[edge_component] - edge_start[edge_component]) / dl[edge_component],
                                                 (b[edge_component] - edge_start[edge_component]) / dl[edge_component]))
                                if hi-lo > tolerance / dl[edge_component]:
                                    intervals.append((max(0, lo), min(1, hi)))
                    intervals = _union_intervals(intervals, tolerance / dl[edge_component])
                    for column in edge_map.get((edge_component, *edge_index), ()):
                        lo, hi = e_intervals[column]
                        covered = sum(max(0, min(hi, b) - max(lo, a)) for a, b in intervals)
                        if covered > tolerance / dl[edge_component]:
                            if abs(covered - (hi-lo)) > tolerance / dl[edge_component]:
                                raise ExpandedTopologyError(CutDiagnostic(component, index, "inconsistent_edge_partition"))
                            rows.append(row)
                            cols.append(column)
                            values.append(sign * covered * dl[edge_component] * dual_length)
                            incident += 1
                if (not incident and index[component] not in (0, size[component])
                        and area < dl[(component+1) % 3] * dl[(component+2) % 3] * (1-1e-10)):
                    diagnostics.append(CutDiagnostic(component, index, "isolated_subcell_region"))
    curl = coo_matrix((values, (rows, cols)), shape=(len(h_indices), len(e_indices))).tocsr()
    packing = YeePacking(CurlOperator(curl, me, mh), np.asarray(e_indices, dtype=np.int64),
                         np.asarray(h_indices, dtype=np.int64), np.asarray(se), np.asarray(sm))
    return CutTopology(packing, np.asarray(e_intervals), np.asarray(areas), tuple(sides),
                       compiler.scene.geometry.digest, tuple(diagnostics), tuple(boundary_ports))

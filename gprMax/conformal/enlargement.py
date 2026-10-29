"""Conservative, topology-aware enlargement at a fixed requested timestep."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

from gprMax.conformal.operator import CurlOperator


@dataclass(frozen=True)
class EnlargementPlan:
    operator: CurlOperator
    prolongation: csr_matrix
    groups: tuple
    accepted: bool
    reason: str | None
    original_operator: CurlOperator

    def packing(self, topology):
        """Transport bulk masses and losses to the accepted enlarged fields."""
        from gprMax.conformal.yee import YeePacking

        if not self.accepted:
            raise ValueError(f"rejected enlargement requires staircase rebuilding: {self.reason}")
        fine = topology.packing
        if fine.operator is not self.original_operator:
            raise ValueError("enlargement plan belongs to a different topology")
        return YeePacking(self.operator, fine.electric_indices, fine.magnetic_indices,
                          fine.electric_loss, np.asarray(self.prolongation.T @ fine.magnetic_loss).ravel(),
                          self.prolongation, fine.operator.magnetic_mass)

    def impedance_ports(self, topology, models, dt):
        """Keep every surface history while restricting its magnetic trace."""
        from gprMax.conformal.impedance import MagneticImpedancePorts

        packing = self.packing(topology)
        if not topology.boundary_ports:
            return None
        ports = topology.boundary_ports
        count = len(ports)
        trace = coo_matrix((np.ones(count), (np.arange(count),
                           [port.magnetic_row for port in ports])),
                           shape=(count, len(topology.packing.magnetic_indices))).tocsr()
        return MagneticImpedancePorts(self.operator, trace @ self.prolongation,
            [port.weight for port in ports], [models[port.model_id] for port in ports], dt,
            bulk_loss=packing.magnetic_loss)

    def restrict_magnetic(self, fine_field):
        """Mass-weighted restriction preserves total magnetic flux."""
        if not self.accepted:
            raise ValueError("cannot initialise a rejected enlargement")
        fine_field = np.asarray(fine_field)
        if fine_field.shape != self.original_operator.magnetic_mass.shape:
            raise ValueError("fine magnetic field has the wrong shape")
        return np.asarray(self.prolongation.T @ (
            self.original_operator.magnetic_mass*fine_field)).ravel() / self.operator.magnetic_mass


def plan_enlargement(topology, dt, *, max_group_size=4):
    """Try neighbouring open control volumes before requesting staircase.

    Adjacency requires opposite incidences of a shared electric DOF and the
    same magnetic component. Separate sheet sides therefore cannot be merged
    even when their dense Yee indices coincide. A returned rejected plan is
    a request to rebuild geometry; callers must not step its operator.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("requested timestep must be finite and positive")
    if isinstance(max_group_size, bool) or not isinstance(max_group_size, int) or max_group_size < 2:
        raise ValueError("max_group_size must be an integer of at least two")
    original = topology.packing.operator
    indices = topology.packing.magnetic_indices
    count = len(indices)
    identity = csr_matrix((np.ones(count), (np.arange(count), np.arange(count))), shape=(count, count))
    if topology.diagnostics:
        return EnlargementPlan(original, identity, tuple((i,) for i in range(count)), False,
                               topology.diagnostics[0].reason, original)
    if dt < original.cfl_limit:
        return EnlargementPlan(original, identity, tuple((i,) for i in range(count)), True, None, original)
    columns = original.curl.tocsc()
    adjacency = set()
    for column in range(columns.shape[1]):
        begin, end = columns.indptr[column:column+2]
        rows, signs = columns.indices[begin:end], columns.data[begin:end]
        for a in range(len(rows)):
            for b in range(a):
                i, j = int(rows[a]), int(rows[b])
                if indices[i, 0] == indices[j, 0] and signs[a]*signs[b] < 0:
                    adjacency.add(tuple(sorted((i, j))))
    if not adjacency:
        return EnlargementPlan(original, identity, tuple((i,) for i in range(count)), False,
                               "no_compatible_open_neighbour", original)
    graph_rows, graph_cols = zip(*adjacency)
    graph = coo_matrix((np.ones(len(adjacency)), (graph_rows, graph_cols)), shape=(count, count)).tocsr()
    _, labels = connected_components(graph, directed=False)
    groups = [set((i,)) for i in range(count)]
    last_operator, last_prolongation = original, identity
    # Local stiffness identifies small control volumes without moving any
    # surface. Re-evaluate after each conservative aggregation.
    for _ in range(count):
        scaled = abs(last_operator.curl)
        scores = np.asarray(scaled.sum(axis=1)).ravel() / np.sqrt(last_operator.magnetic_mass)
        order = np.argsort(-scores, kind="stable")
        owner = {row: group for group, members in enumerate(groups) for row in members}
        candidates = set()
        for a, b in adjacency:
            ga, gb = owner[a], owner[b]
            if ga != gb and len(groups[ga]) + len(groups[gb]) <= max_group_size:
                candidates.add(tuple(sorted((ga, gb))))
        if not candidates:
            break
        # Work on the most restrictive group that still has an admissible
        # open neighbour. Tie-break deterministically by group index.
        chosen = next((int(i) for i in order if any(i in pair for pair in candidates)), None)
        if chosen is None:
            break
        neighbours = sorted(pair for pair in candidates if chosen in pair)
        evaluated = []
        for first, second in neighbours:
            trial = [set(group) for group in groups]
            trial[first] |= trial[second]
            del trial[second]
            ordered = tuple(tuple(sorted(group)) for group in trial)
            aggregated = original.aggregate_magnetic(ordered, open_adjacency=adjacency, region_labels=labels)
            # At equal global CFL, prefer the larger local mass to remove
            # the small-cell stiffness before moving to the next group.
            merged_mass = sum(original.magnetic_mass[list(trial[first])])
            evaluated.append((aggregated.operator.cfl_limit, merged_mass, first, second, trial, aggregated))
        _, _, _, _, groups, best = max(evaluated, key=lambda item: item[:4])
        last_operator, last_prolongation = best.operator, best.prolongation
        if dt < last_operator.cfl_limit:
            return EnlargementPlan(last_operator, last_prolongation,
                                   tuple(tuple(sorted(group)) for group in groups), True, None, original)
    return EnlargementPlan(last_operator, last_prolongation,
                           tuple(tuple(sorted(group)) for group in groups), False,
                           "enlargement_cannot_certify_requested_timestep", original)

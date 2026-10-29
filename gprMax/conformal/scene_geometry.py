"""Continuous, ordered geometry retained before Yee rasterisation.

Both a time-domain cut-cell compiler and a port-section compiler consume
these records. Coordinates belong to the input's physical coordinate frame;
the records never inherit rounded voxel coordinates or synthetic materials.
"""

from dataclasses import dataclass
from types import MappingProxyType
import hashlib
import json

import numpy as np


def _freeze(value):
    if isinstance(value, np.ndarray):
        return tuple(_freeze(item) for item in value.tolist())
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _json_value(value):
    if hasattr(value, "items"):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Opaque custom commands remain provenance, not executable geometry.
    return str(value)


@dataclass(frozen=True)
class GeometryRecord:
    ordinal: int
    command: str
    parameters: object
    material_ids: tuple[str, ...]
    boundary_kind: str
    coordinate_frame: str

    def __reduce__(self):
        # Process-based runs must be able to transfer the detached records;
        # MappingProxyType itself has no pickle representation.
        return (_restore_record, (self.ordinal, self.command, _json_value(self.parameters),
                                 self.material_ids, self.boundary_kind, self.coordinate_frame))

    @property
    def identity(self):
        return f"{self.ordinal}:{self.command}"

    @property
    def is_sheet(self):
        return self.command == "#plate" or (
            self.command == "#triangle" and self.parameters.get("thickness") == 0)

    def serialise(self):
        return dict(ordinal=self.ordinal, command=self.command,
                    parameters=_json_value(self.parameters), material_ids=self.material_ids,
                    boundary_kind=self.boundary_kind, coordinate_frame=self.coordinate_frame)


def _restore_record(ordinal, command, parameters, material_ids, boundary_kind, coordinate_frame):
    return GeometryRecord(ordinal, command, _freeze(parameters), material_ids,
                          boundary_kind, coordinate_frame)


@dataclass(frozen=True)
class ContinuousGeometry:
    """Immutable ordered input records plus explicit compiler capabilities."""

    records: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "records", tuple(self.records))

    def __deepcopy__(self, memo):
        # MappingProxyType is intentionally immutable but not pickleable.
        # Grid copies can share this completely detached value snapshot.
        memo[id(self)] = self
        return self

    @classmethod
    def capture(cls, objects, grid):
        materials = {material.ID: material for material in getattr(grid, "materials", ())}
        surface_models = getattr(grid, "surface_impedance_models", {})
        records = []
        for ordinal, obj in enumerate(objects):
            parameters = _freeze(obj.kwargs)
            ids = ((parameters["material_id"],) if "material_id" in parameters
                   else tuple(parameters.get("material_ids", ())))
            if any(identifier in surface_models for identifier in ids):
                kind = "sibc"
            elif ids and all(getattr(materials.get(identifier), "is_pec", False) for identifier in ids):
                kind = "pec"
            elif any(getattr(materials.get(identifier), "is_pec", False) for identifier in ids):
                kind = "directional"
            else:
                kind = "bulk"
            frame = "input"
            if hasattr(grid, "local_to_global"):
                frame = "input-global" if getattr(obj, "autotranslate", True) else "input-local"
            records.append(GeometryRecord(ordinal, obj.hash, parameters, ids, kind, frame))
        return cls(records)

    def append(self, objects, grid):
        additions = self.capture(objects, grid)
        return ContinuousGeometry(self.records + tuple(
            GeometryRecord(len(self.records) + record.ordinal, record.command,
                           record.parameters, record.material_ids, record.boundary_kind,
                           record.coordinate_frame) for record in additions.records))

    @property
    def digest(self):
        payload = json.dumps([record.serialise() for record in self.records], sort_keys=True,
                             separators=(",", ":"), allow_nan=True).encode()
        return hashlib.sha256(payload).hexdigest()

    def compile(self, *, domain, dl):
        """Compile supported physical primitives, preserving unsupported records.

        An unsupported record is a geometry barrier. A caller must choose
        an explicit staircase patch or reject its affected region; it cannot
        skip the record and expose overwritten conductor surfaces.
        """
        return PrimitiveScene(self, domain=domain, dl=dl)


class UnsupportedGeometry(ValueError):
    pass


class ConvexPrimitive:
    """A convex volume or a finite opaque polygonal sheet.

    Inside satisfies ``normals @ x <= offsets``. Smooth primitives use their
    analytic intersections for segment queries; planar faces are exact.
    """

    def __init__(self, record, domain, dl):
        self.record = record
        self.kind = record.boundary_kind
        self.material_id = record.material_ids[0] if len(record.material_ids) == 1 else None
        self.sheet = record.is_sheet
        self.parameters = record.parameters
        self.domain = np.asarray(domain, dtype=float)
        self.dl = np.asarray(dl, dtype=float)
        self.normals = self.offsets = None
        p = record.parameters
        command = record.command
        if record.coordinate_frame != "input":
            raise UnsupportedGeometry("subgrid coordinate translation requires a local geometry compiler")
        if len(record.material_ids) != 1:
            raise UnsupportedGeometry("directional or implicit material ownership requires staircase geometry")
        if command in ("#box", "#plate"):
            self.lower = self._point(p["p1"], upper=False)
            self.upper = self._point(p["p2"], upper=True)
            if np.any(self.upper < self.lower):
                raise ValueError("continuous box bounds must be ordered")
            extent = self.upper - self.lower
            if self.sheet:
                axes = np.flatnonzero(extent == 0)
                if len(axes) != 1 or np.any(extent[np.arange(3) != axes[0]] <= 0):
                    raise ValueError("plate needs one zero extent and two positive extents")
                self.sheet_normal = np.eye(3)[axes[0]]
                self.sheet_offset = self.lower[axes[0]]
            elif np.any(extent <= 0):
                raise ValueError("continuous box requires positive extent")
            self.normals = np.vstack((np.eye(3), -np.eye(3)))
            self.offsets = np.concatenate((self.upper, -self.lower))
        elif command in ("#sphere", "#ellipsoid"):
            self.center = self._point(p["p1"])
            self.radii = (np.full(3, p["r"], dtype=float) if command == "#sphere"
                          else np.array([p[axis + "r"] for axis in "xyz"], dtype=float))
            if not np.all(np.isfinite(self.radii)) or np.any(self.radii <= 0):
                raise ValueError("ellipsoid radii must be finite and positive")
            self.lower, self.upper = self.center - self.radii, self.center + self.radii
        elif command == "#cylinder":
            self.start = self._point(p["p1"], upper=False)
            self.stop = self._point(p["p2"], upper=True)
            self.length = np.linalg.norm(self.stop - self.start)
            self.radius = float(p["r"])
            if self.length <= 0 or not np.isfinite(self.radius) or self.radius <= 0:
                raise ValueError("cylinder requires distinct endpoints and a positive radius")
            self.axis = (self.stop - self.start) / self.length
            radial = self.radius * np.sqrt(np.maximum(0, 1 - self.axis**2))
            self.lower = np.minimum(self.start, self.stop) - radial
            self.upper = np.maximum(self.start, self.stop) + radial
        elif command == "#triangle":
            vertices = np.array([p[name] for name in ("p1", "p2", "p3")], dtype=float)
            thickness = float(p["thickness"])
            for axis in range(3):
                if np.all(np.isinf(vertices[:, axis])):
                    vertices[:, axis] = 0
                    if np.isinf(thickness):
                        thickness = self.domain[axis]
            if not np.all(np.isfinite(vertices)) or not np.isfinite(thickness) or thickness < 0:
                raise ValueError("triangle coordinates and thickness must be finite")
            normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            length = np.linalg.norm(normal)
            if length <= 0:
                raise ValueError("continuous triangle vertices must not be collinear")
            normal /= length
            # Existing axis-aligned triangular prisms extend along the
            # positive normal axis regardless of vertex winding.
            if np.count_nonzero(np.abs(normal) > 1e-14) == 1:
                normal = np.abs(normal)
            self.sheet_normal = normal
            self.sheet_offset = float(normal @ vertices[0])
            centroid = vertices.mean(axis=0)
            normals, offsets = [], []
            for i in range(3):
                outward = np.cross(vertices[(i + 1) % 3] - vertices[i], normal)
                outward /= np.linalg.norm(outward)
                if outward @ (centroid - vertices[i]) > 0:
                    outward = -outward
                normals.append(outward)
                offsets.append(outward @ vertices[i])
            normals.extend((normal, -normal))
            offsets.extend((self.sheet_offset + thickness, -self.sheet_offset))
            self.normals, self.offsets = np.asarray(normals), np.asarray(offsets)
            all_vertices = np.vstack((vertices, vertices + normal * thickness))
            self.lower, self.upper = all_vertices.min(axis=0), all_vertices.max(axis=0)
        else:
            raise UnsupportedGeometry(f"{command} has no continuous intersection compiler")

    def _point(self, values, upper=None):
        point = np.array(values, dtype=float)
        if point.shape != (3,) or np.any(np.isnan(point)):
            raise ValueError("continuous coordinates require a three-vector")
        if np.any(np.isinf(point)):
            if upper is None:
                raise UnsupportedGeometry("unbounded point requires an extent role")
            point[np.isinf(point)] = self.domain[np.isinf(point)] if upper else 0
        return point

    def contains(self, points, tolerance=0.0):
        points = np.asarray(points, dtype=float)
        if points.shape[-1:] != (3,):
            raise ValueError("points must end in a three-coordinate axis")
        if self.normals is not None:
            return np.all(points @ self.normals.T <= self.offsets + tolerance, axis=-1)
        if self.record.command in ("#sphere", "#ellipsoid"):
            return np.sum(((points - self.center) / self.radii)**2, axis=-1) <= 1 + tolerance
        relative = points - self.start
        axial = relative @ self.axis
        radial = relative - axial[..., None] * self.axis
        return ((axial >= -tolerance) & (axial <= self.length + tolerance)
                & (np.sum(radial**2, axis=-1) <= (self.radius + tolerance)**2))

    def segment_interval(self, start, stop):
        """Exact material interval in segment parameter t in [0,1].

        A sheet returns a repeated endpoint at its intersection. A segment
        within a sheet may return a finite interval; it belongs to the
        boundary and must not be interpreted as a penetrable volume.
        """
        start, stop = np.asarray(start, dtype=float), np.asarray(stop, dtype=float)
        if start.shape != (3,) or stop.shape != (3,) or not np.all(np.isfinite([start, stop])):
            raise ValueError("segment endpoints must be finite three-vectors")
        direction = stop - start
        if np.linalg.norm(direction) == 0:
            raise ValueError("segment endpoints must differ")
        lo, hi = 0.0, 1.0
        if self.normals is not None:
            for normal, offset in zip(self.normals, self.offsets):
                d = float(normal @ direction)
                value = float(offset - normal @ start)
                roundoff = 32*np.finfo(float).eps * max(abs(offset), np.linalg.norm(start), np.linalg.norm(stop))
                if abs(d) <= 32*np.finfo(float).eps * np.linalg.norm(direction):
                    if value < -roundoff:
                        return None
                elif d > 0:
                    hi = min(hi, value / d)
                else:
                    lo = max(lo, value / d)
                if lo > hi + 64*np.finfo(float).eps:
                    return None
            if lo > hi:
                lo = hi = (lo+hi)/2
            return lo, hi
        if self.record.command in ("#sphere", "#ellipsoid"):
            relative, direction = (start - self.center) / self.radii, direction / self.radii
            radius2 = 1.0
        else:
            relative = start - self.start
            axial, delta = float(relative @ self.axis), float(direction @ self.axis)
            if delta == 0:
                if axial < 0 or axial > self.length:
                    return None
            else:
                ends = sorted((-axial / delta, (self.length - axial) / delta))
                lo, hi = max(lo, ends[0]), min(hi, ends[1])
            relative = relative - axial * self.axis
            direction = direction - delta * self.axis
            radius2 = self.radius**2
        a = float(direction @ direction)
        b = float(2 * relative @ direction)
        c = float(relative @ relative - radius2)
        if a == 0:
            return (lo, hi) if c <= 0 and lo <= hi else None
        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return None
        root = np.sqrt(discriminant)
        # Stable quadratic roots preserve very short material intervals.
        q = -0.5 * (b + np.copysign(root, b))
        roots = (-b / (2*a),) * 2 if q == 0 else sorted((q / a, c / q))
        lo, hi = max(lo, roots[0]), min(hi, roots[1])
        return (lo, hi) if lo <= hi else None


@dataclass(frozen=True)
class SegmentRegion:
    start: float
    stop: float
    owner: int | None
    material_id: str
    boundary_kind: str


class PrimitiveScene:
    def __init__(self, geometry, *, domain, dl):
        self.geometry = geometry
        self.primitives, self.unsupported = [], []
        for record in geometry.records:
            try:
                self.primitives.append(ConvexPrimitive(record, domain, dl))
            except UnsupportedGeometry as exc:
                self.unsupported.append((record, str(exc)))
        self.primitives = tuple(self.primitives)
        self.unsupported = tuple(self.unsupported)

    def _require_complete(self):
        if self.unsupported:
            identities = ", ".join(record.identity for record, _ in self.unsupported)
            raise UnsupportedGeometry(f"continuous ownership is unresolved for {identities}")

    def owners(self, points):
        """Last volume wins. Sheets are separate boundaries, not painted bulk."""
        self._require_complete()
        points = np.asarray(points, dtype=float)
        result = np.full(points.shape[:-1], -1, dtype=np.int64)
        for primitive in self.primitives:
            if not primitive.sheet:
                result = np.where(primitive.contains(points), primitive.record.ordinal, result)
        return result

    def partition_segment(self, start, stop):
        """Return ordered volume intervals and surviving opaque sheet crossings."""
        self._require_complete()
        intervals = [(primitive, primitive.segment_interval(start, stop)) for primitive in self.primitives]
        cuts = {0.0, 1.0}
        for _, interval in intervals:
            if interval is not None:
                cuts.update(interval)
        cuts = sorted(cuts)
        regions = []
        for left, right in zip(cuts, cuts[1:]):
            if right <= left:
                continue
            middle = (left + right) / 2
            owner = None
            for primitive, interval in intervals:
                if not primitive.sheet and interval is not None and interval[0] <= middle <= interval[1]:
                    owner = primitive
            regions.append(SegmentRegion(left, right,
                None if owner is None else owner.record.ordinal,
                "free_space" if owner is None else owner.material_id,
                "bulk" if owner is None else owner.kind))
        sheets = []
        for primitive, interval in intervals:
            if not primitive.sheet or interval is None:
                continue
            # A later volume paints over a boundary where it actually covers
            # it. Preserve the exact crossing otherwise.
            surviving = [interval]
            for other, cover in intervals:
                if other.record.ordinal <= primitive.record.ordinal or other.sheet or cover is None:
                    continue
                pieces = []
                for left, right in surviving:
                    if cover[1] < left or cover[0] > right:
                        pieces.append((left, right))
                    else:
                        if left < cover[0]:
                            pieces.append((left, min(right, cover[0])))
                        if right > cover[1]:
                            pieces.append((max(left, cover[1]), right))
                surviving = pieces
            sheets.extend((primitive.record.ordinal, interval) for interval in surviving)
        return tuple(regions), tuple(sheets)

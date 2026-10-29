"""Oriented convex cut cells with opaque, two-sided boundary sheets.

Coordinates are physical coordinates, never voxel occupancy or rounded Yee
indices. A plane must span the cell: finite patches and curved surfaces need
their own intersection compiler. Coincident surfaces are not inferred from
material IDs. Separate sheet sides retain separate boundary ownership even
when their coordinates are identical.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np


def _readonly(values):
    values = np.array(values, dtype=np.float64, copy=True)
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


@dataclass(frozen=True, eq=False)
class Boundary:
    """An opaque PEC or SIBC boundary, with a stable geometric identity.

    ``normal`` points from the negative side to the positive side. For a
    volume, ``keep`` selects the exterior. For a sheet both sides survive;
    there is no through-sheet transmission term.
    """

    identifier: str
    normal: np.ndarray
    offset: float
    kind: str = "pec"
    keep: str = "positive"
    impedance_id: str | None = None

    def __post_init__(self):
        normal = np.asarray(self.normal, dtype=float)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise ValueError("boundary normal must be a finite three-vector")
        length = float(np.linalg.norm(normal))
        if not np.isfinite(length) or length <= 0 or not np.isfinite(self.offset):
            raise ValueError("boundary requires a nonzero normal and finite offset")
        if not isinstance(self.identifier, str) or not self.identifier:
            raise ValueError("boundary requires a nonempty geometric identifier")
        if self.kind not in ("pec", "sibc"):
            raise ValueError("boundary kind must be pec or sibc")
        if self.keep not in ("positive", "negative", "both"):
            raise ValueError("keep must be positive, negative, or both")
        if self.kind == "sibc" and (not isinstance(self.impedance_id, str) or not self.impedance_id):
            raise ValueError("SIBC requires an impedance model identifier")
        if self.kind == "pec" and self.impedance_id is not None:
            raise ValueError("PEC cannot reference an impedance model")
        object.__setattr__(self, "normal", _readonly(normal / length))
        object.__setattr__(self, "offset", float(self.offset) / length)


@dataclass(frozen=True)
class Wall:
    boundary_id: str
    side: str
    kind: str
    impedance_id: str | None


@dataclass(frozen=True, eq=False)
class Face:
    """Counterclockwise vertices as seen from outside the retained volume."""

    vertices: np.ndarray
    wall: Wall | None = None

    def __post_init__(self):
        vertices = np.asarray(self.vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
            raise ValueError("a face needs at least three three-dimensional vertices")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("face coordinates must be finite")
        object.__setattr__(self, "vertices", _readonly(vertices))

    @property
    def area_vector(self):
        # Subtracting an origin avoids cancellation for translated models.
        vertices = self.vertices - self.vertices[0]
        return 0.5 * np.sum(np.cross(vertices, np.roll(vertices, -1, axis=0)), axis=0)

    @property
    def area(self):
        return float(np.linalg.norm(self.area_vector))


@dataclass(frozen=True)
class Region:
    faces: tuple[Face, ...]
    # Each sheet cut appends one side. This is a topological identity, not a
    # coordinate hash: opposing sheet fields must not be deduplicated.
    sides: tuple[tuple[str, str], ...] = ()

    @property
    def volume(self):
        if not self.faces:
            return 0.0
        origin = self.faces[0].vertices[0]
        return float(sum(np.dot(face.vertices[0] - origin, face.area_vector)
                         for face in self.faces) / 3.0)

    @property
    def closure_error(self):
        return np.sum([face.area_vector for face in self.faces], axis=0)


class UnresolvedGeometry(ValueError):
    """A cut that must be reported rather than silently changing topology."""


def box_region(lower, upper):
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if (lower.shape != (3,) or upper.shape != (3,)
            or not np.all(np.isfinite([lower, upper])) or np.any(upper <= lower)):
        raise ValueError("cell bounds must be finite with positive extent on every axis")
    faces = []
    for axis, side in product(range(3), (0, 1)):
        # Cyclic tangents have cross product in the positive axis direction.
        u, v = (axis + 1) % 3, (axis + 2) % 3
        vertices = []
        for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
            point = lower.copy()
            point[axis] = upper[axis] if side else lower[axis]
            point[u] = upper[u] if a else lower[u]
            point[v] = upper[v] if b else lower[v]
            vertices.append(point)
        if not side:
            vertices.reverse()
        faces.append(Face(vertices))
    return Region(tuple(faces))


def _unique(points, tolerance):
    unique = []
    for point in points:
        if not any(np.linalg.norm(point - other) <= tolerance for other in unique):
            unique.append(point)
    return unique


def _clip(region, boundary, side, tolerance):
    # Clip against signed distance <= 0; cap normal points out of the volume.
    sign = 1 if side == "negative" else -1
    normal, offset = sign * boundary.normal, sign * boundary.offset
    all_vertices = np.concatenate([face.vertices for face in region.faces])
    distance = all_vertices @ normal - offset
    if np.max(distance) <= tolerance:
        # A tangent cut still assigns a boundary to an existing cell face.
        faces = []
        for face in region.faces:
            if np.max(np.abs(face.vertices @ normal - offset)) <= tolerance:
                if face.wall is not None:
                    raise UnresolvedGeometry("coincident boundaries require an ownership decision")
                face = Face(face.vertices, Wall(boundary.identifier, side,
                                               boundary.kind, boundary.impedance_id))
            faces.append(face)
        return Region(tuple(faces), region.sides)
    if np.min(distance) >= -tolerance:
        return None
    faces, cap = [], []
    for face in region.faces:
        vertices = face.vertices
        distances = vertices @ normal - offset
        # Snap only within the explicitly supplied geometry tolerance.
        distances[np.abs(distances) <= tolerance] = 0.0
        polygon = []
        for index, start in enumerate(vertices):
            end = vertices[(index + 1) % len(vertices)]
            d0, d1 = distances[index], distances[(index + 1) % len(vertices)]
            if d0 <= 0:
                polygon.append(start)
                if d0 == 0:
                    cap.append(start)
            if (d0 < 0 < d1) or (d1 < 0 < d0):
                point = start + (d0 / (d0 - d1)) * (end - start)
                polygon.append(point)
                cap.append(point)
        polygon = _unique(polygon, tolerance)
        if len(polygon) >= 3:
            clipped = Face(polygon, face.wall)
            if clipped.area > tolerance**2:
                faces.append(clipped)
    cap = _unique(cap, tolerance)
    if len(cap) < 3:
        raise UnresolvedGeometry("plane intersection did not produce a closed cap")
    center = np.mean(cap, axis=0)
    tangent = np.eye(3)[np.argmin(np.abs(normal))]
    tangent = tangent - np.dot(tangent, normal) * normal
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    cap = np.asarray(cap)
    relative = cap - center
    angles = np.arctan2(relative @ bitangent, relative @ tangent)
    faces.append(Face(cap[np.argsort(angles)], Wall(boundary.identifier, side,
                                                   boundary.kind, boundary.impedance_id)))
    result = Region(tuple(faces), region.sides)
    if result.volume <= tolerance**3:
        raise UnresolvedGeometry("retained volume is below the geometry tolerance")
    return result


def cut_cell(lower, upper, boundaries, *, tolerance=None):
    """Cut a cell by full-span opaque planes, returning disconnected regions.

    A small retained region remains explicit. Stabilisation is a subsequent
    operation and must not move this boundary or turn a sheet into a volume.
    Boundaries describe an intersection of exteriors, not last-object-wins
    scene painting. A scene compiler must resolve Boolean ownership first.
    """
    region = box_region(lower, upper)
    scale = float(np.max(np.asarray(upper) - np.asarray(lower)))
    tolerance = scale * 1e-12 if tolerance is None else float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0 or tolerance >= scale:
        raise ValueError("geometry tolerance must be positive and smaller than the cell")
    regions = [region]
    identifiers = set()
    for boundary in boundaries:
        if boundary.identifier in identifiers:
            raise ValueError("boundary identifiers must be unique within a cut cell")
        identifiers.add(boundary.identifier)
        sides = ("negative", "positive") if boundary.keep == "both" else (boundary.keep,)
        result = []
        for region in regions:
            for side in sides:
                clipped = _clip(region, boundary, side, tolerance)
                if clipped is not None:
                    lineage = clipped.sides
                    if boundary.keep == "both":
                        lineage += ((boundary.identifier, side),)
                    result.append(Region(clipped.faces, lineage))
        regions = result
    return tuple(regions)

"""Shared physical cross-sections for conformal cells and modal apertures.

Dielectric pieces retain their material ownership; this module does not
replace the existing dielectric averaging rule. Curved surfaces are faceted
once per primitive, then the same planes are used by all intersecting faces.
"""

from dataclasses import dataclass
from copy import copy

import numpy as np
from scipy.spatial import ConvexHull

from gprMax.conformal.scene_geometry import UnsupportedGeometry


def _area(vertices):
    if len(vertices) < 3:
        return 0.0
    relative = vertices - vertices[0]
    return float(np.linalg.norm(np.sum(np.cross(relative, np.roll(relative, -1, axis=0)), axis=0)) / 2)


def _split(vertices, normal, offset, tolerance):
    """Split a convex polygon; preserve orientation on both sides."""
    distances = vertices @ normal - offset
    distances[np.abs(distances) <= tolerance] = 0
    if np.all(distances <= 0):
        return vertices, None
    if np.all(distances >= 0):
        return None, vertices
    negative, positive = [], []
    for i, start in enumerate(vertices):
        end = vertices[(i + 1) % len(vertices)]
        a, b = distances[i], distances[(i + 1) % len(vertices)]
        if a <= 0:
            negative.append(start)
        if a >= 0:
            positive.append(start)
        if a * b < 0:
            point = start + a / (a - b) * (end - start)
            negative.append(point)
            positive.append(point)
    return np.asarray(negative), np.asarray(positive)


def _subtract(vertices, normals, offsets, tolerance):
    outside = []
    inside = vertices
    for normal, offset in zip(normals, offsets):
        if inside is None:
            break
        inside, piece = _split(inside, normal, offset, tolerance)
        if piece is not None and _area(piece) > tolerance**2:
            outside.append(piece)
    return outside, inside


@dataclass(frozen=True, eq=False)
class SectionPiece:
    vertices: np.ndarray
    owner: int | None
    material_id: str
    boundary_kind: str
    sides: tuple = ()

    @property
    def area(self):
        return _area(self.vertices)


@dataclass(frozen=True, eq=False)
class SheetIntersection:
    owner: int
    start: np.ndarray
    stop: np.ndarray
    normal: np.ndarray
    material_id: str
    boundary_kind: str


@dataclass(frozen=True)
class Section:
    pieces: tuple[SectionPiece, ...]
    sheets: tuple[SheetIntersection, ...]
    geometric_digest: str

    @property
    def retained_area(self):
        return sum(piece.area for piece in self.pieces if piece.boundary_kind not in ("pec", "sibc"))

    @property
    def total_area(self):
        return sum(piece.area for piece in self.pieces)


def _unit_sphere(error):
    golden = (1 + np.sqrt(5)) / 2
    points = []
    for a in (-1, 1):
        for b in (-golden, golden):
            points.extend(((0, a, b), (a, b, 0), (b, 0, a)))
    vertices = np.asarray(points, dtype=float)
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    faces = ConvexHull(vertices).simplices
    for _ in range(8):
        # Every face is inscribed. Distance of its plane from the origin
        # certifies the radial chord error over that face.
        hull = ConvexHull(vertices)
        if np.max(1 + hull.equations[:, 3]) <= error:
            return vertices
        edge_map = {}
        new_vertices = list(vertices)
        new_faces = []
        for a, b, c in faces:
            mids = []
            for start, stop in ((a, b), (b, c), (c, a)):
                key = tuple(sorted((int(start), int(stop))))
                if key not in edge_map:
                    point = vertices[start] + vertices[stop]
                    point /= np.linalg.norm(point)
                    edge_map[key] = len(new_vertices)
                    new_vertices.append(point)
                mids.append(edge_map[key])
            ab, bc, ca = mids
            new_faces.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
        vertices, faces = np.asarray(new_vertices), np.asarray(new_faces)
    raise UnsupportedGeometry("surface faceting tolerance exceeds the bounded geometry compiler")


class SectionCompiler:
    """Compile every section from one ordered continuous scene.

    ``surface_error`` is an absolute geometric chord-error budget in metres.
    No field solver tolerance or stability threshold is hidden in it.
    """

    def __init__(self, scene, *, surface_error):
        scene._require_complete()
        if not np.isfinite(surface_error) or surface_error <= 0:
            raise ValueError("surface_error must be finite and positive")
        self.scene = scene
        self.surface_error = float(surface_error)
        self.planes = {}
        for primitive in scene.primitives:
            if primitive.normals is not None:
                planes = primitive.normals, primitive.offsets
            elif primitive.record.command in ("#sphere", "#ellipsoid"):
                points = _unit_sphere(surface_error / np.max(primitive.radii))
                hull = ConvexHull(primitive.center + points * primitive.radii)
                planes = hull.equations[:, :3], -hull.equations[:, 3]
            else:
                error = min(surface_error / primitive.radius, 1)
                count = max(8, int(np.ceil(np.pi / np.arccos(1 - error))))
                if count > 65536:
                    raise UnsupportedGeometry("cylinder surface tolerance exceeds the geometry compiler")
                axis = primitive.axis
                tangent = np.eye(3)[np.argmin(np.abs(axis))]
                tangent -= (tangent @ axis) * axis
                tangent /= np.linalg.norm(tangent)
                bitangent = np.cross(axis, tangent)
                angles = (np.arange(count) + 0.5) * (2*np.pi / count)
                normals = np.cos(angles)[:, None] * tangent + np.sin(angles)[:, None] * bitangent
                offsets = normals @ primitive.start + primitive.radius * np.cos(np.pi / count)
                planes = (np.vstack((normals, axis, -axis)),
                          np.concatenate((offsets, [axis @ primitive.stop, -axis @ primitive.start])))
            self.planes[primitive.record.ordinal] = planes
        # Edge and face compilers must use identical faceted surfaces. Keep
        # the original analytic scene intact for provenance and error checks.
        self.scene = copy(scene)
        primitives = []
        for primitive in scene.primitives:
            retained = copy(primitive)
            retained.normals, retained.offsets = self.planes[primitive.record.ordinal]
            primitives.append(retained)
        self.scene.primitives = tuple(primitives)

    def section(self, vertices):
        vertices = np.asarray(vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3 or not np.all(np.isfinite(vertices)):
            raise ValueError("section requires finite three-dimensional polygon vertices")
        scale = float(np.max(np.ptp(vertices, axis=0)))
        tolerance = max(scale * 1e-12, np.max(np.abs(vertices)) * np.finfo(float).eps * 16)
        face_normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        norm = np.linalg.norm(face_normal)
        if norm == 0:
            raise ValueError("section polygon is degenerate")
        face_normal /= norm
        if np.max(np.abs((vertices - vertices[0]) @ face_normal)) > tolerance:
            raise ValueError("section vertices must be coplanar")
        # Only convex input polygons are supported; the compiler can create
        # a disconnected/nonconvex result as a set of convex pieces.
        for i in range(len(vertices)):
            outward = np.cross(vertices[(i+1) % len(vertices)] - vertices[i], face_normal)
            if np.max((vertices - vertices[i]) @ outward) > tolerance * np.linalg.norm(outward):
                raise ValueError("input section polygon must be convex and consistently oriented")
        pieces = [SectionPiece(vertices.copy(), None, "free_space", "bulk")]
        sheets = []
        for primitive in self.scene.primitives:
            normals, offsets = self.planes[primitive.record.ordinal]
            if primitive.sheet:
                # The sheet has zero measure. Find its finite intersection
                # with this section independently of volume painting.
                normal, offset = primitive.sheet_normal, primitive.sheet_offset
                distances = vertices @ normal - offset
                if np.all(np.abs(distances) <= tolerance):
                    # A coplanar sheet is explicitly represented by occupied
                    # section pieces. Adjacent sections own its two sides.
                    updated = []
                    for piece in pieces:
                        outside, inside = _subtract(piece.vertices, normals, offsets, tolerance)
                        updated.extend(SectionPiece(poly, piece.owner, piece.material_id,
                                                    piece.boundary_kind, piece.sides) for poly in outside)
                        if inside is not None:
                            updated.append(SectionPiece(inside, primitive.record.ordinal,
                                                        primitive.material_id, primitive.kind, piece.sides))
                    pieces = updated
                    continue
                intersections = []
                for i, start in enumerate(vertices):
                    a, b = distances[i], distances[(i+1) % len(vertices)]
                    if abs(a) <= tolerance:
                        intersections.append(start)
                    elif a * b < 0:
                        intersections.append(start + a / (a-b) * (vertices[(i+1) % len(vertices)] - start))
                if len(intersections) < 2:
                    continue
                direction = np.cross(face_normal, normal)
                if np.linalg.norm(direction) == 0:
                    continue
                intersections = np.asarray(intersections)
                order = np.argsort(intersections @ direction)
                start, stop = intersections[order[0]], intersections[order[-1]]
                if np.linalg.norm(stop - start) <= tolerance:
                    continue
                interval = primitive.segment_interval(start, stop)
                if interval is None or interval[1] - interval[0] <= tolerance / scale:
                    continue
                begin = start + interval[0] * (stop - start)
                end = start + interval[1] * (stop - start)
                sheets.append(SheetIntersection(primitive.record.ordinal, begin, end, normal,
                                                primitive.material_id, primitive.kind))
                updated = []
                for piece in pieces:
                    if piece.boundary_kind in ("pec", "sibc"):
                        updated.append(piece)
                        continue
                    neg, pos = _split(piece.vertices, normal, offset, tolerance)
                    for poly, side in ((neg, "negative"), (pos, "positive")):
                        if poly is not None:
                            updated.append(SectionPiece(poly, piece.owner, piece.material_id,
                                                        piece.boundary_kind,
                                                        piece.sides + ((primitive.record.ordinal, side),)))
                pieces = updated
                continue
            # Discard planes that contain the entire section on their inner
            # side, and skip a primitive separated by any of its planes.
            distances = vertices @ normals.T - offsets
            if np.any(np.all(distances > tolerance, axis=0)):
                continue
            active = np.any(distances > tolerance, axis=0)
            updated = []
            for piece in pieces:
                outside, inside = _subtract(piece.vertices, normals[active], offsets[active], tolerance)
                updated.extend(SectionPiece(poly, piece.owner, piece.material_id, piece.boundary_kind,
                                            piece.sides) for poly in outside)
                if inside is not None and _area(inside) > tolerance**2:
                    # A later volume replaces both the material and earlier
                    # sheet-side lineage within its actual occupied region.
                    updated.append(SectionPiece(inside, primitive.record.ordinal, primitive.material_id,
                                                primitive.kind))
            pieces = updated
            # Remove portions of earlier sheets overwritten by this volume.
            kept = []
            for sheet in sheets:
                interval = primitive.segment_interval(sheet.start, sheet.stop)
                if interval is None:
                    kept.append(sheet)
                    continue
                for lo, hi in ((0, interval[0]), (interval[1], 1)):
                    if hi > lo:
                        delta = sheet.stop - sheet.start
                        kept.append(SheetIntersection(sheet.owner, sheet.start + lo*delta,
                            sheet.start + hi*delta, sheet.normal, sheet.material_id, sheet.boundary_kind))
            sheets = kept
        return Section(tuple(pieces), tuple(sheets), self.scene.geometry.digest)

"""OpenCascade feature regeneration and mesh generation, used only by workers."""

from __future__ import annotations

import math
import numpy as np

from .project import physical_features


def regenerate(project):
    from OCC.Core.BRepPrimAPI import (
        BRepPrimAPI_MakeBox,
        BRepPrimAPI_MakeCylinder,
        BRepPrimAPI_MakePrism,
        BRepPrimAPI_MakeRevol,
    )
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_Transform,
    )
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut, BRepAlgoAPI_Common
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.GC import GC_MakeArcOfCircle
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
    from OCC.Core.GeomAbs import GeomAbs_Plane
    from OCC.Core.TopAbs import TopAbs_REVERSED
    from OCC.Extend.TopologyUtils import TopologyExplorer
    from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Ax1, gp_Ax2, gp_Circ, gp_Trsf
    from toolboxes.STEPtoVoxel.step_parser import ParserConfig, tessellate_shape, shape_for_ops
    from toolboxes.STEPtoVoxel import inspect_step

    shapes, imported = {}, {}
    mesh_spacing = min(project["simulation"]["spacing"])
    cfg = ParserConfig(linear_deflection=mesh_spacing * 0.2, is_relative_deflection=False, angular_deflection=0.15)
    needed = {f["id"] for f in project["features"] if f["enabled"]}
    from .project import dependencies

    by_id = {f["id"]: f for f in project["features"]}

    def include(identity):
        for source in dependencies(by_id[identity]):
            if source not in needed:
                needed.add(source)
                include(source)

    for identity in list(needed):
        include(identity)

    def point(x):
        return gp_Pnt(*map(float, x))

    for feature in project["features"]:
        if feature["id"] not in needed:
            continue
        kind, p = feature["kind"], feature["parameters"]
        try:
            shape = None
            if kind == "box":
                if min(abs(np.array(p["size"]))) == 0:
                    raise ValueError("Box dimensions must be nonzero")
                lower = np.array(p["origin"]) + np.minimum(p["size"], 0)
                shape = BRepPrimAPI_MakeBox(point(lower), *abs(np.array(p["size"]))).Shape()
            elif kind == "cylinder":
                if p["radius"] <= 0 or p["height"] == 0:
                    raise ValueError("Cylinder radius must be positive and height nonzero")
                shape = BRepPrimAPI_MakeCylinder(
                    gp_Ax2(point(p["origin"]), gp_Dir(*(np.array(p["axis"]) * np.sign(p["height"])))),
                    p["radius"],
                    abs(p["height"]),
                ).Shape()
            elif kind == "sketch":
                u, v, origin = np.array(p["u"]), np.array(p["v"]), np.array(p["origin"])
                if not np.isclose(np.linalg.norm(u), 1) or not np.isclose(np.linalg.norm(v), 1) or abs(u @ v) > 1e-8:
                    raise ValueError("Sketch axes must be orthonormal")

                def planar(q):
                    return point(origin + q[0] * u + q[1] * v)

                if p["profile"] == "circle":
                    if p["radius"] <= 0:
                        raise ValueError("Circle radius must be positive")
                    edge = BRepBuilderAPI_MakeEdge(
                        gp_Circ(gp_Ax2(point(origin), gp_Dir(*np.cross(u, v))), p["radius"])
                    ).Edge()
                    wire = BRepBuilderAPI_MakeWire(edge).Wire()
                elif p["profile"] == "segments":
                    builder = BRepBuilderAPI_MakeWire()
                    for segment in p["segments"]:
                        pts = [planar(q) for q in segment]
                        edge = (
                            BRepBuilderAPI_MakeEdge(pts[0], pts[-1])
                            if len(pts) == 2
                            else BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(*pts).Value())
                        )
                        builder.Add(edge.Edge())
                    wire = builder.Wire()
                else:
                    pts = p.get("points", [])
                    if p["profile"] == "rectangle":
                        w, h = p["width"], p["height"]
                        if w == 0 or h == 0:
                            raise ValueError("Profile dimensions must be nonzero")
                        pts = [[0, 0], [w, 0], [w, h], [0, h]]
                    builder = BRepBuilderAPI_MakePolygon()
                    for q in pts:
                        builder.Add(planar(q))
                    builder.Close()
                    wire = builder.Wire()
                shape = BRepBuilderAPI_MakeFace(wire).Shape()
            elif kind == "face_extrude":
                faces = list(TopologyExplorer(shapes[p["source"]]).faces())
                if p["face_index"] >= len(faces):
                    raise ValueError("Selected face no longer exists; select the face again")
                face = faces[p["face_index"]]
                surface = BRepAdaptor_Surface(face)
                if surface.GetType() != GeomAbs_Plane:
                    raise ValueError("Face extrusion requires a planar CAD face")
                normal = np.array(surface.Plane().Axis().Direction().Coord())
                if face.Orientation() == TopAbs_REVERSED:
                    normal *= -1
                if not np.allclose(normal, p["normal"], atol=1e-6):
                    raise ValueError("Selected face orientation changed; select the face again")
                shape = (
                    face
                    if p["thickness"] == 0
                    else BRepPrimAPI_MakePrism(face, gp_Vec(*(normal * p["thickness"]))).Shape()
                )
            elif kind == "extrude":
                if np.linalg.norm(p["vector"]) == 0:
                    raise ValueError("Extrusion vector must be nonzero")
                shape = BRepPrimAPI_MakePrism(shapes[p["source"]], gp_Vec(*p["vector"])).Shape()
            elif kind == "revolve":
                if not 0 < abs(p["angle"]) <= 360:
                    raise ValueError("Revolution angle must lie in (0, 360] degrees")
                shape = BRepPrimAPI_MakeRevol(
                    shapes[p["source"]], gp_Ax1(point(p["origin"]), gp_Dir(*p["axis"])), math.radians(p["angle"])
                ).Shape()
            elif kind in ("union", "subtract", "intersect"):
                operation = {"union": BRepAlgoAPI_Fuse, "subtract": BRepAlgoAPI_Cut, "intersect": BRepAlgoAPI_Common}[
                    kind
                ](shapes[p["target"]], shapes[p["tool"]])
                operation.Build()
                if not operation.IsDone():
                    raise ValueError("Boolean operation failed")
                shape = operation.Shape()
            elif kind == "transform":
                tr = gp_Trsf()
                tr.SetRotation(gp_Ax1(point(p["origin"]), gp_Dir(*p["axis"])), math.radians(p["angle"]))
                tr.SetTranslationPart(gp_Vec(*p["translation"]))
                shape = BRepBuilderAPI_Transform(shapes[p["source"]], tr, True).Shape()
            elif kind == "step":
                asset = p["asset"]
                if asset not in imported:
                    imported[asset] = inspect_step(asset)
                shape = shape_for_ops(imported[asset][p["part_index"]], cfg)
            elif kind not in ("stl", "sheet"):
                raise ValueError(f"Unknown feature type {kind}")
            if shape is not None:
                if feature.get("frame"):
                    frame = feature["frame"]
                    u, v = np.array(frame["u"]), np.array(frame["v"])
                    matrix = np.column_stack((u, v, np.cross(u, v), frame["origin"]))
                    tr = gp_Trsf()
                    tr.SetValues(*matrix.ravel().tolist())
                    shape = BRepBuilderAPI_Transform(shape, tr, True).Shape()
                if shape.IsNull() or not BRepCheck_Analyzer(shape).IsValid():
                    raise ValueError("Operation did not produce valid geometry")
                shapes[feature["id"]] = shape
        except Exception as exc:
            raise ValueError(f"{feature['name']}: {exc}") from exc

    meshes = []
    from OCC.Extend.TopologyUtils import TopologyExplorer

    for feature in physical_features(project):
        if feature["kind"] == "sheet":
            continue
        if feature["kind"] == "stl":
            import pyvista as pv

            surface = pv.read(feature["parameters"]["asset"]).extract_surface().triangulate().clean()
            if surface.n_open_edges:
                raise ValueError(f"{feature['name']}: STL must be a closed watertight surface")
            from scipy.spatial.transform import Rotation

            p = feature["parameters"]
            rotation = Rotation.from_euler("xyz", p.get("rotation", [0, 0, 0]), degrees=True).as_matrix()
            verts = (np.asarray(surface.points) * p["scale"]) @ rotation.T + p.get("translation", [0, 0, 0])
            tris = surface.faces.reshape(-1, 4)[:, 1:]
        else:
            shape = shapes[feature["id"]]
            is_sheet = feature["kind"] == "face_extrude" and feature["parameters"]["thickness"] == 0
            if not is_sheet and not list(TopologyExplorer(shape).solids()):
                raise ValueError(f"{feature['name']}: physical body is not a closed solid")
            feature = dict(feature)
            feature["_sheet"] = is_sheet
            verts, tris, ids, faces = [], [], [], []
            for face_index, face in enumerate(TopologyExplorer(shape).faces()):
                vertices, triangles = tessellate_shape(face, cfg)
                start = len(verts)
                verts.extend(vertices)
                tris.extend((np.array(triangles, dtype=int) + start).tolist())
                ids.extend([face_index] * len(triangles))
                surface = BRepAdaptor_Surface(face)
                normal = None
                if surface.GetType() == GeomAbs_Plane:
                    normal = np.array(surface.Plane().Axis().Direction().Coord())
                    if face.Orientation() == TopAbs_REVERSED:
                        normal *= -1
                    normal = normal.tolist()
                faces.append(dict(normal=normal))
            from toolboxes.STEPtoVoxel.step_parser import topology_vertices

            edges = []
            for edge in TopologyExplorer(shape).edges():
                curve = BRepAdaptor_Curve(edge)
                lo, hi = curve.FirstParameter(), curve.LastParameter()
                if np.isfinite([lo, hi]).all():
                    edges.append([list(curve.Value(float(t)).Coord()) for t in np.linspace(lo, hi, 33)])
            feature["_topology"] = dict(face_ids=ids, faces=faces, points=topology_vertices(shape), edges=edges)
        if not len(tris):
            raise ValueError(f"{feature['name']}: empty body")
        meshes.append((feature, np.asarray(verts, dtype=float), np.asarray(tris, dtype=np.int64)))
    # Sketches are also rendered, but are never sent to the voxelizer.
    sketches = []
    for feature in project["features"]:
        if feature["kind"] == "sketch" and feature["enabled"]:
            verts, tris = tessellate_shape(shapes[feature["id"]], cfg)
            sketches.append((feature, np.asarray(verts), np.asarray(tris)))
    return meshes, sketches


def write_meshes(meshes, directory):
    """Write plain array artifacts; no live VTK objects cross process boundaries."""
    records = []
    for feature, vertices, triangles in meshes:
        filename = feature["id"] + ".npz"
        np.savez(directory / filename, vertices=vertices, triangles=triangles)
        records.append(
            dict(
                id=feature["id"],
                file=filename,
                name=feature["name"],
                material=feature["material"],
                topology=feature.get("_topology"),
            )
        )
    return records

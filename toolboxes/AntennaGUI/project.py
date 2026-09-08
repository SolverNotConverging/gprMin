"""Serializable authoring document; independent of Qt, VTK and the solver."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shutil
import uuid

SCHEMA = "gprmax-antenna-project"
VERSION = 1


def identifier():
    return "id_" + uuid.uuid4().hex


def node(kind, name=None, material="pec", **parameters):
    return dict(
        id=identifier(),
        kind=kind,
        name=name or kind.title(),
        material=material,
        visible=True,
        enabled=True,
        parameters=parameters,
    )


def new_project():
    from .build import settings

    return dict(
        schema=SCHEMA,
        version=VERSION,
        name="Untitled antenna",
        units="mm",
        features=[],
        materials={
            "pec": dict(name="Perfect conductor", builtin="pec", color="#d6a74d"),
            "air": dict(name="Air", builtin="free_space", color="#99cddd"),
            "substrate": dict(name="Illustrative substrate", er=2.2, se=0.0, mr=1.0, sm=0.0, color="#509e83"),
        },
        simulation=dict(
            domain=[0.13, 0.09, 0.07], spacing=[0.001] * 3, time=4e-9, pml=8, fmin=8e9, fmax=12e9, points=101
        ),
        ports=[],
        ntff=[],
        snapshots=[],
        build=settings(),
        coordinates=dict(enabled=False, origin=[0.0, 0.0, 0.0], u=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0]),
    )


def revision(project):
    return hashlib.sha256(json.dumps(project, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_project(path):
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA or document.get("version") != VERSION:
        raise ValueError("Unsupported antenna project format/version")
    # Resolve only asset references, never arbitrary document strings.
    for feature in document["features"]:
        asset = feature["parameters"].get("asset")
        if asset:
            feature["parameters"]["asset"] = str((path.parent / asset).resolve())
    return document


def save_project(project, path):
    path = Path(path).resolve()
    document = copy.deepcopy(project)
    for feature in document["features"]:
        asset = feature["parameters"].get("asset")
        if not asset:
            continue
        source = Path(asset).resolve()
        if not source.is_file():
            raise ValueError(f"Missing imported asset: {source}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        target = path.parent / "assets" / (digest + source.suffix.lower())
        target.parent.mkdir(parents=True, exist_ok=True)
        if source != target:
            shutil.copy2(source, target)
        feature["parameters"]["asset"] = target.relative_to(path.parent).as_posix()
    atomic_json(path, document)


def dependencies(feature):
    p = feature["parameters"]
    if feature["kind"] in ("union", "subtract", "intersect"):
        return [p["target"], p["tool"]]
    if feature["kind"] in ("extrude", "revolve", "transform", "face_extrude"):
        return [p["source"]]
    return []


def physical_features(project):
    consumed = set()
    for item in project["features"]:
        if item["enabled"]:
            if item["kind"] != "face_extrude":
                consumed.update(dependencies(item))
    return [
        item
        for item in project["features"]
        if item["enabled"] and item["id"] not in consumed and item["kind"] != "sketch"
    ]


def finite_vector(value, length=3):
    return (
        isinstance(value, (list, tuple))
        and len(value) == length
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in value)
    )


def validate(project, *, full=False):
    """Cheap authoring checks. Solver preprocessing remains authoritative."""
    errors = []
    frames = [project.get("coordinates", {})] + [f.get("frame", {}) for f in project["features"]]
    for frame in frames:
        if not frame:
            continue
        u, v = frame.get("u"), frame.get("v")
        if not finite_vector(frame.get("origin")) or not finite_vector(u) or not finite_vector(v):
            errors.append("Coordinate frame needs finite origin, U and V vectors")
        elif (
            abs(sum(x * x for x in u) - 1) > 1e-8
            or abs(sum(x * x for x in v) - 1) > 1e-8
            or abs(sum(x * y for x, y in zip(u, v))) > 1e-8
        ):
            errors.append("Coordinate frame U and V must be perpendicular unit vectors")
    sim = project["simulation"]
    for key in ("domain", "spacing"):
        if not finite_vector(sim[key]) or min(sim[key]) <= 0:
            errors.append(f"{key}: enter three positive finite dimensions")
    if errors:
        return errors
    if any(abs(d / h - round(d / h)) > 1e-6 for d, h in zip(sim["domain"], sim["spacing"])):
        errors.append("Domain dimensions must be integer multiples of the mesh spacing")
    if not isinstance(sim["pml"], int) or sim["pml"] < 0:
        errors.append("PML thickness must be a nonnegative integer")
    elif any(d / h <= 2 * sim["pml"] for d, h in zip(sim["domain"], sim["spacing"])):
        errors.append("PML consumes the whole domain")
    if not math.isfinite(sim["time"]) or sim["time"] <= 0:
        errors.append("Time window must be positive")
    known = set()
    for feature in project["features"]:
        if feature["id"] in known:
            errors.append("Duplicate feature identity")
        if any(ref not in known for ref in dependencies(feature)):
            errors.append(f"{feature['name']}: missing or forward feature reference")
        known.add(feature["id"])
        if feature["material"] not in project["materials"]:
            errors.append(f"{feature['name']}: missing material")
        if feature["kind"] == "sheet" and project["materials"].get(feature["material"], {}).get("builtin") != "pec":
            errors.append(f"{feature['name']}: a PEC sheet must use a perfect-conductor material")
    for key, mat in project["materials"].items():
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", str(mat.get("color", "#b5c3cf"))):
            errors.append(f"{mat['name']}: color must use #RRGGBB hexadecimal notation")
        if "builtin" not in mat:
            values = [mat.get(k) for k in ("er", "se", "mr", "sm")]
            if not finite_vector(values, 4) or values[0] < 1 or values[2] < 1 or min(values[1::2]) < 0:
                errors.append(f"{mat['name']}: invalid constitutive properties")
    for region in project["ports"] + project["ntff"] + project["snapshots"]:
        a, b = region["p1"], region["p2"]
        if not finite_vector(a) or not finite_vector(b) or any(x > y for x, y in zip(a, b)):
            errors.append(f"{region['name']}: bounds must be finite and ordered minimum to maximum")
    if not full:
        return errors
    if not 0 < sim["fmin"] < sim["fmax"] or sim["points"] < 2:
        errors.append("Frequency band needs 0 < minimum < maximum and at least two samples")
    numbers = set()
    for port in project["ports"]:
        if port["number"] in numbers or port["number"] < 1:
            errors.append("Port numbers must be unique positive integers")
        numbers.add(port["number"])
        a, b = port["p1"], port["p2"]
        if not finite_vector(a) or not finite_vector(b):
            errors.append(f"Port {port['number']}: invalid bounds")
            continue
        if sum(abs(x - y) < 1e-12 for x, y in zip(a, b)) != 1 or any(x > y for x, y in zip(a, b)):
            errors.append(f"Port {port['number']}: aperture must be an axis-aligned rectangle")
        for point in (a, b):
            if any(x < 0 or x > d for x, d in zip(point, sim["domain"])):
                errors.append(f"Port {port['number']}: outside domain")
            if any(abs(x / h - round(x / h)) > 1e-6 for x, h in zip(point, sim["spacing"])):
                errors.append(f"Port {port['number']}: snap the aperture to the mesh")
        modes = port["modes"]
        if not modes or len(set(modes)) != len(modes) or any(not isinstance(m, int) or m < 1 for m in modes):
            errors.append(f"Port {port['number']}: modes must be unique positive integers")
        if port["excite"] and port["mode"] not in modes:
            errors.append(f"Port {port['number']}: driven mode must be monitored")
    if not any(p["excite"] for p in project["ports"]):
        errors.append("Configure at least one driven eigenmode port")
    for monitor in project["ntff"]:
        if not monitor["frequencies"] or any(not math.isfinite(f) or f <= 0 for f in monitor["frequencies"]):
            errors.append("NTFF needs positive finite frequencies")
        if any(f < sim["fmin"] or f > sim["fmax"] for f in monitor["frequencies"]):
            errors.append("NTFF frequencies must lie inside the eigenmode band")
        if not 0 < monitor["theta_step"] <= 180 or not 0 < monitor["phi_step"] <= 180:
            errors.append("NTFF angular steps must lie in (0, 180] degrees")
    return errors


def default_port():
    return dict(
        id=identifier(),
        name="Port",
        number=1,
        p1=[0.012, 0.033, 0.029],
        p2=[0.012, 0.057, 0.041],
        direction="+",
        modes=[1],
        anchors="auto",
        excite=True,
        mode=1,
        amplitude=1.0,
        phase_deg=0.0,
        delay_s=0.0,
        termination="virtual",
        length_cells=30,
        pml_cells=12,
        source_clearance_cells=6,
    )


def default_ntff():
    return dict(
        id=identifier(),
        name="Far field",
        p1=[0.010, 0.011, 0.011],
        p2=[0.118, 0.079, 0.059],
        frequencies=[8e9, 9e9, 10e9, 11e9, 12e9],
        theta_step=5.0,
        phi_step=5.0,
    )

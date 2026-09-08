"""Explicit CAD-to-FDTD coordinate mapping and editable engineering estimates."""

import copy
import math
import numpy as np

from .project import physical_features

C0 = 299792458.0


def settings():
    return dict(
        enabled=True,
        cells_per_wavelength=20,
        maximum_cell=0.001,
        clearance_wavelengths=0.25,
        clearance_cells=15,
        pml_cells=10,
        round_trips=5,
        automatic_time=True,
        maximum_cells=20_000_000,
    )


def prepare(project, meshes):
    """Return a solver-space copy; never translate the CAD authoring document."""
    solver = copy.deepcopy(project)
    options = {**settings(), **project.get("build", {})}
    enabled = project.get("build", {}).get("enabled", False)
    sim = solver["simulation"]
    offset = np.zeros(3)
    report = dict(cad_origin=[0.0, 0.0, 0.0], warnings=[])
    if not enabled:
        return solver, meshes, report
    for key in (
        "cells_per_wavelength",
        "maximum_cell",
        "clearance_wavelengths",
        "clearance_cells",
        "pml_cells",
        "round_trips",
        "maximum_cells",
    ):
        if not math.isfinite(options[key]) or options[key] <= 0:
            raise ValueError(f"Build setting {key} must be finite and positive")
    if options["cells_per_wavelength"] < 10 or options["pml_cells"] < 10 or options["clearance_cells"] < 15:
        raise ValueError("Automatic build requires ≥10 cells/wavelength, ≥10 PML cells and ≥15 clearance cells")
    if not 0 < sim["fmin"] < sim["fmax"] or not math.isfinite(sim["fmax"]):
        raise ValueError("Enter a positive minimum and maximum frequency before building")
    used = {f["material"] for f in physical_features(project)}
    index = max(
        [1.0]
        + [math.sqrt(m["er"] * m["mr"]) for k, m in project["materials"].items() if k in used and "builtin" not in m]
    )
    wavelength = C0 / sim["fmax"] / index
    h = min(options["maximum_cell"], wavelength / options["cells_per_wavelength"])
    # Resolve thin solid bounding dimensions with at least three cells.
    extents = [x for f, v, _ in meshes if not f.get("_sheet") for x in np.ptp(v, axis=0) if x > 1e-10]
    if extents:
        h = min(h, min(extents) / 3)
    # Round down for reproducible, readable engineering dimensions.
    scale = 10 ** math.floor(math.log10(h))
    h = math.floor(h / scale * 100) / 100 * scale
    points = [v for _, v, _ in meshes]
    for f in physical_features(project):
        if f["kind"] == "sheet":
            points.append(np.array([f["parameters"]["p1"], f["parameters"]["p2"]]))
    if not points:
        raise ValueError("Create physical geometry before building")
    for region in project["ports"] + project["ntff"] + project["snapshots"]:
        points.append(np.array([region["p1"], region["p2"]]))
    points = np.concatenate(points)
    # Padding uses the longest requested free-space wavelength, not just fmax.
    clearance = max(options["clearance_cells"] * h, options["clearance_wavelengths"] * C0 / sim["fmin"])
    pml = int(math.ceil(options["pml_cells"]))
    pad = int(math.ceil(clearance / h)) + pml
    lower = np.floor(points.min(axis=0) / h) - pad
    upper = np.ceil(points.max(axis=0) / h) + pad
    cells = (upper - lower).astype(np.int64)
    if math.prod(map(int, cells)) > options["maximum_cells"]:
        raise ValueError(
            f"Automatic mesh needs {math.prod(map(int, cells)):,} cells ({cells.tolist()}). "
            "Inspect thin dimensions or adjust build settings; the cell budget was exceeded."
        )
    offset = -lower * h
    sim.update(domain=(cells * h).tolist(), spacing=[h] * 3, pml=pml)
    if options["automatic_time"]:
        flight = 2 * np.linalg.norm(cells * h) * index / C0
        # Include pulse duration, modal launch delay and a narrow-band spectral window.
        excitation = 6 / (sim["fmax"] - sim["fmin"]) + max([0.0] + [p["delay_s"] for p in project["ports"]])
        sim["time"] = float(excitation + options["round_trips"] * flight)
    for f in solver["features"]:
        if f["kind"] == "sheet":
            for key in ("p1", "p2"):
                f["parameters"][key] = (np.array(f["parameters"][key]) + offset).tolist()
    for region in solver["ports"] + solver["ntff"] + solver["snapshots"]:
        for key in ("p1", "p2"):
            original = np.array(region[key]) + offset
            snapped = np.round(original / h) * h
            region[key] = snapped.tolist()
            if np.max(abs(snapped - original)) > h * 1e-6:
                report["warnings"].append(
                    f"{region['name']} {key} snapped by up to {np.max(abs(snapped-original))*1e3:.4g} mm"
                )
        if "dl" in region:
            region["dl"] = [max(1, round(x / h)) * h for x in region["dl"]]
    solver["build"]["enabled"] = False
    report.update(
        cad_origin=(-offset).tolist(),
        domain=sim["domain"],
        spacing=sim["spacing"],
        pml_cells=pml,
        clearance=clearance,
        cells=cells.tolist(),
        time=sim["time"],
        refractive_index=index,
    )
    report["warnings"].append(
        "Automatic mesh/padding and run time are starting estimates; verify mesh convergence and late-time decay for resonant antennas."
    )
    return solver, [(f, v + offset, t) for f, v, t in meshes], report

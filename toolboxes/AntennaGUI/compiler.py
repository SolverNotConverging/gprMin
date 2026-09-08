"""Compile one authoring revision to deterministic solver assets and Python."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .project import physical_features, atomic_json, revision, validate


def compile_geometry(project, meshes, directory):
    from toolboxes.STEPtoVoxel.voxeliser import make_grid_from_bbox, world_to_voxel_coords, voxelise_solid_scanline
    from toolboxes.GeometryImport.common import write_geometry_hdf5
    from gprMax.material_database import create_database_document, write_database

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    errors = validate(project)
    if errors:
        raise ValueError("\n".join(errors))
    materials = {}
    for key, mat in project["materials"].items():
        if "builtin" in mat:
            materials[key] = dict(name=mat["name"], model="builtin", builtin=mat["builtin"])
        else:
            materials[key] = dict(
                name=mat["name"],
                model="constant",
                base=dict(
                    relative_permittivity=mat["er"],
                    electric_conductivity_s_per_m=mat["se"],
                    relative_permeability=mat["mr"],
                    magnetic_conductivity_s_per_m=mat["sm"],
                ),
            )
    keys = list(materials)
    write_database(directory / "materials.json", create_database_document("materials", materials))
    warnings = []
    surface_meshes = [(f, v, t) for f, v, t in meshes if f.get("_sheet")]
    meshes = [(f, v, t) for f, v, t in meshes if not f.get("_sheet")]
    surface_triangles = []
    for feature, vertices, triangles in surface_meshes:
        count_before = len(surface_triangles)
        if project["materials"][feature["material"]].get("builtin") != "pec":
            raise ValueError(
                f"{feature['name']}: zero-thickness sheets require PEC; give other materials a finite thickness"
            )
        if not np.any(np.ptp(vertices, axis=0) < 1e-10):
            raise ValueError(
                f"{feature['name']}: gprMax thin sheets must align to a global coordinate plane; use a finite thickness for an inclined sheet"
            )
        for tri in triangles:
            pts = vertices[tri]
            snapped = np.round(pts / project["simulation"]["spacing"])
            if not np.any(np.cross(snapped[1] - snapped[0], snapped[2] - snapped[0])):
                warnings.append(f"{feature['name']}: a sheet triangle collapsed on the grid")
                continue
            surface_triangles.append(
                dict(p1=pts[0].tolist(), p2=pts[1].tolist(), p3=pts[2].tolist(), thickness=0, material_id="pec")
            )
        if len(surface_triangles) == count_before:
            raise ValueError(f"{feature['name']}: sheet disappeared on the mesh")
    origin = None
    if meshes:
        lower = np.min([v.min(axis=0) for _, v, _ in meshes], axis=0)
        upper = np.max([v.max(axis=0) for _, v, _ in meshes], axis=0)
        domain = np.asarray(project["simulation"]["domain"])
        if np.any(lower < -1e-10) or np.any(upper > domain + 1e-10):
            raise ValueError("Physical geometry lies outside the simulation domain")
        spacing = project["simulation"]["spacing"]
        grid = make_grid_from_bbox(lower, upper, dx=spacing[0], dy=spacing[1], dz=spacing[2], pad=0)
        cells = int(np.prod(grid.nxyz))
        import psutil

        if cells * 24 > psutil.virtual_memory().available * 0.65:
            raise ValueError(f"Mesh needs approximately {cells*24/1e9:.1f} GB for compilation. Use a coarser grid.")
        data = np.full(tuple(grid.nxyz), -1, dtype=np.int16)
        tags = np.zeros(tuple(grid.nxyz), dtype=np.uint32)
        names = ["untagged"]
        for i, (feature, vertices, triangles) in enumerate(meshes, 1):
            if np.any(np.ptp(vertices, axis=0) < np.asarray(spacing)):
                warnings.append(f"{feature['name']}: at least one body dimension is smaller than a mesh cell")
            mask = voxelise_solid_scanline(
                world_to_voxel_coords(vertices, grid), triangles, tuple(grid.nxyz), preserve_thin_features=False
            )
            if not mask.any():
                raise ValueError(f"{feature['name']} disappeared on the mesh. Refine the mesh or use a PEC sheet.")
            data[mask] = keys.index(feature["material"])
            tags[mask] = i
            names.append(feature["id"])
        for i, (feature, _, _) in enumerate(meshes, 1):
            if not np.any(tags == i):
                warnings.append(f"{feature['name']}: fully covered by later bodies")
        origin = grid.origin_world.tolist()
        write_geometry_hdf5(
            directory / "geometry.h5",
            data,
            spacing,
            origin=origin,
            material_keys=keys,
            material_database="materials",
            tag_data=tags,
            tag_names=names,
        )
    sheets = [f for f in physical_features(project) if f["kind"] == "sheet"]
    for sheet in sheets:
        a, b = sheet["parameters"]["p1"], sheet["parameters"]["p2"]
        if sum(abs(x - y) < 1e-12 for x, y in zip(a, b)) != 1 or any(x > y for x, y in zip(a, b)):
            raise ValueError(f"{sheet['name']}: sheet needs an axis-aligned rectangular extent")
    manifest = dict(
        revision=revision(project),
        origin=origin,
        warnings=warnings,
        mesh=project["simulation"]["spacing"],
        sheets=sheets,
        surface_triangles=surface_triangles,
    )
    atomic_json(directory / "geometry_manifest.json", manifest)
    return manifest


def constructor_calls(project, geometry, purpose="full"):
    """A single intermediate representation serves live preview and script export."""
    sim = project["simulation"]
    calls = [
        ("Title", dict(name=project["name"])),
        ("Domain", dict(p1=sim["domain"])),
        ("Discretisation", dict(p1=sim["spacing"])),
        ("TimeWindow", dict(time=sim["time"])),
        ("PMLThickness", dict(thickness=sim["pml"])),
    ]
    if geometry["origin"] is not None:
        calls.append(
            ("GeometryObjectsRead", dict(p1=geometry["origin"], geofile="@geometry.h5", material_database="materials"))
        )
    for sheet in geometry["sheets"]:
        calls.append(("Plate", dict(**sheet["parameters"], material_id="pec", tag=sheet["id"])))
    for triangle in geometry.get("surface_triangles", []):
        calls.append(("Triangle", triangle))
    calls.append(
        (
            "GeometryView",
            dict(p1=[0, 0, 0], p2=sim["domain"], dl=sim["spacing"], filename="solver_geometry", output_type="n"),
        )
    )
    if purpose == "geometry":
        return calls
    extra = sorted({f for n in project["ntff"] for f in n["frequencies"]})
    calls.append(
        ("EigenmodeBand", dict(id="band", fmin=sim["fmin"], fmax=sim["fmax"], points=sim["points"], frequencies=extra))
    )
    for port in project["ports"]:
        calls.append(
            (
                "EigenmodePort",
                dict(
                    port=port["number"],
                    p1=port["p1"],
                    p2=port["p2"],
                    direction=port["direction"],
                    modes=port["modes"],
                    anchors=port["anchors"],
                    plot_fields=False,
                ),
            )
        )
        if port["termination"] == "virtual":
            calls.append(
                (
                    "VirtualWaveguide",
                    dict(
                        port=port["number"],
                        **{k: port[k] for k in ("length_cells", "pml_cells", "source_clearance_cells")},
                    ),
                )
            )
        if port["excite"]:
            calls.append(
                (
                    "EigenmodeExcitation",
                    dict(
                        port=port["number"],
                        mode=port["mode"],
                        waveform="auto",
                        plot_waveform=False,
                        **{k: port[k] for k in ("amplitude", "phase_deg", "delay_s")},
                    ),
                )
            )
    if purpose in ("modes", "validate", "run"):
        calls.append(("EigenmodeFieldOutput", dict(filename="port_modes")))
    if purpose == "modes":
        return calls
    for item in project["ntff"]:
        surface, transform = item["id"], item["id"] + "_band"
        calls.extend(
            [
                ("NTFFSurface", dict(p1=item["p1"], p2=item["p2"], id=surface)),
                (
                    "NTFFFrequencyTransform",
                    dict(surface_id=surface, id=transform, frequencies=item["frequencies"], window="rectangular"),
                ),
                (
                    "NTFFAntennaPorts",
                    dict(transform_id=transform, port_ids=[f"port{p['number']}" for p in project["ports"]]),
                ),
                (
                    "NTFFFarFieldArray",
                    dict(
                        transform_id=transform,
                        id="pattern",
                        theta_start=0,
                        theta_stop=180,
                        theta_step=item["theta_step"],
                        phi_start=0,
                        phi_stop=360 - item["phi_step"],
                        phi_step=item["phi_step"],
                        outputs=[
                            "Etheta",
                            "Ephi",
                            "directivity_dbi",
                            "gain_dbi",
                            "realized_gain_dbi",
                            "radiation_efficiency",
                            "total_efficiency",
                        ],
                    ),
                ),
            ]
        )
    for item in project["snapshots"]:
        for i, time in enumerate(item["times"]):
            calls.append(
                (
                    "Snapshot",
                    dict(
                        p1=item["p1"],
                        p2=item["p2"],
                        dl=item["dl"],
                        time=time,
                        filename=f"{item['id']}_{i:04d}",
                        fileext=".h5",
                        outputs=item["outputs"],
                    ),
                )
            )
    return calls


def scene_from_calls(calls, directory):
    import gprMax

    scene = gprMax.Scene()
    for name, kwargs in calls:
        args = {
            k: str(Path(directory) / v[1:]) if isinstance(v, str) and v.startswith("@") else v
            for k, v in kwargs.items()
        }
        scene.add(getattr(gprMax, name)(**args))
    return scene


def export_script(project, geometry, directory):
    import gprMax

    directory = Path(directory)
    lines = [
        '"""Exported antenna model. Edit materials.json and run settings below.',
        "Geometry and mesh spacing are compiled assets; rebuild them in Antenna GUI.",
        '"""',
        "from pathlib import Path",
        "import argparse",
        "import gprMax",
        "",
        "ROOT = Path(__file__).resolve().parent",
        f'TIME_WINDOW = {project["simulation"]["time"]!r}',
        "",
        "def build_scene():",
        "    scene = gprMax.Scene()",
    ]
    for name, kwargs in constructor_calls(project, geometry):
        args = []
        for k, v in kwargs.items():
            expression = f"str(ROOT / {v[1:]!r})" if isinstance(v, str) and v.startswith("@") else repr(v)
            if name == "TimeWindow" and k == "time":
                expression = "TIME_WINDOW"
            args.append(f"{k}={expression}")
        lines.append(f"    scene.add(gprMax.{name}({', '.join(args)}))")
    lines.extend(
        [
            "    return scene",
            "",
            'if __name__ == "__main__":',
            "    parser = argparse.ArgumentParser(description=__doc__)",
            '    parser.add_argument("--geometry-only", action="store_true")',
            '    parser.add_argument("--gpu", type=int)',
            '    parser.add_argument("--output", type=Path, default=ROOT / "results" / "antenna")',
            "    args = parser.parse_args()",
            "    args.output.parent.mkdir(parents=True, exist_ok=True)",
            '    options = {} if args.gpu is None else {"gpu": [args.gpu]}',
            "    gprMax.run(scenes=[build_scene()], outputfile=args.output, geometry_only=args.geometry_only, **options)",
            "",
        ]
    )
    (directory / "run_antenna.py").write_text("\n".join(lines), encoding="utf-8")
    atomic_json(
        directory / "export_manifest.json",
        dict(
            schema="gprmax-antenna-export",
            version=1,
            project_revision=revision(project),
            solver_version=gprMax.__version__,
            simulation=project["simulation"],
            snapshots=project["snapshots"],
            ntff=project["ntff"],
            warnings=geometry["warnings"],
        ),
    )

"""Isolated CAD/preprocessing jobs. Completion is atomic and revision-tagged."""

from pathlib import Path
import argparse
import traceback

from .project import load_project, atomic_json, revision, validate, node, physical_features


def execute(project, operation, directory):
    from .cad import regenerate, write_meshes
    from .compiler import compile_geometry, constructor_calls, scene_from_calls, export_script

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    errors = validate(project, full=False)
    if errors:
        raise ValueError("\n".join(errors))
    print("Building CAD geometry", flush=True)
    meshes, sketches = regenerate(project)
    result = dict(
        revision=revision(project),
        operation=operation,
        meshes=write_meshes(meshes, directory),
        sketches=write_meshes(sketches, directory),
    )
    if operation == "cad":
        return result
    if not physical_features(project):
        raise ValueError("Add at least one enabled physical body before solver preprocessing")
    from .build import prepare

    project, meshes, report = prepare(project, meshes)
    result["build"] = report
    atomic_json(directory / "build_report.json", report)
    atomic_json(
        directory / "solver_settings.json", {key: project[key] for key in ("simulation", "ports", "ntff", "snapshots")}
    )
    print("Build settings: " + str(report), flush=True)
    errors = validate(project, full=operation in ("modes", "validate", "export", "run"))
    if errors:
        raise ValueError("\n".join(errors))
    print("Voxelizing physical bodies", flush=True)
    geometry = compile_geometry(project, meshes, directory)
    result.update(warnings=report["warnings"] + geometry["warnings"])
    purpose = "validate" if operation == "export" else "geometry" if operation == "build" else operation
    calls = constructor_calls(project, geometry, purpose)
    import gprMax

    print("Building solver geometry and requested modes", flush=True)
    # material_database is resolved relative to the input/output model path.
    gprMax.run(
        scenes=[scene_from_calls(calls, directory)],
        outputfile=directory / "antenna",
        geometry_only=operation != "run",
        hide_progress_bars=True,
    )
    result["geometry"] = "solver_geometry.vtkhdf"
    if operation in ("modes", "validate", "export", "run"):
        result["modes"] = "port_modes.modes.h5"
    if operation == "run":
        result["output"] = "antenna.h5"
        from .results import discover_results

        result["plots"] = discover_results(directory)
        names = {item["id"]: item["name"] for item in project["ntff"]}
        for plot in result["plots"]:
            if plot["kind"] == "farfield":
                surface = plot["group"].split("/")[1]
                plot["name"] = "Far field · " + names.get(surface, surface)
    if operation == "export":
        export_script(project, geometry, directory)
        result["script"] = "run_antenna.py"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument(
        "operation", choices=("cad", "geometry", "build", "modes", "validate", "export", "import", "run")
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--asset", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    try:
        project = load_project(args.project)
        if args.operation == "import":
            from toolboxes.STEPtoVoxel import inspect_step

            parts = inspect_step(args.asset)
            result = dict(
                revision=revision(project),
                operation="import",
                features=[
                    node("step", p.name, asset=str(args.asset.resolve()), part_index=i)
                    for i, p in enumerate(parts)
                    if p.cad.get("vol_m3", 0) > 0
                ],
            )
        else:
            result = execute(project, args.operation, args.directory)
        result["ok"] = True
        atomic_json(args.directory / "completed.json", result)
    except Exception as exc:
        traceback.print_exc()
        atomic_json(args.directory / "completed.json", dict(ok=False, error=str(exc)))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Authoring, real CAD/voxel export, and modal-reader contracts."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from toolboxes.AntennaGUI.project import (
    new_project,
    node,
    save_project,
    load_project,
    revision,
    validate,
    physical_features,
)
from toolboxes.AntennaGUI.compiler import compile_geometry, constructor_calls, export_script


def test_project_roundtrip_and_assets(tmp_path):
    source = tmp_path / "source.stl"
    source.write_text("source")
    project = new_project()
    project["features"] = [node("stl", asset=str(source), scale=0.001)]
    path = tmp_path / "saved" / "project.json"
    save_project(project, path)
    reopened = load_project(path)
    assert reopened["features"][0]["id"] == project["features"][0]["id"]
    assert Path(reopened["features"][0]["parameters"]["asset"]).read_text() == "source"
    assert not validate(reopened)
    assert source.is_file()


def test_dependencies_and_material_identity():
    project = new_project()
    a = node("box", origin=[0.01] * 3, size=[0.01] * 3)
    b = node("box", origin=[0.015] * 3, size=[0.01] * 3)
    cut = node("subtract", target=a["id"], tool=b["id"])
    project["features"] = [a, b, cut]
    assert physical_features(project) == [cut]
    a["name"] = "Renamed"
    assert cut["parameters"]["target"] == a["id"]
    assert validate(project) == []
    project["features"] = [cut, a, b]
    assert "forward" in " ".join(validate(project))


def test_invalid_preview_properties_are_rejected():
    project = new_project()
    project["materials"]["pec"]["color"] = "invalid color"
    project["ntff"] = [dict(name="Region", p1=[0.02] * 3, p2=[0.01] * 3)]
    errors = validate(project)
    assert any("#RRGGBB" in error for error in errors)
    assert any("ordered" in error for error in errors)


def test_real_cad_boolean_and_voxel_materials(tmp_path):
    pytest.importorskip("OCC")
    from toolboxes.AntennaGUI.cad import regenerate

    project = new_project()
    a = node("box", origin=[0.01] * 3, size=[0.01] * 3)
    b = node("cylinder", origin=[0.015, 0.015, 0.009], axis=[0, 0, 1], radius=0.002, height=0.012)
    cut = node("subtract", target=a["id"], tool=b["id"])
    fill = node("box", "Dielectric", "substrate", origin=[0.018, 0.01, 0.01], size=[0.004, 0.01, 0.01])
    project["features"] = [a, b, cut, fill]
    meshes, _ = regenerate(project)
    result = compile_geometry(project, meshes, tmp_path)
    with h5py.File(tmp_path / "geometry.h5") as f:
        assert f["tag_names"][:].tolist() == [b"untagged", cut["id"].encode(), fill["id"].encode()]
        assert set(np.unique(f["tag_data"][:])) == {0, 1, 2}
        assert f["data"][5, 5, 5] == -1  # bore through conductor
    from gprMax.material_database import load_material_spec

    spec = load_material_spec("materials", "substrate", search_directory=tmp_path)
    assert spec.relative_permittivity == 2.2
    assert result["origin"] == pytest.approx([0.01] * 3)


def test_sketch_extrude_transform(tmp_path):
    pytest.importorskip("OCC")
    from toolboxes.AntennaGUI.cad import regenerate

    project = new_project()
    sketch = node(
        "sketch", profile="rectangle", origin=[0.01] * 3, u=[1.0, 0, 0], v=[0, 1.0, 0], width=0.01, height=0.005
    )
    extrusion = node("extrude", source=sketch["id"], vector=[0, 0, 0.005])
    moved = node(
        "transform", source=extrusion["id"], origin=[0, 0, 0], axis=[0, 0, 1], angle=0.0, translation=[0.01, 0, 0]
    )
    project["features"] = [sketch, extrusion, moved]
    meshes, _ = regenerate(project)
    assert meshes[0][1].min(axis=0) == pytest.approx([0.02, 0.01, 0.01])
    assert meshes[0][1].max(axis=0) == pytest.approx([0.03, 0.015, 0.015])


def test_disappearing_body_blocks_export(tmp_path):
    pytest.importorskip("OCC")
    from toolboxes.AntennaGUI.cad import regenerate

    project = new_project()
    project["features"] = [node("box", origin=[0.0101] * 3, size=[0.00001] * 3)]
    meshes, _ = regenerate(project)
    with pytest.raises(ValueError, match="disappeared"):
        compile_geometry(project, meshes, tmp_path)


def test_relocatable_script_has_no_gui_dependencies(tmp_path):
    project = new_project()
    geometry = dict(origin=None, sheets=[], warnings=[])
    export_script(project, geometry, tmp_path)
    script = (tmp_path / "run_antenna.py").read_text()
    assert "AntennaGUI" not in script and "OCC" not in script and "pyvista" not in script
    compile(script, "run_antenna.py", "exec")
    spec = importlib.util.spec_from_file_location("exported", tmp_path / "run_antenna.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.build_scene() is not None


@pytest.mark.parametrize("normal", [0, 1, 2])
@pytest.mark.parametrize("direction", ["+", "-"])
def test_modal_banks_roundtrip_and_centering(tmp_path, normal, direction):
    from gprMax.eigenmode_field_output import write_eigenmode_fields
    from toolboxes.AntennaGUI.modes import read_mode

    axes = tuple(i for i in range(3) if i != normal)
    e_shapes = [(2, 4), (3, 3), (3, 4)]
    h_shapes = [(3, 3), (2, 4), (2, 3)]

    def bank(shapes):
        components = [None] * 3
        for local, global_axis in enumerate((*axes, normal)):
            components[global_axis] = np.full(shapes[local], 2 + 3j)
        return [[components, components]]

    owner = SimpleNamespace(
        normal_axis=normal,
        transverse_axes=axes,
        transverse_start=(2, 3),
        transverse_stop=(4, 6),
        plane_index=7,
        direction=direction,
        requested_anchor_policy="auto",
        resolved_anchor_policy="explicit",
    )
    monitor = SimpleNamespace(
        owner=owner,
        port_index=1,
        output_id="port1",
        anchor_frequencies=np.array([1e9]),
        mode_indices=(1, 3),
        anchor_neff=np.array([[1 + 0.1j, 2 + 0.2j]]),
        anchor_operator_neff=None,
        anchor_mode_valid=np.array([[True, False]]),
        anchor_mode_reference_valid=np.array([[True, True]]),
        anchor_mode_propagating=np.array([[True, False]]),
        anchor_balanced_power=np.ones((1, 2)),
        mode_anchor_policies=("explicit", "explicit"),
        anchor_e=bank(e_shapes),
        anchor_h=bank(h_shapes),
    )
    grid = SimpleNamespace(eigenmodeports=[monitor], dl=np.array([0.001] * 3))
    path = write_eigenmode_fields(tmp_path / "modes.h5", grid)
    result = read_mode(path, 1, 0, 1)
    assert not result["valid"]
    for values in result["fields"].values():
        assert values.shape == (2, 3)
        assert np.all(values == 2 + 3j)
    assert result["u"] == pytest.approx([0.0025, 0.0035])
    assert result["v"] == pytest.approx([0.0035, 0.0045, 0.0055])
    assert np.all(monitor.anchor_e[0][0][0] == 2 + 3j)


@pytest.mark.integration
def test_horn_preprocessing_and_export(tmp_path):
    pytest.importorskip("OCC")
    from toolboxes.AntennaGUI.examples import horn
    from toolboxes.AntennaGUI.worker import execute

    project = horn()
    result = execute(project, "export", tmp_path)
    assert result["script"] == "run_antenna.py"
    with h5py.File(tmp_path / result["modes"]) as f:
        group = f["ports/1"]
        assert group["fields/Ex/values"].ndim == 4
        assert group.attrs["normal_axis"] == 0
        assert np.any(group["anchor_mode_valid"][:])
    assert (tmp_path / result["geometry"]).is_file()
    # Compare the GUI's voxelized CAD against the existing native-primitive example.
    import runpy
    import gprMax

    original = (
        Path(__file__).parents[2] / "examples/features/eigenmode_ports/example_3_antenna_and_farfield/horn_antenna.py"
    )
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    gprMax.run(
        scenes=[runpy.run_path(str(original))["build_scene"]()],
        outputfile=baseline / "horn",
        geometry_only=True,
        hide_progress_bars=True,
    )
    with h5py.File(tmp_path / result["geometry"]) as a, h5py.File(baseline / "horn_antenna.vtkhdf") as b:
        np.testing.assert_array_equal(a["VTKHDF/CellData/Material"][:], b["VTKHDF/CellData/Material"][:])


@pytest.mark.integration
def test_patch_preprocessing(tmp_path):
    pytest.importorskip("OCC")
    from toolboxes.AntennaGUI.examples import patch
    from toolboxes.AntennaGUI.worker import execute

    result = execute(patch(), "validate", tmp_path)
    assert (tmp_path / result["modes"]).is_file()

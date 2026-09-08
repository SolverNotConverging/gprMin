import copy
import numpy as np
import pytest

from toolboxes.AntennaGUI.project import new_project, node, physical_features
from toolboxes.AntennaGUI.build import prepare, C0


def test_automatic_bounds_mapping_material_resolution_and_time():
    p = new_project()
    body = node("box", material="substrate", origin=[-0.01] * 3, size=[0.02] * 3)
    p["features"] = [body]
    vertices = np.array([[-0.01] * 3, [0.01] * 3])
    before = copy.deepcopy(p)
    solver, meshes, report = prepare(p, [(body, vertices, np.array([[0, 1, 0]]))])
    assert p == before
    assert np.all(meshes[0][1] > 0)
    np.testing.assert_allclose(meshes[0][1] + report["cad_origin"], vertices)
    sim = solver["simulation"]
    h = sim["spacing"][0]
    assert h <= C0 / sim["fmax"] / np.sqrt(2.2) / 20
    assert np.min(meshes[0][1]) >= (sim["pml"] + 15) * h
    assert sim["time"] > 5 * 2 * np.linalg.norm(sim["domain"]) * np.sqrt(2.2) / C0
    np.testing.assert_allclose(np.array(sim["domain"]) / h, report["cells"])


def test_cell_budget_blocks_unbounded_allocation():
    p = new_project()
    p["build"]["maximum_cells"] = 100
    f = node("box", origin=[-0.01] * 3, size=[0.02] * 3)
    p["features"] = [f]
    with pytest.raises(ValueError, match="cell budget"):
        prepare(p, [(f, np.array([[-0.01] * 3, [0.01] * 3]), [])])


@pytest.mark.integration
def test_auto_negative_waveguide_full_run(tmp_path):
    pytest.importorskip("OCC")
    from toolboxes.AntennaGUI.project import default_port
    from toolboxes.AntennaGUI.worker import execute
    from toolboxes.AntennaGUI.results import read_sparameters

    p = new_project()
    p["features"] = [
        node("box", origin=[-0.015, -0.016, -0.010], size=[0.030, 0.004, 0.020]),
        node("box", origin=[-0.015, 0.012, -0.010], size=[0.030, 0.004, 0.020]),
        node("box", origin=[-0.015, -0.012, -0.010], size=[0.030, 0.024, 0.004]),
        node("box", origin=[-0.015, -0.012, 0.006], size=[0.030, 0.024, 0.004]),
    ]
    port = default_port()
    port.update(p1=[-0.011, -0.012, -0.006], p2=[-0.011, 0.012, 0.006])
    p["ports"] = [port]
    result = execute(p, "run", tmp_path)
    assert result["build"]["cad_origin"][0] < 0
    assert (tmp_path / result["modes"]).exists()
    trace = next(iter(read_sparameters(tmp_path / "antenna_sparameters.csv").values()))
    assert trace["power_wave_valid"].any()
    assert np.isfinite(trace["value"][trace["power_wave_valid"]]).all()


def test_local_signed_geometry_face_extrusion_and_sheet(tmp_path):
    pytest.importorskip("OCC")
    from toolboxes.AntennaGUI.cad import regenerate
    from toolboxes.AntennaGUI.worker import execute

    p = new_project()
    p["simulation"].update(fmin=1e9, fmax=2e9)
    p["build"].update(maximum_cell=0.002, clearance_wavelengths=0.05)
    body = node("box", origin=[0.0, 0.0, 0.0], size=[-0.012, 0.012, 0.012])
    body["frame"] = dict(origin=[-0.02, -0.02, -0.02], u=[0.0, 1.0, 0.0], v=[-1.0, 0.0, 0.0])
    p["features"] = [body]
    meshes, _ = regenerate(p)
    np.testing.assert_allclose(meshes[0][1].min(axis=0), [-0.032, -0.032, -0.02])
    topology = meshes[0][0]["_topology"]
    assert len(topology["faces"]) == 6 and len(topology["points"]) == 8
    face_index = next(i for i, f in enumerate(topology["faces"]) if np.allclose(f["normal"], [0, 0, 1]))
    extrusion = node("face_extrude", source=body["id"], face_index=face_index, normal=[0.0, 0.0, 1.0], thickness=0.006)
    p["features"].append(extrusion)
    assert len(physical_features(p)) == 2
    meshes, _ = regenerate(p)
    np.testing.assert_allclose(meshes[1][1].max(axis=0)[2], -0.002)
    extrusion["parameters"]["thickness"] = 0.0
    result = execute(p, "build", tmp_path)
    assert (tmp_path / result["geometry"]).exists()
    assert result["build"]["cad_origin"][0] < 0
    import json

    manifest = json.loads((tmp_path / "geometry_manifest.json").read_text())
    assert len(manifest["surface_triangles"]) == 2

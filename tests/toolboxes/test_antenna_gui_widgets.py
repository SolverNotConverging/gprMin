"""Native GUI acceptance tests, requiring the optional GUI environment."""

import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyvistaqt")
pytest.importorskip("pytestqt")
pytest.importorskip("OCC")

from toolboxes.AntennaGUI.project import new_project, node, revision


def test_native_local_face_pick_and_automatic_build(qtbot, tmp_path, monkeypatch):
    import numpy as np
    from PySide6 import QtWidgets as W
    from toolboxes.AntennaGUI.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    p = new_project()
    p["simulation"].update(fmin=1e9, fmax=2e9)
    p["build"].update(maximum_cell=0.002, clearance_wavelengths=0.05)
    body = node("box", origin=[-0.012, -0.012, -0.012], size=[0.024, 0.024, 0.024])
    p["features"] = [body]
    try:
        window.open_document(p)
        window.start_job("cad")
        qtbot.waitUntil(lambda: window.job is None, timeout=60000)
        assert body["id"] in window.topology
        window.viewport.camera_position = [(0.0, 0.0, 0.15), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        window.viewport.render()
        renderer = window.viewport.renderer
        renderer.SetWorldPoint(0.003, 0.003, 0.012, 1.0)
        renderer.WorldToDisplay()
        x, y, _ = renderer.GetDisplayPoint()
        for kind in ("point", "edge", "face"):
            window.pick_topology(kind)
            window.viewport.iren.picker.Pick(x, y, 0.0, renderer)
            qtbot.waitUntil(lambda: window.project["coordinates"]["enabled"], timeout=5000)
            window.timer.stop()
            frame = window.project["coordinates"]
            assert np.dot(frame["u"], frame["v"]) == pytest.approx(0)
            if kind == "point":
                np.testing.assert_allclose(frame["origin"], [0.012, 0.012, 0.012], atol=1e-5)
        np.testing.assert_allclose(window.project["coordinates"]["origin"], [0.003, 0.003, 0.012], atol=1e-5)
        window.timer.stop()
        monkeypatch.setattr(W.QInputDialog, "getDouble", lambda *args: (0.0, True))
        window.extrude_face()
        window.timer.stop()
        assert window.project["features"][-1]["kind"] == "face_extrude"
        window.build_model()
        qtbot.waitUntil(lambda: window.job is None, timeout=60000)
        assert window.last_result[1]["operation"] == "build", window.log.toPlainText()
        report = window.last_result[1]["build"]
        np.testing.assert_allclose(window.geometry_grid.origin, report["cad_origin"])
        assert np.min(window.actors[body["id"]].mapper.dataset.points) < 0
        qa = Path(os.environ.get("ANTENNA_GUI_QA", tmp_path))
        qa.mkdir(parents=True, exist_ok=True)
        window.viewport.reset_camera()
        qtbot.wait(100)
        window.grab().save(str(qa / "local-build.png"))
    finally:
        window.saved_revision = revision(window.project)
        window.cancel_job()
        window.close()


def test_polar_cuts_and_surface(qtbot, tmp_path):
    import h5py
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from toolboxes.AntennaGUI.result_view import ResultView

    path = tmp_path / "pattern.h5"
    group = "ntff/surface/frequency/band/far_field/pattern"
    theta, phi = np.meshgrid(np.arange(0, 181, 5), np.arange(0, 360, 5), indexing="ij")
    power = 0.02 + (1 + np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))) ** 2
    with h5py.File(path, "w") as f:
        g = f.create_group(group)
        g.parent.parent["frequencies"] = [1e9]
        g["theta"], g["phi"] = theta.ravel(), phi.ravel()
        g["fields/directivity_dbi"] = (10 * np.log10(power)).reshape(1, -1)
    dialog = ResultView(path, dict(kind="farfield", group=group, name="Pattern"))
    qtbot.addWidget(dialog)
    dialog.show()
    qa = Path(os.environ.get("ANTENNA_GUI_QA", tmp_path))
    qa.mkdir(parents=True, exist_ok=True)
    for index, name in ((1, "theta-polar"), (3, "phi-polar"), (2, "surface")):
        dialog.view.setCurrentIndex(index)
        qtbot.wait(100)
        dialog.canvas.draw()
        ax = dialog.figure.axes[0]
        if index in (1, 3):
            assert ax.name == "polar"
            assert ax.lines[0].get_xdata()[-1] == pytest.approx(2 * np.pi)
            assert ax.lines[0].get_ydata()[0] == ax.lines[0].get_ydata()[-1]
        else:
            assert any(isinstance(artist, Poly3DCollection) for artist in ax.collections)
        dialog.grab().save(str(qa / (name + ".png")))
    dialog.close()


@pytest.mark.integration
def test_window_worker_selection_and_undo(qtbot, tmp_path):
    from toolboxes.AntennaGUI.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    project = new_project()
    project["features"] = [node("box", origin=[0.02] * 3, size=[0.01] * 3)]
    try:
        window.commit(project, "Add test box")
        qtbot.waitUntil(lambda: window.last_result is not None, timeout=60000)
        assert len(window.actors) == 1
        window.selected = ("features", project["features"][0]["id"])
        window.refresh_tree()
        assert window.properties.widget() is not None
        window.undo.undo()
        assert window.project["features"] == []
        window.undo.redo()
        assert len(window.project["features"]) == 1
        path = Path(os.environ.get("ANTENNA_GUI_QA", tmp_path))
        path.mkdir(parents=True, exist_ok=True)
        window.grab().save(str(path / "editor.png"))
    finally:
        window.saved_revision = revision(window.project)
        window.cancel_job()
        window.close()


@pytest.mark.integration
def test_full_run_result_browser_and_reopen(qtbot, tmp_path):
    import numpy as np
    from PySide6 import QtWidgets as W
    from toolboxes.AntennaGUI.app import MainWindow
    from toolboxes.AntennaGUI.examples import horn
    from toolboxes.AntennaGUI.project import save_project
    from toolboxes.AntennaGUI.result_view import ResultView
    from toolboxes.AntennaGUI.results import read_sparameters, read_farfield

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    project = horn()
    path = tmp_path / "horn.json"
    save_project(project, path)
    try:
        window.open_document(project, path)
        window.start_job("modes")
        qtbot.waitUntil(lambda: window.job is None, timeout=60000)
        assert len(window.results) == 1, window.log.toPlainText()
        assert window.result_records(window.results[0])[0]["kind"] == "modes"
        for dialog in window.findChildren(W.QDialog):
            dialog.close()
        window.start_job("run")
        job = window.job
        window.start_job("cad")
        assert window.job is job  # Automatic/repeated actions cannot terminate the solver.
        qtbot.waitUntil(lambda: window.job is None, timeout=120000)
        assert len(window.results) == 2, window.log.toPlainText()
        run = window.results[-1]
        records = window.result_records(run)
        assert {r["kind"] for r in records} == {"modes", "sparameters", "farfield"}
        output = Path(run["directory"])
        s = next(r for r in records if r["kind"] == "sparameters")
        trace = next(iter(read_sparameters(output / s["file"]).values()))
        assert trace["power_wave_valid"].any()
        assert np.isfinite(trace["value"][trace["power_wave_valid"]]).all()
        far = next(r for r in records if r["kind"] == "farfield")
        data = read_farfield(output / far["file"], far["group"], "directivity_dbi", 0)
        assert np.isfinite(data["value"]).any()
        qa = Path(os.environ.get("ANTENNA_GUI_QA", tmp_path))
        qa.mkdir(parents=True, exist_ok=True)
        for record in (s, far):
            dialog = ResultView(output / record["file"], record, window)
            qtbot.addWidget(dialog)
            dialog.show()
            if record is far:
                for view in range(3):
                    dialog.view.setCurrentIndex(view)
                    dialog.canvas.draw()
                dialog.view.setCurrentIndex(0)
                dialog.quantity.setCurrentText("radiation_efficiency")
                dialog.canvas.draw()
                dialog.quantity.setCurrentText("directivity_dbi")
            dialog.canvas.draw()
            dialog.grab().save(str(qa / (record["kind"] + ".png")))
            dialog.close()
        root = window.tree.topLevelItem(window.tree.topLevelItemCount() - 1)
        assert root.text(0) == "Results" and root.childCount() == 2
        window.tree.setCurrentItem(root.child(1).child(0))
        window.grab().save(str(qa / "results-browser.png"))
        window.start_job("cad")
        qtbot.waitUntil(lambda: window.job is None, timeout=60000)
        assert len(window.results) == 2
        window.project["name"] = "Changed model"
        window.refresh_tree()
        root = window.tree.topLevelItem(window.tree.topLevelItemCount() - 1)
        assert "older model" in root.child(0).text(0)
        window.saved_revision = revision(window.project)
        window.open_document(project, path)
        window.timer.stop()
        assert len(window.results) == 2
        window.start_job("run")
        window.cancel_job()
        assert len(window.results) == 2 and window.job is None
    finally:
        window.saved_revision = revision(window.project)
        window.cancel_job()
        window.close()

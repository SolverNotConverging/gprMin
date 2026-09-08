"""Optional Qt desktop editor; expensive work stays in isolated processes."""

from __future__ import annotations

import copy
import json
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6 import QtCore as C, QtGui as G, QtWidgets as W

from .project import (
    new_project,
    node,
    identifier,
    load_project,
    save_project,
    atomic_json,
    revision,
    validate,
    dependencies,
    physical_features,
    default_port,
    default_ntff,
)
from .forms import Properties, SketchDialog
from .mode_view import ModeView


class Edit(G.QUndoCommand):
    def __init__(self, window, after, text):
        super().__init__(text)
        self.window = window
        self.before = copy.deepcopy(window.project)
        self.after = copy.deepcopy(after)

    def redo(self):
        self.window.set_document(copy.deepcopy(self.after))

    def undo(self):
        self.window.set_document(copy.deepcopy(self.before))


class MainWindow(W.QMainWindow):
    def __init__(self):
        super().__init__()
        self.project = new_project()
        self.path = None
        self.saved_revision = revision(self.project)
        self.process = None
        self.job = None
        self.last_result = None
        self.results = []
        self.selected = None
        self.actors = {}
        self.topology = {}
        self.surface_selection = None
        self.geometry_actor = None
        self.transform_widget = None
        self.work = Path(tempfile.mkdtemp(prefix="gprmax-antenna-"))
        self.undo = G.QUndoStack(self)
        self.setWindowTitle("gprMax Antenna Studio")
        self.resize(1440, 940)
        self.viewport = QtInteractor(self)
        self.setCentralWidget(self.viewport)
        self.viewport.set_background("#192733")
        self.viewport.add_axes()
        self.viewport.enable_mesh_picking(callback=self.picked, use_actor=True, show=False)
        self.tree = W.QTreeWidget()
        self.tree.setHeaderLabels(["Model / results"])
        self.tree.itemSelectionChanged.connect(self.selection_changed)
        self.tree.itemActivated.connect(lambda item, column: self.open_result_item(item))
        self.dock("Model", self.tree, C.Qt.DockWidgetArea.LeftDockWidgetArea)
        self.properties = W.QScrollArea()
        self.properties.setWidgetResizable(True)
        self.properties.setMinimumWidth(330)
        self.dock("Properties", self.properties, C.Qt.DockWidgetArea.RightDockWidgetArea)
        self.log = W.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(3000)
        self.log.setMaximumHeight(180)
        self.dock("Validation and jobs", self.log, C.Qt.DockWidgetArea.BottomDockWidgetArea)
        self.timer = C.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(lambda: self.start_job("cad"))
        self.build_actions()
        self.refresh_tree()
        self.resizeDocks(
            [self.tree.parentWidget(), self.properties.parentWidget()], [250, 380], C.Qt.Orientation.Horizontal
        )
        self.statusBar().showMessage("Create a body or open an example. Dimensions are shown in mm.")

    def dock(self, title, widget, area):
        dock = W.QDockWidget(title, self)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)

    def action(self, menu, title, callback, shortcut=None):
        action = G.QAction(title, self)
        action.triggered.connect(lambda checked=False: callback())
        if shortcut:
            action.setShortcut(shortcut)
        menu.addAction(action)
        return action

    def build_actions(self):
        file = self.menuBar().addMenu("File")
        self.action(file, "New", lambda: self.open_document(new_project()), "Ctrl+N")
        self.action(file, "Open…", self.open_file, "Ctrl+O")
        self.action(file, "Save", self.save, "Ctrl+S")
        self.action(file, "Save as…", lambda: self.save(True), "Ctrl+Shift+S")
        self.action(file, "Import STEP / STL…", self.import_asset)
        self.action(file, "Export validated run script…", self.export)
        examples = file.addMenu("Examples")
        from .examples import horn, patch

        self.action(examples, "Waveguide-fed horn", lambda: self.open_document(horn()))
        self.action(examples, "Probe-fed patch", lambda: self.open_document(patch()))
        edit = self.menuBar().addMenu("Edit")
        edit.addAction(self.undo.createUndoAction(self, "Undo"))
        edit.actions()[-1].setShortcut("Ctrl+Z")
        edit.addAction(self.undo.createRedoAction(self, "Redo"))
        edit.actions()[-1].setShortcut("Ctrl+Y")
        self.action(edit, "Duplicate", self.duplicate, "Ctrl+D")
        self.action(edit, "Delete", self.delete, "Delete")
        self.action(edit, "Earlier in overlap order", lambda: self.reorder(-1))
        self.action(edit, "Later in overlap order", lambda: self.reorder(1))
        cad = self.menuBar().addMenu("Geometry")
        coordinates = cad.addMenu("Local coordinates")
        self.action(coordinates, "Enable local coordinates", lambda: self.enable_local(True))
        self.action(coordinates, "Use global coordinates", lambda: self.enable_local(False))
        self.action(coordinates, "Align to point", lambda: self.pick_topology("point"))
        self.action(coordinates, "Align to edge", lambda: self.pick_topology("edge"))
        self.action(coordinates, "Align to face", lambda: self.pick_topology("face"))
        self.action(cad, "Select face for extrusion", lambda: self.pick_topology("extrude"))
        self.action(cad, "Extrude selected face…", self.extrude_face)
        self.action(cad, "Box", lambda: self.add_feature(node("box", origin=[0.0, 0.0, 0.0], size=[0.01, 0.01, 0.01])))
        self.action(
            cad,
            "Cylinder",
            lambda: self.add_feature(
                node("cylinder", origin=[0.0, 0.0, 0.0], axis=[0.0, 0.0, 1.0], radius=0.005, height=0.01)
            ),
        )
        self.action(cad, "Planar sketch…", self.sketch)
        self.action(cad, "Capture sketch plane from surface", self.capture_plane)
        self.action(cad, "Extrude selected sketch", lambda: self.dependent("extrude"))
        self.action(cad, "Revolve selected sketch", lambda: self.dependent("revolve"))
        self.action(cad, "Transform selected body", lambda: self.dependent("transform"))
        self.action(cad, "Drag / rotate selected body", self.transform_selected)
        self.action(cad, "Union…", lambda: self.boolean("union"))
        self.action(cad, "Subtract…", lambda: self.boolean("subtract"))
        self.action(cad, "Intersect…", lambda: self.boolean("intersect"))
        self.action(
            cad,
            "PEC sheet (final overlay)",
            lambda: self.add_feature(node("sheet", p1=[0.02, 0.02, 0.02], p2=[0.04, 0.04, 0.02])),
        )
        material = self.menuBar().addMenu("Materials")
        self.action(material, "New material", self.add_material)
        sim = self.menuBar().addMenu("Simulation")
        self.action(sim, "Build model automatically", self.build_model)
        self.action(sim, "Run simulation (CPU)", lambda: self.start_job("run"), "F5")
        self.action(sim, "Cancel running job", self.cancel_job)
        self.action(sim, "Open completed result folder…", self.import_results)
        self.action(sim, "Add eigenmode port", self.add_port)
        self.action(sim, "Snap selected port to grid", self.snap_port)
        self.action(sim, "Add NTFF surface", lambda: self.add_item("ntff", default_ntff()))
        self.action(
            sim,
            "Add field snapshots",
            lambda: self.add_item(
                "snapshots",
                dict(
                    id=identifier(),
                    name="Field snapshots",
                    p1=[0.01] * 3,
                    p2=[0.05] * 3,
                    dl=self.project["simulation"]["spacing"],
                    times=[1e-9],
                    outputs=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
                ),
            ),
        )
        view = self.menuBar().addMenu("View")
        self.action(view, "Fit model", self.viewport.reset_camera, "F")
        for label, method in (("Isometric", "view_isometric"), ("XY", "view_xy"), ("XZ", "view_xz"), ("YZ", "view_yz")):
            self.action(view, label, lambda m=method: getattr(self.viewport, m)())
        self.action(view, "Toggle orthographic", self.toggle_projection)
        self.action(
            view,
            "Measure two points",
            lambda: self.viewport.add_measurement_widget(
                lambda a, b, d: self.statusBar().showMessage(f"Distance: {d*1e3:.6g} mm")
            ),
        )
        self.action(view, "Voxel cross-section", self.slice_geometry)
        self.action(view, "Inspect saved modes", self.show_modes)
        bar = self.addToolBar("Workflow")
        self.action(bar, "Build model", self.build_model)
        for title, operation in (
            ("Rebuild CAD", "cad"),
            ("Preview solver geometry", "geometry"),
            ("Inspect port modes", "modes"),
            ("Validate export", "validate"),
            ("Run simulation", "run"),
        ):
            self.action(bar, title, lambda op=operation: self.start_job(op))
        self.action(bar, "Export…", self.export)
        self.action(bar, "Cancel job", self.cancel_job)
        self.design_toggle = W.QCheckBox("Design")
        self.design_toggle.setChecked(True)
        self.design_toggle.toggled.connect(self.update_visibility)
        bar.addWidget(self.design_toggle)
        self.voxel_toggle = W.QCheckBox("Solver geometry")
        self.voxel_toggle.toggled.connect(self.update_visibility)
        bar.addWidget(self.voxel_toggle)
        self.opacity = W.QSlider(C.Qt.Orientation.Horizontal)
        self.opacity.setRange(10, 100)
        self.opacity.setValue(100)
        self.opacity.setMaximumWidth(90)
        self.opacity.valueChanged.connect(self.update_visibility)
        bar.addWidget(W.QLabel("Opacity"))
        bar.addWidget(self.opacity)

    def set_document(self, document):
        from .build import settings

        document.setdefault("build", {**settings(), "enabled": False})
        document.setdefault(
            "coordinates", dict(enabled=False, origin=[0.0, 0.0, 0.0], u=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0])
        )
        self.project = document
        self.refresh_tree()
        self.setWindowTitle("gprMax Antenna Studio — " + document["name"] + " *")
        if not self.job or self.job["operation"] == "cad":
            self.timer.start(400)

    def commit(self, document, text):
        errors = validate(document)
        if errors:
            self.error("\n".join(errors))
            return
        self.undo.push(Edit(self, document, text))

    def error(self, message):
        self.log.appendPlainText(message)
        W.QMessageBox.warning(self, "Antenna Studio", message)

    def refresh_tree(self):
        self.tree.blockSignals(True)
        self.tree.clear()
        for title, key in (
            ("Geometry (later bodies win; PEC sheets last)", "features"),
            ("Materials", "materials"),
            ("Simulation domain / mesh", "simulation"),
            ("Automatic build settings", "build"),
            ("Local coordinate system", "coordinates"),
            ("Eigenmode ports", "ports"),
            ("Near-to-far-field", "ntff"),
            ("Field snapshots", "snapshots"),
        ):
            root = W.QTreeWidgetItem([title])
            self.tree.addTopLevelItem(root)
            if key in ("simulation", "build", "coordinates"):
                root.setData(0, C.Qt.ItemDataRole.UserRole, (key, None))
                continue
            items = self.project[key].items() if key == "materials" else ((v["id"], v) for v in self.project[key])
            for identity, value in items:
                suffix = "" if key != "features" else "  [" + value["kind"] + "]"
                child = W.QTreeWidgetItem([value["name"] + suffix])
                child.setData(0, C.Qt.ItemDataRole.UserRole, (key, identity))
                root.addChild(child)
                if self.selected == (key, identity):
                    self.tree.setCurrentItem(child)
            root.setExpanded(True)
        root = W.QTreeWidgetItem(["Results"])
        self.tree.addTopLevelItem(root)
        for index, run in enumerate(self.results):
            stale = run["result"]["revision"] != revision(self.project)
            branch = W.QTreeWidgetItem([run["label"] + (" · older model" if stale else "")])
            root.addChild(branch)
            records = self.result_records(run)
            for number, record in enumerate(records):
                child = W.QTreeWidgetItem([record["name"]])
                child.setData(0, C.Qt.ItemDataRole.UserRole, ("results", index, number))
                child.setToolTip(0, str(Path(run["directory"]) / record["file"]))
                branch.addChild(child)
                if self.selected == ("results", index, number):
                    self.tree.setCurrentItem(child)
            branch.setExpanded(True)
        root.setExpanded(True)
        self.tree.blockSignals(False)
        self.selection_changed()

    def lookup(self, document, selection=None):
        key, identity = selection or self.selected
        if key in ("simulation", "build", "coordinates"):
            return document[key]
        if key == "materials":
            return document[key][identity]
        return next(v for v in document[key] if v["id"] == identity)

    def selection_changed(self):
        item = self.tree.currentItem()
        selected = item.data(0, C.Qt.ItemDataRole.UserRole) if item else None
        if not selected:
            self.selected = None
            self.properties.setWidget(W.QWidget())
            return
        self.selected = tuple(selected)
        if self.selected[0] == "results":
            frame = W.QWidget()
            layout = W.QVBoxLayout(frame)
            run = self.results[self.selected[1]]
            note = W.QLabel(run["label"] + "\n" + run["directory"])
            note.setWordWrap(True)
            note.setSizePolicy(W.QSizePolicy.Policy.Ignored, W.QSizePolicy.Policy.Preferred)
            layout.addWidget(note)
            button = W.QPushButton("Open result")
            button.clicked.connect(lambda: self.open_result_item(self.tree.currentItem()))
            layout.addWidget(button)
            folder = W.QPushButton("Open output folder")
            folder.clicked.connect(lambda: G.QDesktopServices.openUrl(C.QUrl.fromLocalFile(run["directory"])))
            layout.addWidget(folder)
            layout.addStretch()
            self.properties.setWidget(frame)
            return
        try:
            value = self.lookup(self.project)
        except (KeyError, StopIteration):
            return
        frame = W.QWidget()
        layout = W.QVBoxLayout(frame)
        if self.selected[0] == "simulation" and self.project.get("build", {}).get("enabled"):
            note = W.QLabel(
                "Automatic build uses this frequency band. Domain and spacing are calculated at build time; "
                "time is also calculated when Automatic time is enabled. Calculated values appear in the build log."
            )
            note.setWordWrap(True)
            layout.addWidget(note)
        refs = {"material": [(k, v["name"]) for k, v in self.project["materials"].items()]}
        features = self.project["features"]
        previous = features[: features.index(value)] if self.selected[0] == "features" else features
        for key in ("source", "target", "tool"):
            refs[key] = [(v["id"], v["name"]) for v in previous if v["kind"] not in ("stl", "sheet")]
        main = {k: v for k, v in value.items() if k not in ("parameters", "kind", "builtin", "frame")}
        if value.get("frame"):
            label = W.QLabel("Dimensions use the local coordinate frame captured when this feature was created.")
            label.setWordWrap(True)
            layout.addWidget(label)
        editor = Properties(main, references=refs)
        layout.addWidget(editor)
        params = Properties(value["parameters"], references=refs) if "parameters" in value else None
        if params:
            layout.addWidget(params)
        button = W.QPushButton("Apply changes")

        def apply():
            try:
                document = copy.deepcopy(self.project)
                target = self.lookup(document)
                target.update(editor.values())
                if params:
                    target["parameters"] = params.values()
                self.commit(document, "Edit properties")
            except (ValueError, TypeError) as exc:
                self.error(str(exc))

        button.clicked.connect(apply)
        layout.addWidget(button)
        if self.selected[0] == "ports":
            warning = W.QLabel("Virtual waveguide is experimental. A port alone is not an absorbing termination.")
            warning.setWordWrap(True)
            layout.addWidget(warning)
        if self.selected[0] == "features" and value["kind"] == "sketch":
            edit = W.QPushButton("Replace drawn profile…")

            def redraw():
                dialog = SketchDialog(self, value["parameters"])
                if dialog.exec():
                    document = copy.deepcopy(self.project)
                    self.lookup(document)["parameters"] = dialog.values()
                    self.commit(document, "Edit sketch")

            edit.clicked.connect(redraw)
            layout.addWidget(edit)
        layout.addStretch()
        self.properties.setWidget(frame)
        for identity, actor in self.actors.items():
            actor.prop.show_edges = identity == self.selected[1]
        self.viewport.render()

    def add_feature(self, feature):
        frame = self.project.get("coordinates", {})
        if frame.get("enabled") and feature["kind"] in ("box", "cylinder", "sketch") and not feature.get("frame"):
            feature["frame"] = copy.deepcopy(frame)
        self.add_item("features", feature)

    def add_item(self, key, item):
        document = copy.deepcopy(self.project)
        document[key].append(item)
        self.selected = (key, item["id"])
        self.commit(document, "Add " + item["name"])

    def sketch(self):
        dialog = SketchDialog(self)
        if dialog.exec():
            try:
                self.add_feature(node("sketch", **dialog.values()))
            except ValueError as exc:
                self.error(str(exc))

    def selected_feature(self):
        if not self.selected or self.selected[0] != "features":
            self.error("Select a geometry feature first")
            return None
        return self.lookup(self.project)

    def dependent(self, kind):
        feature = self.selected_feature()
        if not feature:
            return
        if kind in ("extrude", "revolve") and feature["kind"] != "sketch":
            self.error("Select a planar sketch")
            return
        if feature["kind"] in ("stl", "sheet"):
            self.error("STL and PEC sheets use explicit placement coordinates; solid operations require CAD bodies")
            return
        p = dict(source=feature["id"])
        p.update(
            dict(vector=[0.0, 0.0, 0.005])
            if kind == "extrude"
            else dict(origin=[0.0, 0.0, 0.0], axis=[0.0, 0.0, 1.0], angle=360.0 if kind == "revolve" else 0.0)
        )
        if kind == "transform":
            p["translation"] = [0.0, 0.0, 0.0]
        self.add_feature(node(kind, material=feature["material"], **p))

    def boolean(self, kind):
        target = self.selected_feature()
        if not target:
            return
        candidates = [
            f for f in physical_features(self.project) if f["id"] != target["id"] and f["kind"] not in ("stl", "sheet")
        ]
        if target["kind"] in ("stl", "sheet", "sketch") or not candidates:
            self.error("Select a CAD solid and create another solid to use as the tool")
            return
        labels = [f"{i+1}: {f['name']}" for i, f in enumerate(candidates)]
        text, ok = W.QInputDialog.getItem(self, kind.title(), "Tool body", labels, 0, False)
        if ok:
            self.add_feature(
                node(kind, material=target["material"], target=target["id"], tool=candidates[labels.index(text)]["id"])
            )

    def duplicate(self):
        feature = self.selected_feature()
        if feature:
            duplicate = copy.deepcopy(feature)
            duplicate.update(id=identifier(), name=feature["name"] + " copy")
            self.add_item("features", duplicate)

    def delete(self):
        if not self.selected or self.selected[0] in ("simulation", "build", "coordinates", "results"):
            return
        key, identity = self.selected
        document = copy.deepcopy(self.project)
        if key == "features" and any(identity in dependencies(f) for f in document["features"]):
            self.error("Delete dependent features first")
            return
        if key == "materials":
            if any(f["material"] == identity for f in document["features"]):
                self.error("Reassign bodies using this material first")
                return
            del document[key][identity]
        else:
            document[key] = [v for v in document[key] if v["id"] != identity]
        self.selected = None
        self.commit(document, "Delete item")

    def reorder(self, offset):
        feature = self.selected_feature()
        if feature:
            document = copy.deepcopy(self.project)
            features = document["features"]
            index = next(i for i, f in enumerate(features) if f["id"] == feature["id"])
            target = index + offset
            if 0 <= target < len(features):
                features[index], features[target] = features[target], features[index]
                self.commit(document, "Reorder body")

    def add_material(self):
        document = copy.deepcopy(self.project)
        key = identifier()
        document["materials"][key] = dict(name="New material", er=2.2, se=0.0, mr=1.0, sm=0.0, color="#6fa391")
        self.selected = ("materials", key)
        self.commit(document, "Add material")

    def add_port(self):
        port = default_port()
        port["number"] = max([p["number"] for p in self.project["ports"]] + [0]) + 1
        port["name"] = f"Port {port['number']}"
        self.add_item("ports", port)

    def snap_port(self):
        if not self.selected or self.selected[0] != "ports":
            return
        document = copy.deepcopy(self.project)
        port = self.lookup(document)
        for k in ("p1", "p2"):
            port[k] = [round(x / h) * h for x, h in zip(port[k], document["simulation"]["spacing"])]
        self.commit(document, "Snap port")

    def maybe_save(self):
        if revision(self.project) == self.saved_revision:
            return True
        answer = W.QMessageBox.question(
            self,
            "Unsaved project",
            "Save the current project?",
            W.QMessageBox.StandardButton.Save
            | W.QMessageBox.StandardButton.Discard
            | W.QMessageBox.StandardButton.Cancel,
        )
        return (
            self.save()
            if answer == W.QMessageBox.StandardButton.Save
            else answer == W.QMessageBox.StandardButton.Discard
        )

    def open_document(self, document, path=None):
        if not self.maybe_save():
            return
        self.cancel_job()
        self.results = []
        self.last_result = None
        self.selected = None
        self.path = path
        if path:
            index = Path(path).with_suffix(".results.json")
            if index.exists():
                try:
                    self.results = json.loads(index.read_text(encoding="utf-8"))
                except (ValueError, OSError) as exc:
                    self.log.appendPlainText(f"Could not load result index: {exc}")
        self.undo.clear()
        self.set_document(document)
        self.saved_revision = revision(document)

    def open_file(self):
        path, _ = W.QFileDialog.getOpenFileName(self, "Open antenna project", "", "Antenna project (*.json)")
        if path:
            try:
                self.open_document(load_project(path), Path(path))
            except Exception as exc:
                self.error(str(exc))

    def save(self, save_as=False):
        path = self.path
        if not path or save_as:
            name, _ = W.QFileDialog.getSaveFileName(
                self, "Save antenna project", "project.json", "Antenna project (*.json)"
            )
            if not name:
                return False
            path = Path(name)
        try:
            save_project(self.project, path)
            self.path = path
            self.saved_revision = revision(self.project)
            self.save_result_index()
            self.setWindowTitle("gprMax Antenna Studio — " + self.project["name"])
            return True
        except Exception as exc:
            self.error(str(exc))
            return False

    def import_asset(self):
        filename, _ = W.QFileDialog.getOpenFileName(self, "Import geometry", "", "Geometry (*.step *.stp *.stl)")
        if not filename:
            return
        if Path(filename).suffix.lower() == ".stl":
            units, ok = W.QInputDialog.getItem(self, "STL units", "Source length unit", ["mm", "m", "µm"], 0, False)
            if ok:
                self.add_feature(
                    node(
                        "stl",
                        Path(filename).stem,
                        asset=filename,
                        scale={"mm": 1e-3, "m": 1.0, "µm": 1e-6}[units],
                        translation=[0.0, 0.0, 0.0],
                        rotation=[0.0, 0.0, 0.0],
                    )
                )
        else:
            self.start_job("import", asset=filename)

    def export(self):
        parent = W.QFileDialog.getExistingDirectory(self, "Export into a new subdirectory")
        if parent:
            name = re.sub(r"[^A-Za-z0-9_-]+", "_", self.project["name"])
            directory = Path(parent) / (name + "_" + identifier()[-8:])
            self.start_job("export", directory=directory)

    def cancel_job(self):
        self.timer.stop()
        process = self.process
        self.process = None
        self.job = None
        if process and process.state() != C.QProcess.ProcessState.NotRunning:
            process.kill()
            process.waitForFinished(1000)
            self.statusBar().showMessage("Job cancelled; completed results retained")

    def start_job(self, operation, directory=None, asset=None):
        if self.job and self.job["operation"] != "cad":
            self.statusBar().showMessage("A job is running. Cancel it before starting another job.")
            return
        self.cancel_job()
        if directory is None and self.path and operation in ("run", "modes", "validate", "build"):
            directory = self.path.parent / (self.path.stem + "_results") / identifier()
        directory = directory or self.work / identifier()
        directory.mkdir(parents=True, exist_ok=True)
        project_file = directory / "input.json"
        atomic_json(project_file, self.project)
        process = C.QProcess(self)
        process.setProcessChannelMode(C.QProcess.ProcessChannelMode.MergedChannels)
        arguments = ["-u", "-m", "toolboxes.AntennaGUI.worker", str(project_file), operation, str(directory)]
        if asset:
            arguments.extend(["--asset", asset])
        job = dict(process=process, directory=directory, revision=revision(self.project), operation=operation)
        self.process, self.job = process, job
        process.readyReadStandardOutput.connect(
            lambda: self.log.appendPlainText(
                bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace").rstrip()
            )
        )
        process.finished.connect(lambda code, status: self.job_finished(job, code))

        def failed(error):
            self.log.appendPlainText("Worker: " + process.errorString())
            if error == C.QProcess.ProcessError.FailedToStart and self.job is job:
                self.process = None
                self.job = None
                self.statusBar().showMessage("Worker could not start; completed results retained")

        process.errorOccurred.connect(failed)
        process.start(sys.executable, arguments)
        self.statusBar().showMessage(f"{operation.title()} in progress…")

    def job_finished(self, job, code):
        if self.job is not job:
            return
        self.process = None
        self.job = None
        directory = job["directory"]
        path = directory / "completed.json"
        if not path.exists():
            self.statusBar().showMessage("Job cancelled or worker failed; previous view retained")
            return
        result = json.loads(path.read_text(encoding="utf-8"))
        if code != 0 or not result["ok"]:
            self.log.appendPlainText("ERROR: " + result.get("error", f"Worker exited with code {code}"))
            self.statusBar().showMessage("Build failed; previous view retained")
            return
        if result.get("modes") or result.get("plots"):
            self.register_result(directory, result)
        if revision(self.project) != job["revision"]:
            self.log.appendPlainText("Completed results belong to an older model revision; geometry preview retained")
            self.statusBar().showMessage("Completed for an older model; see Results")
            return
        if result["operation"] == "import":
            document = copy.deepcopy(self.project)
            document["features"].extend(result["features"])
            self.commit(document, "Import assembly")
            return
        self.last_result = (directory, result)
        self.display_result(directory, result)
        for warning in result.get("warnings", []):
            self.log.appendPlainText("WARNING: " + warning)
        self.statusBar().showMessage("Completed: " + result["operation"])
        if result["operation"] == "modes":
            self.show_modes()
        if result.get("script"):
            self.log.appendPlainText("Exported: " + str(directory / result["script"]))
            W.QMessageBox.information(
                self,
                "Export complete",
                f"Runnable model:\n{directory / result['script']}\n\nGeometry is fixed; edit materials.json and run settings outside the GUI.",
            )

    def picked(self, actor):
        identity = next((k for k, v in self.actors.items() if v is actor), None)
        if actor is self.geometry_actor and self.geometry_actor is not None:
            picker = self.viewport.iren.picker
            cell = picker.GetCellId() if hasattr(picker, "GetCellId") else -1
            surface = actor.mapper.dataset
            if cell >= 0 and "TagID" in surface.cell_data and "geometry_tag_names" in self.geometry_grid.field_data:
                tag = int(surface.cell_data["TagID"][cell])
                identity = str(self.geometry_grid.field_data["geometry_tag_names"][tag])
        if identity:
            self.selected = ("features", identity)
            self.refresh_tree()

    def display_result(self, directory, result):
        camera = self.viewport.camera_position if self.actors else None
        if self.transform_widget:
            self.transform_widget.remove()
            self.transform_widget = None
        self.viewport.clear()
        self.viewport.add_axes()
        self.actors = {}
        self.topology = {}
        self.surface_selection = None
        self.geometry_actor = None
        self.geometry_grid = None
        by_id = {f["id"]: f for f in self.project["features"]}
        for record in result["meshes"] + result.get("sketches", []):
            with np.load(directory / record["file"]) as data:
                faces = np.column_stack((np.full(len(data["triangles"]), 3), data["triangles"])).ravel()
                mesh = pv.PolyData(data["vertices"], faces)
            if record.get("topology"):
                mesh.cell_data["CADFace"] = record["topology"]["face_ids"]
                self.topology[record["id"]] = record["topology"]
            feature = by_id[record["id"]]
            actor = self.viewport.add_mesh(
                mesh,
                color=self.project["materials"][feature["material"]].get("color", "#b5c3cf"),
                name=feature["id"],
                reset_camera=False,
            )
            self.actors[feature["id"]] = actor
        for feature in physical_features(self.project):
            if feature["kind"] == "sheet":
                mesh = self.rectangle(feature["parameters"]["p1"], feature["parameters"]["p2"])
                self.actors[feature["id"]] = self.viewport.add_mesh(
                    mesh, color="#d6a74d", name=feature["id"], reset_camera=False
                )
        built = result.get("build", {})
        cad_origin = np.array(built.get("cad_origin", [0.0, 0.0, 0.0]))
        domain = np.array(built.get("domain", self.project["simulation"]["domain"]))
        self.viewport.add_mesh(
            pv.Box(bounds=tuple(v for a, d in zip(cad_origin, domain) for v in (a, a + d))).outline(),
            color="#8193a0",
            pickable=False,
        )
        if built.get("domain"):
            pad = np.array(built["spacing"]) * built["pml_cells"]
            self.viewport.add_mesh(
                pv.Box(
                    bounds=tuple(v for a, d, p in zip(cad_origin, domain, pad) for v in (a + p, a + d - p))
                ).outline(),
                color="#ee9a58",
                pickable=False,
            )
            self.log.appendPlainText(
                f"Built {built['cells']} cells; Δ={built['spacing'][0]*1e3:.5g} mm; "
                f"PML={built['pml_cells']} cells; run time={built['time']*1e9:.5g} ns; CAD origin={built['cad_origin']}"
            )
        for port in self.project["ports"]:
            if sum(abs(a - b) < 1e-12 for a, b in zip(port["p1"], port["p2"])) == 1:
                mesh = self.rectangle(port["p1"], port["p2"])
                self.viewport.add_mesh(mesh, color="#df6aad", opacity=0.45, pickable=False)
                axis = int(np.argmin(np.abs(np.array(port["p2"]) - port["p1"])))
                direction = np.eye(3)[axis] * (1 if port["direction"] == "+" else -1)
                center = (np.array(port["p1"]) + port["p2"]) / 2
                self.viewport.add_mesh(
                    pv.Arrow(start=center, direction=direction, scale=0.008), color="#df6aad", pickable=False
                )
        for n in self.project["ntff"] + self.project["snapshots"]:
            bounds = tuple(x for pair in zip(n["p1"], n["p2"]) for x in pair)
            self.viewport.add_mesh(pv.Box(bounds=bounds).outline(), color="#4fddcf", pickable=False)
        if result.get("geometry"):
            grid = pv.read(directory / result["geometry"])
            grid = grid.translate(cad_origin, inplace=False)
            self.geometry_grid = grid
            # Preserve the regular grid for slices; render occupied material surfaces.
            if "TagID" in grid.cell_data:
                surface = grid.threshold(0.5, scalars="TagID").extract_surface()
                scalar = "TagID"
            else:
                surface = grid.extract_surface()
                scalar = "Material"
            self.geometry_actor = self.viewport.add_mesh(
                surface, scalars=scalar, show_scalar_bar=False, opacity=0.7, pickable=True
            )
            self.voxel_toggle.setChecked(True)
        if camera:
            self.viewport.camera_position = camera
        else:
            self.viewport.view_isometric()
            self.viewport.reset_camera()
        self.update_visibility()
        self.draw_local_frame()

    def build_model(self):
        if not self.project.get("build", {}).get("enabled"):
            from .build import settings

            document = copy.deepcopy(self.project)
            document["build"] = {**settings(), **document.get("build", {}), "enabled": True}
            self.commit(document, "Enable automatic model build")
        self.start_job("build")

    def enable_local(self, enabled):
        document = copy.deepcopy(self.project)
        document["coordinates"]["enabled"] = enabled
        self.commit(document, "Change coordinate system")

    def draw_local_frame(self):
        frame = self.project.get("coordinates", {})
        if not frame.get("enabled"):
            return
        u, v = np.array(frame["u"]), np.array(frame["v"])
        length = max(self.viewport.length * 0.08, 0.001)
        for axis, color, name in zip((u, v, np.cross(u, v)), ("#ef5350", "#66bb6a", "#42a5f5"), ("U", "V", "W")):
            self.viewport.add_mesh(
                pv.Arrow(start=frame["origin"], direction=axis, scale=length),
                color=color,
                name="local_" + name,
                pickable=False,
                reset_camera=False,
            )
            self.viewport.add_point_labels(
                [np.array(frame["origin"]) + axis * length],
                [name],
                name="label_" + name,
                point_size=0,
                always_visible=True,
            )

    def pick_topology(self, kind):
        self.viewport.disable_picking()
        self.statusBar().showMessage(f"Right-click the CAD {kind if kind != 'extrude' else 'face'} to select it")

        def picked(position, picker):
            actor = picker.GetActor()
            identity = next((key for key, value in self.actors.items() if value is actor), None)
            if identity not in self.topology:
                self.error("Select a CAD body in Design view; mesh imports have no CAD topology")
                return
            topology = self.topology[identity]
            cell = picker.GetCellId()
            if cell < 0 or cell >= len(topology["face_ids"]):
                return
            face_index = topology["face_ids"][cell]
            normal = topology["faces"][face_index]["normal"]
            point = np.array(position)
            frame = copy.deepcopy(self.project["coordinates"])
            if kind == "point":
                points = np.array(topology["points"])
                if not len(points):
                    self.error("This CAD body has no topological vertices")
                    return
                point = points[np.argmin(np.linalg.norm(points - point, axis=1))]
            elif kind == "edge":
                candidates = []
                for edge in topology["edges"]:
                    a, b = np.array(edge[:-1]), np.array(edge[1:])
                    d = b - a
                    valid = np.sum(d * d, axis=1) > 1e-24
                    a, d = a[valid], d[valid]
                    if len(a):
                        t = np.clip(np.sum((point - a) * d, axis=1) / np.sum(d * d, axis=1), 0, 1)
                        points = a + t[:, None] * d
                        i = np.argmin(np.linalg.norm(points - point, axis=1))
                        candidates.append((np.linalg.norm(points[i] - point), points[i], d[i]))
                if not candidates:
                    self.error("No CAD edge is available")
                    return
                _, point, u = min(candidates, key=lambda value: value[0])
                u = u / np.linalg.norm(u)
                seed = np.array(normal) if normal is not None else np.eye(3)[np.argmin(abs(u))]
                if np.linalg.norm(np.cross(seed, u)) < 1e-8:
                    seed = np.eye(3)[np.argmin(abs(u))]
                v = np.cross(seed, u)
                frame.update(u=u.tolist(), v=(v / np.linalg.norm(v)).tolist())
            elif normal is None:
                self.error("Select a planar face for alignment or extrusion")
                return
            else:
                normal = np.array(normal)
                u = np.eye(3)[np.argmin(abs(normal))]
                u -= np.dot(u, normal) * normal
                u /= np.linalg.norm(u)
                frame.update(u=u.tolist(), v=np.cross(normal, u).tolist())
            self.surface_selection = dict(
                source=identity,
                face_index=face_index,
                normal=normal.tolist() if isinstance(normal, np.ndarray) else normal,
            )
            self.viewport.disable_picking()
            self.viewport.enable_mesh_picking(callback=self.picked, use_actor=True, show=False)
            if kind == "extrude":
                C.QTimer.singleShot(0, self.extrude_face)
            else:
                document = copy.deepcopy(self.project)
                frame.update(origin=point.tolist(), enabled=True)
                document["coordinates"] = frame
                self.commit(document, "Align local coordinates to " + kind)

        self.viewport.enable_surface_point_picking(
            callback=picked, use_picker=True, show_point=False, show_message=False
        )

    def extrude_face(self):
        if not self.surface_selection or self.surface_selection["normal"] is None:
            self.pick_topology("extrude")
            return
        selection = copy.deepcopy(self.surface_selection)
        thickness, accepted = W.QInputDialog.getDouble(
            self, "Extrude CAD face", "Signed thickness (mm); 0 creates a PEC sheet", 1.0, -1e6, 1e6, 6
        )
        if accepted:
            source = next(f for f in self.project["features"] if f["id"] == selection["source"])
            self.add_feature(
                node(
                    "face_extrude",
                    source["name"] + (" sheet" if thickness == 0 else " extrusion"),
                    material="pec" if thickness == 0 else source["material"],
                    thickness=thickness * 1e-3,
                    **selection,
                )
            )

    @staticmethod
    def rectangle(a, b):
        a, b = np.array(a), np.array(b)
        axis = int(np.argmin(abs(b - a)))
        u, v = [i for i in range(3) if i != axis]
        points = np.array([a, a, a, a], dtype=float)
        points[1, u] = b[u]
        points[2, [u, v]] = b[[u, v]]
        points[3, v] = b[v]
        return pv.PolyData(points, [4, 0, 1, 2, 3])

    def update_visibility(self, *_):
        features = {f["id"]: f for f in self.project["features"]}
        for identity, actor in self.actors.items():
            actor.visibility = self.design_toggle.isChecked() and features.get(identity, {}).get("visible", False)
            actor.prop.opacity = self.opacity.value() / 100
        if self.geometry_actor is not None:
            self.geometry_actor.visibility = self.voxel_toggle.isChecked()
        self.viewport.render()

    def toggle_projection(self):
        self.viewport.camera.parallel_projection = not self.viewport.camera.parallel_projection
        self.viewport.render()

    def transform_selected(self):
        feature = self.selected_feature()
        if not feature or feature["id"] not in self.actors:
            return
        if feature["kind"] in ("sheet", "sketch"):
            self.error("Use coordinate properties for sheets and sketch planes")
            return
        if self.transform_widget:
            self.transform_widget.remove()
        from scipy.spatial.transform import Rotation

        def released(matrix):
            matrix = np.asarray(matrix).copy()

            def commit_transform():
                if feature["kind"] == "stl":
                    document = copy.deepcopy(self.project)
                    target = next(f for f in document["features"] if f["id"] == feature["id"])
                    p = target["parameters"]
                    old = Rotation.from_euler("xyz", p.get("rotation", [0, 0, 0]), degrees=True).as_matrix()
                    p["rotation"] = Rotation.from_matrix(matrix[:3, :3] @ old).as_euler("xyz", degrees=True).tolist()
                    p["translation"] = (
                        matrix[:3, :3] @ np.asarray(p.get("translation", [0, 0, 0])) + matrix[:3, 3]
                    ).tolist()
                    self.commit(document, "Transform imported mesh")
                else:
                    vector = Rotation.from_matrix(matrix[:3, :3]).as_rotvec()
                    angle = float(np.linalg.norm(vector))
                    axis = (vector / angle).tolist() if angle > 1e-12 else [0.0, 0.0, 1.0]
                    self.add_feature(
                        node(
                            "transform",
                            feature["name"] + " placement",
                            material=feature["material"],
                            source=feature["id"],
                            origin=[0.0, 0.0, 0.0],
                            axis=axis,
                            angle=float(np.rad2deg(angle)),
                            translation=matrix[:3, 3].tolist(),
                        )
                    )

            C.QTimer.singleShot(0, commit_transform)

        self.transform_widget = self.viewport.add_affine_transform_widget(
            self.actors[feature["id"]], release_callback=released
        )

    def capture_plane(self):
        self.viewport.disable_picking()
        self.statusBar().showMessage("Right-click a planar surface to capture a detached sketch plane")

        def captured(position, picker):
            normal = np.asarray(picker.GetPickNormal(), dtype=float)
            if np.linalg.norm(normal) < 1e-10:
                return
            normal /= np.linalg.norm(normal)
            axis = np.eye(3)[int(np.argmin(abs(normal)))]
            u = np.cross(axis, normal)
            u /= np.linalg.norm(u)
            v = np.cross(normal, u)

            def create():
                self.viewport.disable_picking()
                self.viewport.enable_mesh_picking(callback=self.picked, use_actor=True, show=False)
                dialog = SketchDialog(self, dict(origin=list(position), u=u.tolist(), v=v.tolist()))
                if dialog.exec():
                    try:
                        self.add_item("features", node("sketch", **dialog.values()))
                    except ValueError as exc:
                        self.error(str(exc))

            C.QTimer.singleShot(0, create)

        self.viewport.enable_surface_point_picking(
            callback=captured, use_picker=True, show_point=False, show_message=False
        )

    def slice_geometry(self):
        if getattr(self, "geometry_grid", None) is None:
            self.error("Preview solver geometry first")
            return
        self.viewport.clear_plane_widgets()
        self.viewport.add_mesh_slice(self.geometry_grid, scalars="Material", normal="z", name="material_slice")

    def show_modes(self):
        run = next((run for run in reversed(self.results) if run["result"].get("modes")), None)
        if run is None:
            self.error("Inspect port modes first")
            return
        self.open_result(run, self.result_records(run)[0])

    @staticmethod
    def result_records(run):
        result = run["result"]
        records = []
        if result.get("modes"):
            records.append(dict(kind="modes", file=result["modes"], name="Modal fields"))
        return records + result.get("plots", [])

    def register_result(self, directory, result):
        self.results.append(
            dict(
                directory=str(Path(directory).resolve()),
                result=result,
                label=result["operation"].title() + " · " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        )
        self.save_result_index()
        self.refresh_tree()
        root = self.tree.topLevelItem(self.tree.topLevelItemCount() - 1)
        branch = root.child(root.childCount() - 1)
        if branch and branch.childCount():
            self.tree.scrollToItem(branch.child(branch.childCount() - 1))

    def save_result_index(self):
        if self.path:
            try:
                atomic_json(self.path.with_suffix(".results.json"), self.results)
            except OSError as exc:
                self.log.appendPlainText(f"Result index could not be saved: {exc}")

    def import_results(self):
        folder = W.QFileDialog.getExistingDirectory(self, "Open completed Antenna Studio result folder")
        if folder:
            try:
                result = json.loads((Path(folder) / "completed.json").read_text(encoding="utf-8"))
                if not result.get("ok") or not (result.get("modes") or result.get("plots")):
                    raise ValueError("This folder contains no completed modal or simulation results")
                self.register_result(Path(folder), result)
            except (OSError, ValueError) as exc:
                self.error(str(exc))

    def open_result_item(self, item):
        selected = item.data(0, C.Qt.ItemDataRole.UserRole) if item else None
        if selected and selected[0] == "results":
            run = self.results[selected[1]]
            self.open_result(run, self.result_records(run)[selected[2]])

    def open_result(self, run, record):
        try:
            path = Path(run["directory"]) / record["file"]
            if record["kind"] == "modes":
                dialog = ModeView(path, self)
            else:
                from .result_view import ResultView

                dialog = ResultView(path, record, self)
            suffix = " · older model" if run["result"]["revision"] != revision(self.project) else ""
            dialog.setWindowTitle(record["name"] + " — " + run["label"] + suffix)
            dialog.setAttribute(C.Qt.WidgetAttribute.WA_DeleteOnClose)
            dialog.show()
        except (OSError, ValueError, KeyError) as exc:
            self.error(f"Cannot open result: {exc}")

    def closeEvent(self, event):
        if not self.maybe_save():
            event.ignore()
            return
        self.cancel_job()
        if self.transform_widget:
            self.transform_widget.remove()
            self.transform_widget = None
        self.viewport.close()
        event.accept()


def main():
    app = W.QApplication.instance() or W.QApplication(sys.argv)
    app.setApplicationName("gprMax Antenna Studio")
    window = MainWindow()
    if len(sys.argv) > 1:
        try:
            window.open_document(load_project(sys.argv[1]), Path(sys.argv[1]))
        except Exception as exc:
            window.error(str(exc))
    window.show()
    app.exec()

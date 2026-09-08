"""Small typed property editors with explicit engineering units."""

import copy
from PySide6 import QtCore, QtGui, QtWidgets as W

LENGTHS = {
    "origin",
    "size",
    "radius",
    "height",
    "width",
    "vector",
    "translation",
    "p1",
    "p2",
    "domain",
    "spacing",
    "dl",
    "maximum_cell",
    "thickness",
}
TIMES = {"time", "delay_s", "times"}
FREQUENCIES = {"fmin", "fmax", "frequencies"}


class EngineeringSpin(W.QDoubleSpinBox):
    def textFromValue(self, value):
        return f"{value:.9g}"


class Properties(W.QWidget):
    def __init__(self, values, parent=None, references=None):
        super().__init__(parent)
        self.original = copy.deepcopy(values)
        self.getters = {}
        layout = W.QFormLayout(self)
        layout.setRowWrapPolicy(W.QFormLayout.RowWrapPolicy.WrapLongRows)
        layout.setFieldGrowthPolicy(W.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for key, value in values.items():
            if key in ("id", "asset", "segments", "face_index", "normal"):
                continue
            factor, label = (
                (1e3, "mm")
                if key in LENGTHS
                else (1e9, "ns") if key in TIMES else (1e-9, "GHz") if key in FREQUENCIES else (1.0, "")
            )
            title = key.replace("_", " ").title() + (f" ({label})" if label else "")
            if key in ("source", "target", "tool", "material") and references and key in references:
                widget = W.QComboBox()
                for identity, name in references[key]:
                    widget.addItem(name, identity)
                widget.setCurrentIndex(widget.findData(value))
                getter = widget.currentData
            elif key in ("direction", "termination", "profile"):
                widget = W.QComboBox()
                widget.addItems(
                    {
                        "direction": ["+", "-"],
                        "termination": ["virtual", "guide_to_pml"],
                        "profile": ["rectangle", "circle", "polyline", "segments"],
                    }[key]
                )
                widget.setCurrentText(value)
                getter = widget.currentText
            elif isinstance(value, bool):
                widget = W.QCheckBox()
                widget.setChecked(value)
                getter = widget.isChecked
            elif isinstance(value, (int, float)):
                widget = EngineeringSpin()
                widget.setDecimals(0 if isinstance(value, int) else 9)
                widget.setRange(-1e12, 1e12)
                widget.setValue(value * factor)
                widget.setKeyboardTracking(False)
                getter = lambda w=widget, f=factor, t=type(value): t(w.value() / f)
            elif isinstance(value, list) and len(value) == 3 and key in LENGTHS | {"axis", "u", "v"}:
                widget = W.QWidget()
                row = W.QHBoxLayout(widget)
                row.setContentsMargins(0, 0, 0, 0)
                inputs = []
                for x in value:
                    spin = EngineeringSpin()
                    spin.setDecimals(9)
                    spin.setRange(-1e12, 1e12)
                    spin.setValue(x * factor)
                    spin.setMinimumWidth(55)
                    spin.setSizePolicy(W.QSizePolicy.Policy.Ignored, W.QSizePolicy.Policy.Fixed)
                    row.addWidget(spin)
                    inputs.append(spin)
                getter = lambda ws=inputs, f=factor: [w.value() / f for w in ws]
            elif isinstance(value, list):
                widget = W.QLineEdit()
                if key == "points":
                    widget.setText("; ".join(", ".join(f"{x*1e3:g}" for x in q) for q in value))
                    title += " (mm; x,y pairs)"
                    getter = lambda w=widget: [
                        [float(x) * 1e-3 for x in p.split(",")] for p in w.text().split(";") if p.strip()
                    ]
                else:
                    widget.setText(", ".join(f"{x*factor:g}" if isinstance(x, (float, int)) else str(x) for x in value))
                    element = type(value[0]) if value else float
                    getter = lambda w=widget, f=factor, t=element: [
                        t(float(x) / f) if t in (int, float) else x.strip() for x in w.text().split(",") if x.strip()
                    ]
            else:
                widget = W.QLineEdit(str(value))
                getter = widget.text
                if key == "anchors":
                    title = "Anchors (auto or Hz, comma-separated)"
                    getter = lambda w=widget: (
                        "auto" if w.text().strip() == "auto" else [float(x) for x in w.text().split(",")]
                    )
            caption = W.QLabel(title)
            caption.setWordWrap(True)
            caption.setMaximumWidth(165)
            layout.addRow(caption, widget)
            self.getters[key] = getter

    def values(self):
        result = copy.deepcopy(self.original)
        for key, getter in self.getters.items():
            result[key] = getter()
        return result


class SketchCanvas(W.QGraphicsView):
    changed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.setScene(W.QGraphicsScene(self))
        self.setSceneRect(-20, -20, 100, 100)
        self.setMinimumSize(430, 350)
        self.points = []
        self.segments = []
        self.pending = []
        self.arc = False
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        p = self.mapToScene(event.position().toPoint())
        q = [round(p.x(), 1) * 1e-3, round(-p.y(), 1) * 1e-3]
        self.points.append(q)
        if not self.pending:
            self.pending = [q]
        elif not self.arc:
            self.segments.append([self.pending[0], q])
            self.pending = [q]
        elif len(self.pending) == 1:
            self.pending.append(q)
        else:
            self.segments.append(self.pending + [q])
            self.pending = [q]
        self.redraw()
        self.changed.emit()

    def redraw(self):
        import numpy as np

        self.scene().clear()
        pen = QtGui.QPen(QtGui.QColor("#1587c2"))
        pen.setCosmetic(True)
        for seg in self.segments:
            points = seg
            if len(seg) == 3:
                a, b, c = np.asarray(seg) * 1e3
                try:
                    center = np.linalg.solve(2 * np.array([b - a, c - a]), np.array([b @ b - a @ a, c @ c - a @ a]))
                    angles = np.arctan2(np.array([a, b, c])[:, 1] - center[1], np.array([a, b, c])[:, 0] - center[0])
                    sweep = (angles[2] - angles[0]) % (2 * np.pi)
                    if (angles[1] - angles[0]) % (2 * np.pi) > sweep:
                        sweep -= 2 * np.pi
                    theta = np.linspace(angles[0], angles[0] + sweep, 40)
                    points = (
                        center + np.linalg.norm(a - center) * np.column_stack((np.cos(theta), np.sin(theta)))
                    ) / 1e3
                except np.linalg.LinAlgError:
                    pass
            path = QtGui.QPainterPath()
            path.moveTo(points[0][0] * 1e3, -points[0][1] * 1e3)
            for q in points[1:]:
                path.lineTo(q[0] * 1e3, -q[1] * 1e3)
            self.scene().addPath(path, pen)
        for q in self.points:
            self.scene().addEllipse(q[0] * 1e3 - 0.3, -q[1] * 1e3 - 0.3, 0.6, 0.6, pen)


class SketchDialog(W.QDialog):
    def __init__(self, parent=None, initial=None):
        super().__init__(parent)
        self.setWindowTitle("Planar sketch")
        layout = W.QVBoxLayout(self)
        parameters = dict(
            profile="rectangle",
            origin=[0.02, 0.02, 0.02],
            u=[1.0, 0.0, 0.0],
            v=[0.0, 1.0, 0.0],
            width=0.02,
            height=0.012,
            radius=0.005,
            points=[],
        )
        parameters.update(initial or {})
        self.properties = Properties(parameters)
        layout.addWidget(self.properties)
        layout.addWidget(
            W.QLabel(
                "For a drawn profile: click start/end points. Arc mode uses midpoint then endpoint.\nCoordinates snap to 0.1 mm. The final edge closes to the start."
            )
        )
        self.canvas = SketchCanvas()
        self.canvas.segments = copy.deepcopy(parameters.get("segments", []))
        if self.canvas.segments:
            self.canvas.points = [segment[0] for segment in self.canvas.segments]
            self.canvas.pending = [self.canvas.segments[-1][-1]]
            self.canvas.redraw()
        layout.addWidget(self.canvas)
        arc = W.QCheckBox("Draw circular arc segments")
        arc.toggled.connect(lambda checked: setattr(self.canvas, "arc", checked))
        layout.addWidget(arc)
        clear = W.QPushButton("Clear drawn profile")

        def reset():
            self.canvas.points = []
            self.canvas.segments = []
            self.canvas.pending = []
            self.canvas.redraw()

        clear.clicked.connect(reset)
        layout.addWidget(clear)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Ok | W.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self):
        p = self.properties.values()
        if self.canvas.segments:
            p["profile"] = "segments"
            p["segments"] = copy.deepcopy(self.canvas.segments)
            first, last = p["segments"][0][0], p["segments"][-1][-1]
            if first != last:
                p["segments"].append([last, first])
        return p

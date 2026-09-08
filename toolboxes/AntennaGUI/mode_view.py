"""Interactive 2D mode maps embedded in Qt."""

import h5py
import numpy as np
from PySide6 import QtWidgets as W
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from .modes import read_mode


class ModeView(W.QDialog):
    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path
        self.setWindowTitle("Modal basis fields")
        self.resize(880, 720)
        layout = W.QVBoxLayout(self)
        controls = W.QHBoxLayout()
        layout.addLayout(controls)
        self.port, self.anchor, self.mode, self.quantity, self.display = [W.QComboBox() for _ in range(5)]
        for title, widget in zip(
            ("Port", "Anchor", "Mode", "Field", "Display"),
            (self.port, self.anchor, self.mode, self.quantity, self.display),
        ):
            controls.addWidget(W.QLabel(title))
            controls.addWidget(widget)
        self.quantity.addItems(["|E|", "|H|", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])
        self.display.addItems(["Magnitude", "Real", "Imaginary", "Phase"])
        self.phase = W.QDoubleSpinBox()
        self.phase.setRange(-360, 360)
        self.phase.setSuffix("° display phase")
        controls.addWidget(self.phase)
        self.vectors = W.QCheckBox("Vectors")
        self.vectors.setChecked(True)
        controls.addWidget(self.vectors)
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas)
        self.diagnostic = W.QLabel()
        self.diagnostic.setWordWrap(True)
        layout.addWidget(self.diagnostic)
        with h5py.File(path) as f:
            self.port.addItems(list(f["ports"]))
        self.port.currentTextChanged.connect(self.populate)
        for widget in (self.anchor, self.mode, self.quantity, self.display):
            widget.currentIndexChanged.connect(self.render)
        self.phase.valueChanged.connect(self.render)
        self.vectors.toggled.connect(self.render)
        self.populate()

    def populate(self):
        for widget in (self.anchor, self.mode):
            widget.blockSignals(True)
            widget.clear()
        with h5py.File(self.path) as f:
            group = f[f"ports/{self.port.currentText()}"]
            self.anchor.addItems([f"{x/1e9:.5g} GHz" for x in group["frequencies"][:]])
            self.mode.addItems([str(x) for x in group["mode_indices"][:]])
        for widget in (self.anchor, self.mode):
            widget.blockSignals(False)
        self.render()

    def render(self, *_):
        if min(self.anchor.currentIndex(), self.mode.currentIndex()) < 0:
            return
        result = read_mode(self.path, self.port.currentText(), self.anchor.currentIndex(), self.mode.currentIndex())
        fields = {k: v * np.exp(1j * np.deg2rad(self.phase.value())) for k, v in result["fields"].items()}
        quantity = self.quantity.currentText()
        family = "H" if "H" in quantity else "E"
        if quantity.startswith("|"):
            values = np.sqrt(sum(np.abs(fields[family + c]) ** 2 for c in "xyz"))
            label = "Magnitude (basis normalization)"
        else:
            value = fields[quantity]
            display = self.display.currentText()
            values = {
                "Magnitude": np.abs,
                "Real": np.real,
                "Imaginary": np.imag,
                "Phase": lambda a: np.angle(a, deg=True),
            }[display](value)
            label = display + (" (degrees)" if display == "Phase" else " (basis normalization)")
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        u, v = result["u"] * 1e3, result["v"] * 1e3
        plot = ax.pcolormesh(u, v, values.T, shading="auto", cmap="viridis")
        self.figure.colorbar(plot, ax=ax, label=label)
        a, b = result["transverse_axes"]
        if self.vectors.isChecked():
            step = max(1, int(max(values.shape) / 22))
            uu, vv = np.meshgrid(u[::step], v[::step], indexing="ij")
            ax.quiver(
                uu,
                vv,
                np.real(fields[family + "xyz"[a]])[::step, ::step],
                np.real(fields[family + "xyz"[b]])[::step, ::step],
                color="white",
            )
        ax.set(
            xlabel=f"{'xyz'[a]} (mm)",
            ylabel=f"{'xyz'[b]} (mm)",
            title=f"Port {self.port.currentText()}, mode {self.mode.currentText()} — {quantity}",
            aspect="equal",
        )
        self.diagnostic.setText(
            f"n_eff = {result['neff']:.6g} | Propagating: {result['propagating']} | Power-valid: {result['valid']} | Tracked reference: {result['reference_valid']}\nPositive-normal modal basis, not driven simulation fields. E and H share the display phase; normalization is retained from the solver."
        )
        self.canvas.draw_idle()

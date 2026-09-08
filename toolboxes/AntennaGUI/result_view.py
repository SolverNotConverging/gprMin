"""Interactive S-parameter and radiation plots from completed solver artifacts."""

import h5py
import numpy as np
from PySide6 import QtWidgets as W
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .results import read_sparameters, read_farfield, angular_grid, polar_cut


class ResultView(W.QDialog):
    def __init__(self, path, record, parent=None):
        super().__init__(parent)
        self.path, self.record = path, record
        self.setWindowTitle(record["name"])
        self.resize(950, 760)
        layout = W.QVBoxLayout(self)
        self.controls = W.QGridLayout()
        layout.addLayout(self.controls)
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas, 1)
        self.note = W.QLabel()
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        if record["kind"] == "sparameters":
            self.traces = read_sparameters(path)
            self.trace = self.combo("Trace", ["All traces", *self.traces])
            self.quantity = self.combo(
                "Display", ["Magnitude (dB)", "Magnitude", "Phase (degrees)", "Real", "Imaginary"]
            )
            self.validity = self.combo("Validity", ["Power waves", "Coefficients"])
        else:
            with h5py.File(path) as f:
                group = f[record["group"]]
                self.frequencies = group.parent.parent["frequencies"][:]
                quantities = list(group["fields"])
                self.phis = np.unique(group["phi"][:])
                self.thetas = np.unique(group["theta"][:])
            self.frequency = self.combo("Frequency", [f"{v/1e9:g} GHz" for v in self.frequencies])
            self.quantity = self.combo("Quantity", quantities)
            self.quantity.setCurrentText("directivity_dbi")
            self.view = self.combo("View", ["Angular map", "Theta cut (polar)", "3D surface", "Phi cut (polar)"])
            self.phi = self.combo("Fixed φ", [f"{v:g}°" for v in self.phis])
            self.theta = self.combo("Fixed θ", [f"{v:g}°" for v in self.thetas])
            self.theta.setCurrentIndex(int(np.argmin(abs(self.thetas - 90))))
        for i in range(self.controls.count()):
            widget = self.controls.itemAt(i).widget()
            if isinstance(widget, W.QComboBox):
                widget.currentIndexChanged.connect(self.draw)
        self.draw()

    def combo(self, label, items):
        pair = self.controls.count() // 2
        row, column = pair // 3, 2 * (pair % 3)
        self.controls.addWidget(W.QLabel(label), row, column)
        widget = W.QComboBox()
        widget.addItems(items)
        widget.setMinimumWidth(90)
        widget.setSizePolicy(W.QSizePolicy.Policy.Ignored, W.QSizePolicy.Policy.Fixed)
        self.controls.addWidget(widget, row, column + 1)
        self.controls.setColumnStretch(column + 1, 1)
        return widget

    def draw(self, *_):
        self.figure.clear()
        if self.record["kind"] == "sparameters":
            ax = self.figure.add_subplot()
            omitted = 0
            for label, trace in self.traces.items():
                if self.trace.currentIndex() and label != self.trace.currentText():
                    continue
                z = trace["value"]
                valid = trace["coefficient_valid"] & np.isfinite(z)
                if self.validity.currentIndex() == 0:
                    valid &= trace["power_wave_valid"]
                with np.errstate(divide="ignore"):
                    values = [20 * np.log10(np.abs(z)), np.abs(z), np.angle(z, deg=True), z.real, z.imag][
                        self.quantity.currentIndex()
                    ]
                ax.plot(trace["frequency"] / 1e9, np.where(valid, values, np.nan), label=label)
                omitted += int((~valid).sum())
            ax.set(xlabel="Frequency (GHz)", ylabel=self.quantity.currentText())
            ax.grid(True, alpha=0.3)
            ax.legend()
            self.note.setText(
                f"{omitted} invalid samples omitted. A single excitation yields one S-matrix column; simultaneous drives yield active S-parameters."
            )
        else:
            self.draw_farfield()
        self.canvas.draw_idle()

    def draw_farfield(self):
        self.phi.setEnabled(self.view.currentIndex() == 1)
        self.theta.setEnabled(self.view.currentIndex() == 3)
        q, index = self.quantity.currentText(), self.frequency.currentIndex()
        data = read_farfield(self.path, self.record["group"], q, index)
        values = data["value"]
        if np.iscomplexobj(values):
            values = np.abs(values)
        label = "|" + q + "| (solver field units)" if q.startswith("E") else q
        if not np.isfinite(values).any():
            ax = self.figure.add_subplot()
            ax.axis("off")
            ax.text(0.5, 0.5, "No valid samples at this frequency", ha="center", va="center")
        elif values.ndim == 0:
            ax = self.figure.add_subplot()
            ax.axis("off")
            ax.text(0.5, 0.5, f"{q}\n{float(values):.5g}", ha="center", va="center", fontsize=22)
        elif self.view.currentIndex() in (1, 3):
            ax = self.figure.add_subplot(projection="polar")
            is_theta = self.view.currentIndex() == 1
            fixed = self.phis[self.phi.currentIndex()] if is_theta else self.thetas[self.theta.currentIndex()]
            angles, cut = polar_cut(data["theta"], data["phi"], values, "theta" if is_theta else "phi", fixed)
            ax.plot(angles, cut)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_thetamin(0)
            ax.set_thetamax(360)
            finite = cut[np.isfinite(cut)]
            if len(finite):
                lower = np.floor(finite.min() / 5) * 5 if q.endswith("_dbi") else 0
                upper = np.ceil(finite.max() / 5) * 5 if q.endswith("_dbi") else finite.max() * 1.05
                ax.set_ylim(lower, max(upper, lower + 1))
            title = f"θ cut · φ = {fixed:g}° / {(fixed+180)%360:g}°" if is_theta else f"φ cut · θ = {fixed:g}°"
            ax.set_title(title + "\n" + label + " · " + self.frequency.currentText(), pad=20)
            ax.grid(True, alpha=0.3)
        elif self.view.currentIndex() == 2:
            ax = self.figure.add_subplot(projection="3d")
            theta, phi, grid = angular_grid(data["theta"], data["phi"], values)
            grid = np.column_stack((grid, grid[:, 0]))
            phi = np.r_[phi, phi[0] + 360]
            phi, theta = np.meshgrid(np.deg2rad(phi), np.deg2rad(theta))
            finite = np.isfinite(grid)
            radius = np.full_like(grid, np.nan)
            if finite.any():
                if q.endswith("_dbi"):
                    radius[finite] = 10 ** ((grid[finite] - np.max(grid[finite])) / 10)
                else:
                    peak = np.max(np.abs(grid[finite]))
                    radius[finite] = np.abs(grid[finite]) / peak if peak else 0
            norm = Normalize(vmin=np.nanmin(grid), vmax=np.nanmax(grid))
            cmap = colormaps["viridis"]
            ax.plot_surface(
                radius * np.sin(theta) * np.cos(phi),
                radius * np.sin(theta) * np.sin(phi),
                radius * np.cos(theta),
                facecolors=cmap(norm(grid)),
                rstride=1,
                cstride=1,
                linewidth=0,
                antialiased=True,
                shade=False,
            )
            self.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=label, shrink=0.7)
            ax.set(xlabel="x", ylabel="y", zlabel="z", title="Radius: normalized linear magnitude/power")
            ax.set_box_aspect((1, 1, 1))
        else:
            ax = self.figure.add_subplot()
            artist = ax.scatter(data["phi"], data["theta"], c=values, marker="s", s=24, cmap="viridis")
            self.figure.colorbar(artist, ax=ax, label=label)
            ax.set(xlabel="φ (degrees)", ylabel="θ (degrees)", title=self.frequency.currentText())
        self.note.setText(
            "Angles use the solver's global spherical coordinates. "
            + (
                "Theta cuts join opposite azimuths to cover 360°; unsampled opposite azimuths are interpolated."
                if data["valid"]
                else "Power normalization is invalid at this frequency; values are omitted."
            )
        )

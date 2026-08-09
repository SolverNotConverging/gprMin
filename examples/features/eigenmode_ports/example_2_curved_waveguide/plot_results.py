"""Plot S-parameters and transient fields for the curved waveguide."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, Wedge
import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = EXAMPLE_DIR / "curved_waveguide"
SPARAMETER_PLOT_PATH = EXAMPLE_DIR / "curved_waveguide_sparameters.png"
FIELD_PLOT_PATH = EXAMPLE_DIR / "curved_waveguide_field_propagation.png"
PLOT_FLOOR_DB = -100.0
DIELECTRIC_COLOR = "#969696"
PORT_COLOR = "#d62728"
FIELD_ALPHA_FLOOR = 0.08


def field_overlay_alpha(values, limit):
    """Keep geometry visible where the common-scale field is weak."""

    normalized = np.clip(np.abs(values) / limit, 0.0, 1.0)
    return FIELD_ALPHA_FLOOR + (1.0 - FIELD_ALPHA_FLOOR) * np.sqrt(normalized)


def draw_material_background(axis):
    """Draw the straight and annular dielectric guide sections in millimetres."""

    style = {
        "facecolor": DIELECTRIC_COLOR,
        "edgecolor": "none",
        "zorder": 0,
    }
    axis.add_patch(Rectangle((0.0, 35.0), 100.0, 20.0, **style))
    axis.add_patch(
        Wedge(
            (100.0, 60.0),
            25.0,
            270.0,
            360.0,
            width=20.0,
            **style,
        )
    )
    axis.add_patch(Rectangle((105.0, 60.0), 20.0, 105.0, **style))


def draw_port_overlays(axis):
    """Mark both orthogonal eigenmode planes over their actual apertures."""

    axis.plot(
        (20.0, 20.0),
        (5.0, 85.0),
        color=PORT_COLOR,
        linestyle="--",
        linewidth=1.2,
        zorder=3,
    )
    axis.text(
        21.5,
        83.5,
        "P1 (+x)",
        color=PORT_COLOR,
        fontsize=7,
        rotation=90,
        ha="left",
        va="top",
        zorder=3,
    )
    axis.plot(
        (75.0, 155.0),
        (160.0, 160.0),
        color=PORT_COLOR,
        linestyle="--",
        linewidth=1.2,
        zorder=3,
    )
    axis.text(
        76.5,
        158.5,
        "P2 (-y)",
        color=PORT_COLOR,
        fontsize=7,
        ha="left",
        va="top",
        zorder=3,
    )


def read_sparameters(stem: Path):
    """Return valid CSV rows grouped by source/destination port and mode."""

    path = stem.with_name(stem.name + "_sparameters.csv")
    traces = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not bool(int(row["valid"])):
                continue
            key = (
                int(row["source_port"]),
                int(row["source_mode"]),
                int(row["destination_port"]),
                int(row["destination_mode"]),
            )
            traces[key].append((float(row["frequency_hz"]) * 1e-9, float(row["S_magnitude_db"])))
    if not traces:
        raise ValueError(f"No valid S-parameter rows found in {path}")
    return {key: np.asarray(sorted(values), dtype=np.float64) for key, values in traces.items()}


def plot_sparameters(axis, traces):
    """Plot every valid source/destination port and mode combination."""

    for (
        source_port,
        source_mode,
        destination_port,
        destination_mode,
    ), data in sorted(traces.items()):
        axis.plot(
            data[:, 0],
            np.maximum(data[:, 1], PLOT_FLOOR_DB),
            marker="o",
            markersize=3,
            linewidth=1.7,
            label=(f"S{destination_port}{source_port}, " f"mode {destination_mode}<-{source_mode}"),
        )
    axis.set_title("Reflection, transmission, and modal conversion")
    axis.set_xlabel("Frequency (GHz)")
    axis.set_ylabel(f"Magnitude (dB; floor {PLOT_FLOOR_DB:g} dB)")
    axis.set_ylim(PLOT_FLOOR_DB, 5)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize="small", ncols=2)


def read_field_snapshots(stem: Path, field: str = "Ez"):
    """Read adjacent 2D snapshots in physical-time order."""

    snapshot_dir = stem.with_name(stem.name + "_snaps")
    paths = list(snapshot_dir.glob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"No snapshots found in {snapshot_dir}; run the model before plotting")

    snapshots = []
    for path in paths:
        with h5py.File(path, "r") as output:
            if field not in output:
                raise KeyError(f"Snapshot {path} does not contain field {field!r}")
            values = np.squeeze(output[field][...])
            if values.ndim != 2:
                raise ValueError(f"Snapshot {path} field {field!r} is not two-dimensional")
            spacing = np.asarray(output.attrs["dx_dy_dz"], dtype=np.float64)
            extent = (
                0.0,
                values.shape[0] * spacing[0] * 1e3,
                0.0,
                values.shape[1] * spacing[1] * 1e3,
            )
            snapshots.append((float(output.attrs["time"]) * 1e9, values.T, extent))
    return sorted(snapshots, key=lambda snapshot: snapshot[0])


def plot_field_snapshots(stem: Path, plot_path: Path, field: str = "Ez"):
    """Write one common-scale panel for each requested snapshot time."""

    snapshots = read_field_snapshots(stem, field)
    limit = max(float(np.max(np.abs(values))) for _, values, _ in snapshots)
    if not np.isfinite(limit) or limit == 0:
        raise ValueError(f"Snapshots contain no finite non-zero {field} values")

    columns = min(4, (len(snapshots) + 1) // 2)
    rows = int(np.ceil(len(snapshots) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.2 * columns, 3.5 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    active_axes = []
    for axis, (time_ns, values, extent) in zip(axes.flat, snapshots):
        active_axes.append(axis)
        draw_material_background(axis)
        image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
            aspect="equal",
            alpha=field_overlay_alpha(values, limit),
            zorder=1,
        )
        draw_port_overlays(axis)
        axis.set_title(f"t = {time_ns:.2f} ns")
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
    for axis in axes.flat[len(snapshots) :]:
        axis.set_visible(False)
    active_axes[0].legend(
        handles=[
            Patch(
                facecolor=DIELECTRIC_COLOR,
                edgecolor="none",
                label="Dielectric core",
            ),
            Line2D(
                (0,),
                (0,),
                color=PORT_COLOR,
                linestyle="--",
                linewidth=1.2,
                label="Eigenmode port",
            ),
        ],
        loc="upper right",
        fontsize="small",
    )
    figure.colorbar(image, ax=active_axes, label=f"{field} (V/m)")
    figure.suptitle("Curved dielectric waveguide: Ez propagation")
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    print(f"Wrote {plot_path}")


def main():
    traces = read_sparameters(OUTPUT_STEM)
    figure, axis = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    plot_sparameters(axis, traces)
    figure.savefig(SPARAMETER_PLOT_PATH, dpi=180)
    plt.close(figure)
    print(f"Wrote {SPARAMETER_PLOT_PATH}")
    plot_field_snapshots(OUTPUT_STEM, FIELD_PLOT_PATH)


if __name__ == "__main__":
    main()

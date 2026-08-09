"""Plot S-parameters and fields for the PML-free matched microstrip."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = EXAMPLE_DIR / "matched_waveguide"
SPARAMETER_PLOT_PATH = EXAMPLE_DIR / "matched_waveguide_sparameters.png"
FIELD_PLOT_PATH = EXAMPLE_DIR / "matched_waveguide_field_propagation.png"

DOMAIN_MM = (300.0, 12.0, 6.0)
PORTS = (
    (3.0, 0.5, 6.0, "P1 active (+x)"),
    (297.0, 0.5, 6.0, "P2 passive (-x)"),
)
MATCHED_DEPTH_REGIONS_MM = ((0.0, 3.0), (297.0, 300.0))
MATCHED_BOUNDARY_INDICES = {1: 0, 2: 600}
MATCH_DEPTH_CELLS = 6
MATCHED_FORMULATION = "PowerAdjointModalAdmittanceADE"
FIELD_CUT_MM = 6.25

PLOT_FLOOR_DB = -40.0
FIELD_ALPHA_FLOOR = 0.10
DIELECTRIC_COLOR = "#b7b7b7"
PEC_COLOR = "#f2c94c"
MATCHED_DEPTH_COLOR = "#78b7d6"
MATCHED_BOUNDARY_COLOR = "#21618c"
PORT_COLOR = "#d62728"
AXIS_NAMES = ("x", "y", "z")
SNAPSHOT_STAGES = (
    "launch",
    "guided pulse travels",
    "guided pulse travels",
    "leading edge reaches P2",
    "packet spans the guide",
    "packet exits at P2",
    "main pulse absorbed",
    "late-time ring-down",
)


def field_overlay_alpha(values, limit):
    """Keep the material drawing visible where the common-scale field is weak."""

    normalized = np.clip(np.abs(values) / limit, 0.0, 1.0)
    return FIELD_ALPHA_FLOOR + (1.0 - FIELD_ALPHA_FLOOR) * np.sqrt(normalized)


def draw_microstrip_background(axis):
    """Draw the material cut and the ordinary matched-port depth regions."""

    axis.add_patch(
        Rectangle(
            (0.0, 0.5),
            DOMAIN_MM[0],
            1.5,
            facecolor=DIELECTRIC_COLOR,
            edgecolor="none",
            zorder=0,
        )
    )
    axis.add_patch(
        Rectangle(
            (0.0, 0.0),
            DOMAIN_MM[0],
            0.5,
            facecolor=PEC_COLOR,
            edgecolor="none",
            zorder=0.1,
        )
    )
    # The x-z cut passes through the centre of the 2 mm-wide strip.
    axis.add_patch(
        Rectangle(
            (0.0, 2.0),
            DOMAIN_MM[0],
            0.5,
            facecolor=PEC_COLOR,
            edgecolor="none",
            zorder=0.1,
        )
    )
    for start, stop in MATCHED_DEPTH_REGIONS_MM:
        axis.axvspan(
            start,
            stop,
            facecolor=MATCHED_DEPTH_COLOR,
            alpha=0.20,
            edgecolor="none",
            zorder=0.2,
        )


def draw_matched_port_overlays(axis, *, annotate=False):
    """Mark reference planes and ADE faces, with optional detailed labels."""

    for x, z_min, z_max, label in PORTS:
        axis.plot(
            (x, x),
            (z_min, z_max),
            color=PORT_COLOR,
            linestyle="--",
            linewidth=1.4,
            zorder=3,
        )
        if annotate:
            text_x = x + 4.0 if x < DOMAIN_MM[0] / 2 else x - 4.0
            axis.text(
                text_x,
                z_max - 0.15,
                label,
                color=PORT_COLOR,
                fontsize=7,
                rotation=0,
                ha="left" if x < DOMAIN_MM[0] / 2 else "right",
                va="top",
                zorder=3,
            )

    for x, horizontal_alignment in (
        (0.0, "left"),
        (DOMAIN_MM[0], "right"),
    ):
        axis.axvline(
            x,
            color=MATCHED_BOUNDARY_COLOR,
            linewidth=2.5,
            zorder=3,
            clip_on=False,
        )
        if annotate:
            offset = 0.25 if horizontal_alignment == "left" else -0.25
            axis.text(
                x + offset,
                3.25,
                "ADE boundary",
                color=MATCHED_BOUNDARY_COLOR,
                fontsize=7,
                rotation=90,
                ha=horizontal_alignment,
                va="center",
                zorder=3,
            )

    if not annotate:
        return
    for region_index, (start, stop) in enumerate(MATCHED_DEPTH_REGIONS_MM):
        left_region = region_index == 0
        text_x = 18.0 if left_region else DOMAIN_MM[0] - 18.0
        axis.annotate(
            r"$d=6$ cells",
            xy=(0.5 * (start + stop), 5.10),
            xytext=(text_x, 5.10),
            arrowprops={
                "arrowstyle": "->",
                "color": MATCHED_BOUNDARY_COLOR,
                "linewidth": 1.0,
            },
            color=MATCHED_BOUNDARY_COLOR,
            fontsize=7,
            ha="left" if left_region else "right",
            va="center",
            zorder=3,
        )


def read_sparameters(stem: Path):
    """Return valid mode-1 S11 and S21 rows from the generated CSV file."""

    path = stem.with_name(stem.name + "_sparameters.csv")
    traces = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not bool(int(row["valid"])):
                continue
            if int(row["source_port"]) != 1 or int(row["source_mode"]) != 1:
                continue
            if int(row["destination_mode"]) != 1:
                continue
            destination = int(row["destination_port"])
            traces[destination].append(
                (
                    float(row["frequency_hz"]) * 1e-9,
                    float(row["S_magnitude_db"]),
                )
            )
    if not traces:
        raise ValueError(f"No valid mode-1 S-parameter rows found in {path}")
    ordered = {
        destination: np.asarray(sorted(values), dtype=np.float64)
        for destination, values in traces.items()
    }
    missing = {1, 2}.difference(ordered)
    if missing:
        raise ValueError(
            f"Missing valid mode-1 S-parameters for destination port(s) "
            f"{sorted(missing)} in {path}"
        )
    return ordered


def validate_matched_output(stem: Path):
    """Confirm that the output used two modal-admittance ADE boundaries."""

    with h5py.File(stem.with_suffix(".h5"), "r") as output:
        for port_number in (1, 2):
            port = output[f"eigenmode_ports/port{port_number}"]
            if not bool(port.attrs.get("Matched", False)):
                raise ValueError(f"Output port {port_number} is not matched")
            if int(port.attrs["MatchDepthCells"]) != MATCH_DEPTH_CELLS:
                raise ValueError(
                    f"Output port {port_number} does not use the example's "
                    f"{MATCH_DEPTH_CELLS}-cell uniform depth"
                )
            boundary_index = int(port.attrs["MatchedBoundaryIndex"])
            if boundary_index != MATCHED_BOUNDARY_INDICES[port_number]:
                raise ValueError(
                    f"Output port {port_number} terminates boundary index "
                    f"{boundary_index}, expected "
                    f"{MATCHED_BOUNDARY_INDICES[port_number]}"
                )
            if tuple(port.attrs["ModeIndices"]) != (1,):
                raise ValueError(
                    f"Output port {port_number} does not match only mode 1"
                )
            if port.attrs["MatchedFormulation"] != MATCHED_FORMULATION:
                raise ValueError(
                    f"Output port {port_number} uses an unexpected matched "
                    "formulation"
                )
            time_constant = np.asarray(
                port.attrs["MatchedModalHalfCellTimeConstant"],
                dtype=np.float64,
            )
            if time_constant.shape != (1,) or not np.all(time_constant > 0):
                raise ValueError(
                    f"Output port {port_number} has an invalid ADE time constant"
                )


def plot_sparameters(axis, traces):
    """Plot reflection and transmission of the launched quasi-TEM mode."""

    styles = {1: (r"$S_{11}$: reflected", "o"), 2: (r"$S_{21}$: transmitted", "s")}
    for destination, data in sorted(traces.items()):
        label, marker = styles[destination]
        axis.plot(
            data[:, 0],
            np.maximum(data[:, 1], PLOT_FLOOR_DB),
            marker=marker,
            markersize=4,
            linewidth=2,
            label=label,
        )
    axis.axhline(0, color="0.35", linestyle=":", linewidth=1)
    axis.set_title("Quasi-TEM matched-port response")
    axis.set_xlabel("Frequency (GHz)")
    axis.set_ylabel(f"Magnitude (dB; floor {PLOT_FLOOR_DB:g} dB)")
    axis.set_ylim(PLOT_FLOOR_DB, 2)
    axis.grid(True, alpha=0.3)
    axis.legend()


def _extract_2d_plane(raw_values, spacing, path):
    """Squeeze a one-cell-thick 3D snapshot and retain its physical axes."""

    if raw_values.ndim != 3:
        raise ValueError(f"Snapshot {path} field array is not three-dimensional")
    plane_axes = tuple(index for index, size in enumerate(raw_values.shape) if size > 1)
    if len(plane_axes) != 2:
        raise ValueError(
            f"Snapshot {path} must be one cell thick along exactly one axis; "
            f"got shape {raw_values.shape}"
        )
    values = np.squeeze(raw_values)
    horizontal_axis, vertical_axis = plane_axes
    extent = (
        0.0,
        values.shape[0] * spacing[horizontal_axis] * 1e3,
        0.0,
        values.shape[1] * spacing[vertical_axis] * 1e3,
    )
    labels = (AXIS_NAMES[horizontal_axis], AXIS_NAMES[vertical_axis])
    return values.T, extent, labels


def read_field_snapshots(stem: Path, field: str = "Ez", *, include_axes=False):
    """Read one-cell-thick snapshots in physical-time order."""

    snapshot_dir = stem.with_name(stem.name + "_snaps")
    paths = list(snapshot_dir.glob("*.h5"))
    if not paths:
        raise FileNotFoundError(
            f"No snapshots found in {snapshot_dir}; run the model before plotting"
        )

    snapshots = []
    for path in paths:
        with h5py.File(path, "r") as output:
            if field not in output:
                raise KeyError(f"Snapshot {path} does not contain field {field!r}")
            spacing = np.asarray(output.attrs["dx_dy_dz"], dtype=np.float64)
            values, extent, labels = _extract_2d_plane(
                np.asarray(output[field][...]),
                spacing,
                path,
            )
            snapshot = (float(output.attrs["time"]) * 1e9, values, extent)
            if include_axes:
                snapshot += labels
            snapshots.append(snapshot)
    return sorted(snapshots, key=lambda snapshot: snapshot[0])


def plot_field_snapshots(stem: Path, plot_path: Path, field: str = "Ez"):
    """Plot the pulse on an x-z material cut with one common field scale."""

    snapshots = read_field_snapshots(stem, field, include_axes=True)
    plane_labels = {(horizontal, vertical) for *_, horizontal, vertical in snapshots}
    if plane_labels != {("x", "z")}:
        raise ValueError(
            f"Expected x-z microstrip snapshots, found planes {sorted(plane_labels)}"
        )
    limit = max(float(np.max(np.abs(values))) for _, values, _, _, _ in snapshots)
    if not np.isfinite(limit) or limit == 0:
        raise ValueError(f"Snapshots contain no finite non-zero {field} values")

    columns = min(2, len(snapshots))
    rows = int(np.ceil(len(snapshots) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(7.2 * columns, 2.5 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    active_axes = []
    for snapshot_index, (
        axis,
        (time_ns, values, extent, horizontal, vertical),
    ) in enumerate(zip(axes.flat, snapshots)):
        active_axes.append(axis)
        draw_microstrip_background(axis)
        image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
            # The 50:1 x-z geometry needs an exaggerated vertical scale for
            # the substrate, strip, and modal field to remain legible.
            aspect="auto",
            alpha=field_overlay_alpha(values, limit),
            zorder=1,
        )
        draw_matched_port_overlays(axis, annotate=snapshot_index == 0)
        axis.set_xlim(0.0, DOMAIN_MM[0])
        axis.set_ylim(0.0, DOMAIN_MM[2])
        stage = (
            SNAPSHOT_STAGES[snapshot_index]
            if snapshot_index < len(SNAPSHOT_STAGES)
            else ""
        )
        title_suffix = f" — {stage}" if stage else ""
        axis.set_title(f"t = {time_ns:.2f} ns{title_suffix}")
        axis.set_xlabel(f"{horizontal} (mm)")
        axis.set_ylabel(f"{vertical} (mm)")
    for axis in axes.flat[len(snapshots) :]:
        axis.set_visible(False)

    active_axes[0].legend(
        handles=[
            Patch(
                facecolor=DIELECTRIC_COLOR,
                edgecolor="none",
                label=r"Dielectric ($\epsilon_r=4.4$)",
            ),
            Patch(facecolor=PEC_COLOR, edgecolor="none", label="PEC ground and strip"),
            Patch(
                facecolor=MATCHED_DEPTH_COLOR,
                edgecolor="none",
                alpha=0.20,
                label="Uniform 6-cell match depth",
            ),
            Line2D(
                (0,),
                (0,),
                color=PORT_COLOR,
                linestyle="--",
                linewidth=1.4,
                label="Port reference plane",
            ),
            Line2D(
                (0,),
                (0,),
                color=MATCHED_BOUNDARY_COLOR,
                linewidth=2.5,
                label="Matched ADE boundary",
            ),
        ],
        loc="upper center",
        fontsize=6.5,
        framealpha=0.92,
    )
    figure.colorbar(image, ax=active_axes, label=f"{field} (V/m)")
    figure.suptitle(
        f"PML-free 300 mm matched microstrip: {field} on y = "
        f"{FIELD_CUT_MM:g} mm (z scale exaggerated)"
    )
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    print(f"Wrote {plot_path}")


def main():
    validate_matched_output(OUTPUT_STEM)
    traces = read_sparameters(OUTPUT_STEM)
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    plot_sparameters(axis, traces)
    figure.suptitle("Two matched microstrip terminations; no PML")
    figure.savefig(SPARAMETER_PLOT_PATH, dpi=180)
    plt.close(figure)
    print(f"Wrote {SPARAMETER_PLOT_PATH}")
    print(
        f"Worst |S11| = {np.max(traces[1][:, 1]):.2f} dB; "
        f"minimum |S21| = {np.min(traces[2][:, 1]):.2f} dB"
    )
    plot_field_snapshots(OUTPUT_STEM, FIELD_PLOT_PATH)


if __name__ == "__main__":
    main()

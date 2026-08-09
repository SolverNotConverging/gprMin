"""Plot S-parameters and fields for the PML-free matched waveguide."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = EXAMPLE_DIR / "matched_waveguide"
SPARAMETER_PLOT_PATH = EXAMPLE_DIR / "matched_waveguide_sparameters.png"
FIELD_PLOT_PATH = EXAMPLE_DIR / "matched_waveguide_field_propagation.png"
PLOT_FLOOR_DB = -80.0
MATCHED_BUFFERS_MM = ((0.0, 10.0), (110.0, 120.0))
MATCHED_BOUNDARY_INDICES = {1: 0, 2: 120}


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
    """Confirm that the result came from two matched modal boundaries."""

    with h5py.File(stem.with_suffix(".h5"), "r") as output:
        for port_number in (1, 2):
            port = output[f"eigenmode_ports/port{port_number}"]
            if not bool(port.attrs.get("Matched", False)):
                raise ValueError(f"Output port {port_number} is not matched")
            if int(port.attrs["MatchDepthCells"]) != 10:
                raise ValueError(
                    f"Output port {port_number} does not use the example's "
                    "10-cell translation buffer"
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
            if (
                port.attrs["MatchedFormulation"]
                != "Alimenti2000NumericalModalTranslation"
            ):
                raise ValueError(
                    f"Output port {port_number} uses an unexpected matched "
                    "formulation"
                )


def plot_sparameters(axis, traces):
    """Plot the reflection and transmission of the launched mode."""

    styles = {1: ("S11: reflected", "o"), 2: ("S21: transmitted", "s")}
    for destination, data in sorted(traces.items()):
        label, marker = styles.get(
            destination,
            (f"S{destination}1", "o"),
        )
        axis.plot(
            data[:, 0],
            np.maximum(data[:, 1], PLOT_FLOOR_DB),
            marker=marker,
            markersize=4,
            linewidth=2,
            label=label,
        )
    axis.axhline(0, color="0.35", linestyle=":", linewidth=1)
    axis.set_title("Uniform-guide matched-port response")
    axis.set_xlabel("Frequency (GHz)")
    axis.set_ylabel(f"Magnitude (dB; floor {PLOT_FLOOR_DB:g} dB)")
    axis.set_ylim(PLOT_FLOOR_DB, 5)
    axis.grid(True, alpha=0.3)
    axis.legend()


def read_field_snapshots(stem: Path, field: str = "Ez"):
    """Read adjacent 2D snapshots in physical-time order."""

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
            values = np.squeeze(output[field][...])
            if values.ndim != 2:
                raise ValueError(
                    f"Snapshot {path} field {field!r} is not two-dimensional"
                )
            spacing = np.asarray(output.attrs["dx_dy_dz"], dtype=np.float64)
            extent = (
                0.0,
                values.shape[0] * spacing[0] * 1e3,
                0.0,
                values.shape[1] * spacing[1] * 1e3,
            )
            snapshots.append(
                (float(output.attrs["time"]) * 1e9, values.T, extent)
            )
    return sorted(snapshots, key=lambda snapshot: snapshot[0])


def plot_field_snapshots(stem: Path, plot_path: Path, field: str = "Ez"):
    """Plot the pulse and both translation buffers on one common field scale."""

    snapshots = read_field_snapshots(stem, field)
    limit = max(float(np.max(np.abs(values))) for _, values, _ in snapshots)
    if not np.isfinite(limit) or limit == 0:
        raise ValueError(f"Snapshots contain no finite non-zero {field} values")

    columns = min(4, len(snapshots))
    rows = int(np.ceil(len(snapshots) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.2 * columns, 3.25 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    active_axes = []
    for axis, (time_ns, values, extent) in zip(axes.flat, snapshots):
        active_axes.append(axis)
        image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
            aspect="equal",
        )
        for start, stop in MATCHED_BUFFERS_MM:
            axis.axvspan(start, stop, color="0.5", alpha=0.12)
        axis.axvline(10, color="0.25", linestyle="--", linewidth=0.8)
        axis.axvline(110, color="0.25", linestyle="--", linewidth=0.8)
        axis.set_title(f"t = {time_ns:.2f} ns")
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
    for axis in axes.flat[len(snapshots) :]:
        axis.set_visible(False)
    figure.colorbar(image, ax=active_axes, label=f"{field} (V/m)")
    figure.suptitle("PML-free matched guide: uniform translation buffers")
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    print(f"Wrote {plot_path}")


def main():
    validate_matched_output(OUTPUT_STEM)
    traces = read_sparameters(OUTPUT_STEM)
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    plot_sparameters(axis, traces)
    figure.suptitle("Two matched eigenmode terminations; no PML")
    figure.savefig(SPARAMETER_PLOT_PATH, dpi=180)
    plt.close(figure)
    print(f"Wrote {SPARAMETER_PLOT_PATH}")
    plot_field_snapshots(OUTPUT_STEM, FIELD_PLOT_PATH)


if __name__ == "__main__":
    main()

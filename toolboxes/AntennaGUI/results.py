"""Read solver results without importing Qt or recomputing physics."""

import csv
from pathlib import Path

import h5py
import numpy as np


def angular_grid(theta, phi, values):
    """Order sampled directions into a theta-by-phi grid, retaining missing samples."""
    theta, ti = np.unique(theta, return_inverse=True)
    phi, pi = np.unique(phi, return_inverse=True)
    grid = np.full((len(theta), len(phi)), np.nan)
    grid[ti, pi] = values
    return theta, phi, grid


def polar_cut(theta, phi, values, kind, angle):
    """Full-circle cuts; theta's second half uses the opposite azimuth plane."""
    theta, phi, grid = angular_grid(theta, phi, values)
    if kind == "phi":
        row = grid[np.argmin(abs(theta - angle))]
        return np.deg2rad(np.r_[phi, phi[0] + 360]), np.r_[row, row[0]]

    def column(azimuth):
        # Periodic interpolation also supports grids whose step does not divide 180.
        extended = np.r_[phi[-1] - 360, phi, phi[0] + 360]
        samples = np.column_stack((grid[:, -1], grid, grid[:, 0]))
        return np.array([np.interp(azimuth % 360, extended, row) for row in samples])

    front, back = column(angle), column(angle + 180)
    interior = (theta > 0) & (theta < 180)
    degrees = np.r_[theta, 360 - theta[interior][::-1], 360]
    return np.deg2rad(degrees), np.r_[front, back[interior][::-1], front[0]]


def discover_results(directory):
    directory = Path(directory)
    plots = []
    for path in sorted(directory.glob("antenna*parameters.csv")):
        plots.append(
            dict(
                kind="sparameters",
                file=path.name,
                name="Active S-parameters" if "active_" in path.name else "S-parameters",
            )
        )
    path = directory / "antenna.h5"
    if path.exists():
        with h5py.File(path) as f:

            def visit(name, obj):
                if isinstance(obj, h5py.Group) and "/far_field/" in name and "fields" in obj and "theta" in obj:
                    plots.append(
                        dict(kind="farfield", file=path.name, group=name, name="Far field · " + name.split("/")[1])
                    )

            f.visititems(visit)
    return plots


def read_sparameters(path):
    """Preserve coefficient and power-wave validity independently."""
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    traces = {}
    for row in rows:
        active = "active_S_real" in row
        if active:
            label = f"Active S · port {row['port']} mode {row['mode']}"
        else:
            label = (
                f"S · {row['destination_port']}:{row['destination_mode']} ← {row['source_port']}:{row['source_mode']}"
            )
        prefix = "active_S" if active else "S"
        traces.setdefault(label, []).append(
            (
                float(row["frequency_hz"]),
                complex(float(row[prefix + "_real"]), float(row[prefix + "_imag"])),
                bool(int(row["coefficient_valid"])),
                bool(int(row["power_wave_valid"])),
            )
        )
    result = {}
    for label, samples in traces.items():
        samples.sort(key=lambda item: item[0])
        result[label] = dict(
            frequency=np.array([r[0] for r in samples]),
            value=np.array([r[1] for r in samples]),
            coefficient_valid=np.array([r[2] for r in samples]),
            power_wave_valid=np.array([r[3] for r in samples]),
        )
    if not result:
        raise ValueError("No S-parameter samples in this result")
    return result


def read_farfield(path, group, quantity, frequency):
    with h5py.File(path) as f:
        g = f[group]
        values = np.asarray(g["fields"][quantity][frequency]).copy()
        mask_name = {
            "gain_dbi": "gain_valid",
            "radiation_efficiency": "gain_valid",
            "realized_gain_dbi": "realized_gain_valid",
            "total_efficiency": "realized_gain_valid",
        }.get(quantity)
        valid = mask_name is None or bool(g["port_power"][mask_name][frequency])
        if not valid:
            values[...] = np.nan
        return dict(theta=g["theta"][:], phi=g["phi"][:], value=values, valid=valid)

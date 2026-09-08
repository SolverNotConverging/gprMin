"""Modal artifact reader and display interpolation; no Qt dependency."""

import h5py
import numpy as np


def read_mode(path, port, anchor, mode):
    with h5py.File(path, "r") as f:
        if f.attrs.get("schema") != "gprmax-eigenmode-fields" or f.attrs["schema_version"] != 1:
            raise ValueError("Unsupported modal field artifact")
        g = f[f"ports/{port}"]
        fields = {}
        target_shape = np.min(
            [g[f"fields/{c}/values"].shape[-2:] for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")], axis=0
        )
        coords = None
        for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            field = g[f"fields/{c}"]
            a = field["values"][anchor, mode]
            uv = [field[d][:] for d in ("u", "v")]
            for dim in range(2):
                if a.shape[dim] == target_shape[dim] + 1:
                    a = (
                        np.take(a, range(a.shape[dim] - 1), axis=dim) + np.take(a, range(1, a.shape[dim]), axis=dim)
                    ) * 0.5
                    uv[dim] = (uv[dim][:-1] + uv[dim][1:]) * 0.5
            fields[c] = a
            coords = uv
        return dict(
            fields=fields,
            u=coords[0],
            v=coords[1],
            normal_axis=int(g.attrs["normal_axis"]),
            transverse_axes=tuple(g.attrs["transverse_axes"]),
            frequency=float(g["frequencies"][anchor]),
            neff=complex(g["neff"][anchor, mode]),
            valid=bool(g["anchor_mode_valid"][anchor, mode]),
            propagating=bool(g["anchor_mode_propagating"][anchor, mode]),
            reference_valid=bool(g["anchor_mode_reference_valid"][anchor, mode]),
        )

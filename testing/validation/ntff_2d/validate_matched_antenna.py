"""Reproduce CUDA/CPU analytical references and finite-antenna refinement.

python -m testing.validation.ntff_2d.validate_matched_antenna --backend cuda --output new_directory
The finite open-ended guide has no closed-form pattern: its checks are energy
balance, two independent contour formulations, and mesh convergence. Absolute
complex far-field validation uses the separate unfitted Hankel references.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gprMax

from .matched_antenna import antenna_scene
from .validate_contour import build_scene, analytical_far_field


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--spacings", nargs="+", type=float, default=(0.001, 0.0005, 0.00025))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        status="running",
        backend=args.backend,
        precision="double",
        python=platform.python_version(),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        cases=[],
        source_sha256={},
    )
    sources = (
        "gprMax/ntff/contour.py",
        "gprMax/virtual_waveguide.py",
        "gprMax/virtual_waveguide_2d.py",
        "gprMax/cuda_opencl/knl_virtual_waveguide_2d.py",
        "gprMax/updates/cuda_updates.py",
        "testing/validation/ntff_2d/matched_antenna.py",
        __file__,
    )
    for source in sources:
        manifest["source_sha256"][str(source)] = hashlib.sha256(Path(source).read_bytes()).hexdigest()

    def save():
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    save()
    options = dict(cpu_precision="double", gpu_precision="double", hide_progress_bars=True, log_level=30)
    if args.backend == "cuda":
        options["gpu"] = [0]
    curves = {}
    try:
        for family in ("TE", "TM"):
            for spacing in args.spacings:
                scene, contours = antenna_scene(family, spacing=spacing)
                name = f"antenna_{family}_{spacing:g}"
                gprMax.run(scenes=[scene], outputfile=args.output / name, **options)
                for contour in contours:
                    r = contour.result
                    grid = contour._monitor.grid
                    port = grid.eigenmodeports[0]
                    expected = (2 / (np.sqrt(np.pi) * 22e9) * np.exp(-1)) ** 2
                    row = dict(
                        case=name,
                        method=contour._monitor.formulation,
                        spacing=spacing,
                        incident_power_relative_error=float(
                            abs(r["incident_power_per_port_per_length"][0, 0] / expected - 1)
                        ),
                        efficiency=float(r["radiation_efficiency"][0]),
                        reflection_magnitude=float(abs(port.result.outgoing[0, 0] / port.result.incident[0, 0])),
                        power_valid=bool(np.all(r["port_power_valid"])),
                        gain_valid=bool(np.all(r["gain_valid"])),
                        finite=bool(np.all(np.isfinite(r["scalar"]))),
                    )
                    manifest["cases"].append(row)
                    save()
                    assert row["power_valid"] and row["gain_valid"] and row["finite"]
                    assert row["incident_power_relative_error"] < 0.02
                    assert abs(row["efficiency"] - 1) < 0.08
                    curves[family, spacing, row["method"]] = (contour._monitor.angles, r["scalar"][0].copy())
            for case in ("single", "pair"):
                scene, contours = build_scene(family, 2, 0.002, case)
                name = f"analytical_{family}_{case}"
                gprMax.run(scenes=[scene], outputfile=args.output / name, **options)
                for contour in contours:
                    expected = analytical_far_field(contour._monitor)
                    error = float(np.max(abs(contour.result["scalar"] - expected)) / np.max(abs(expected)))
                    manifest["cases"].append(
                        dict(case=name, method=contour._monitor.formulation, complex_far_field_relative_error=error)
                    )
                    save()
                    assert error < 0.001
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, family in zip(axes, ("TE", "TM")):
            for spacing in args.spacings:
                for method, style in (("ksir", "-"), ("equivalent_current", "--")):
                    angles, far = curves[family, spacing, method]
                    ax.plot(
                        angles,
                        20 * np.log10(np.maximum(abs(far) / np.max(abs(far)), 1e-4)),
                        style,
                        label=f"{spacing*1000:g} mm {method}",
                    )
            ax.set(
                title=f"{family} matched-port antenna",
                xlabel="Angle (degrees)",
                ylabel="Normalized pattern (dB)",
                xlim=(0, 360),
                ylim=(-45, 0),
            )
            ax.grid(alpha=0.3)
        axes[1].legend(fontsize=7)
        fig.savefig(args.output / "antenna_patterns.png", dpi=180)
        plt.close(fig)
        manifest["status"] = "passed"
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()

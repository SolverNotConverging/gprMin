"""Validate both 2-D NTFF formulations against unfitted line-source solutions.

python -m testing.validation.ntff_2d.validate_contour --output contour_validation
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import platform
import subprocess

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gprMax
from gprMax.ntff.contour import contour_directions
from gprMax.ntff.surfaces import COMPONENT_OFFSETS


def build_scene(family, axis, dl, case):
    domain = [0.24] * 3
    domain[axis] = np.inf
    pml = [10] * 6
    pml[axis] = pml[axis + 3] = 0
    scene = gprMax.Scene()
    for item in (
        gprMax.DomainMode(mode=family),
        gprMax.Domain(p1=domain),
        gprMax.Discretisation(p1=(dl,) * 3),
        gprMax.TimeWindow(time=5e-9),
        gprMax.PMLThickness(thickness=pml),
        gprMax.OMPThreads(n=2),
    ):
        scene.add(item)
    source_class = gprMax.HertzianDipole if family == "TM" else gprMax.MagneticDipole
    positions = ((0.12, 1.0),) if case == "single" else ((0.08, 1.0), (0.16, -1.0))
    for index, (coordinate, sign) in enumerate(positions):
        position = [0.12] * 3
        position[(axis + 1) % 3] = coordinate
        position[axis] = np.inf
        amplitude = sign if family == "TM" else sign * dl
        scene.add(gprMax.Waveform(wave_type="ricker", amp=amplitude, freq=1e9, id=f"pulse{index}"))
        scene.add(source_class(p1=position, polarisation="xyz"[axis], waveform_id=f"pulse{index}"))
    outputs = []
    for formulation in ("ksir", "equivalent_current"):
        lower, upper = [0.056] * 3, [0.184] * 3
        lower[axis] = upper[axis] = np.inf
        output = gprMax.NTFFContour(
            lower, upper, formulation, (0.8e9, 1e9, 1.2e9), np.arange(360), formulation=formulation
        )
        scene.add(output)
        outputs.append(output)
    return scene, outputs


def analytical_far_field(monitor):
    """Continuum outgoing Green solution from impressed source spectra only."""
    tm = monitor.geometry.polarisation == "TM"
    sources = monitor.grid.hertziandipoles if tm else monitor.grid.magneticdipoles
    result = np.zeros((len(monitor.frequencies), len(monitor.angles)), dtype=complex)
    for source in sources:
        samples = (source.waveformvalues_halfdt if tm else source.waveformvalues_wholedt)[: monitor.iterations]
        times = (np.arange(len(samples)) + (0.5 if tm else 0)) * monitor.dt
        spectrum = monitor.dt * (np.exp(-2j * np.pi * monitor.frequencies[:, None] * times) @ samples)
        spectrum *= source.dl / monitor.grid.dl[monitor.axis] if tm else 1 / monitor.grid.dl[monitor.axis]
        position = (np.asarray(source.coord) + COMPONENT_OFFSETS[monitor.scalar_component]) * monitor.grid.dl
        phase = np.exp(
            1j
            * monitor.wavenumbers[:, None]
            * ((position - monitor.origin) @ contour_directions(monitor.angles, monitor.axis).T)
        )
        amplitude = -2j * np.pi * monitor.frequencies * (monitor.mu if tm else monitor.epsilon) * spectrum
        green = np.exp(-0.25j * np.pi) / np.sqrt(8 * np.pi * monitor.wavenumbers)
        result += (amplitude * green)[:, None] * phase
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "opencl", "metal"), default="cpu")
    parser.add_argument("--precision", choices=("single", "double"), default="double")
    parser.add_argument("--axes", default="xyz")
    parser.add_argument("--spacings", type=float, nargs="+", default=(0.004, 0.002))
    args = parser.parse_args()
    if not args.axes or any(axis not in "xyz" for axis in args.axes):
        parser.error("--axes must contain x, y or z")
    args.output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    manifest = dict(
        revision=revision,
        python=platform.python_version(),
        backend=args.backend,
        precision=args.precision,
        spacings=args.spacings,
        axes=args.axes,
        time_window=5e-9,
        frequencies=[0.8e9, 1e9, 1.2e9],
        status="running",
        runs=[],
    )
    manifest["source_sha256"] = {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in (
            "gprMax/ntff/contour.py",
            "gprMax/ntff/device.py",
            "gprMax/ntff/equivalent_currents.py",
            "gprMax/user_objects/cmds_output.py",
            "testing/validation/ntff_2d/validate_contour.py",
        )
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    options = dict(
        cpu_precision=args.precision, gpu_precision=args.precision, hide_progress_bars=True, log_level=logging.WARNING
    )
    if args.backend == "cuda":
        options["gpu"] = [0]
    elif args.backend == "opencl":
        options["opencl"] = [0]
    elif args.backend == "metal":
        options["metal"] = True
    plot_data = {}
    try:
        for axis_name in args.axes:
            axis = "xyz".index(axis_name)
            for family in ("TM", "TE"):
                for case in ("single", "pair"):
                    for dl in args.spacings:
                        name = f"{family}{axis_name}_{case}_{dl:g}".replace(".", "p")
                        scene, outputs = build_scene(family, axis, dl, case)
                        path = args.output / name
                        gprMax.run(scenes=[scene], outputfile=path, **options)
                        record = dict(name=name, file=path.with_suffix(".h5").name, errors={})
                        for output in outputs:
                            expected = analytical_far_field(output._monitor)
                            actual = output.result["scalar"]
                            peak = np.max(abs(expected), axis=1)
                            error = np.max(abs(actual - expected), axis=1) / peak
                            record["errors"][output.formulation] = error.tolist()
                            if not np.all(np.isfinite(actual)) or np.max(error) > 0.02:
                                raise AssertionError(f"{name} {output.formulation}: complex relative error {error}")
                            with h5py.File(path.with_suffix(".h5"), "a") as data:
                                data[f"ntff_contours/{output.ID}"].create_dataset("analytical_scalar", data=expected)
                            if axis_name == args.axes[-1] and dl == min(args.spacings):
                                plot_data[family, case, output.formulation] = (
                                    output._monitor.angles,
                                    actual[1],
                                    expected[1],
                                )
                        manifest["runs"].append(record)
                        manifest_path.write_text(json.dumps(manifest, indent=2))
                        print(json.dumps(record), flush=True)
        fig, axes = plt.subplots(2, 2, figsize=(9, 6), subplot_kw={"projection": "polar"})
        for row, family in enumerate(("TM", "TE")):
            for column, case in enumerate(("single", "pair")):
                ax = axes[row, column]
                for method, style in (("ksir", "--"), ("equivalent_current", ":")):
                    angles, actual, expected = plot_data[family, case, method]
                    scale = np.max(abs(expected))
                    ax.plot(np.deg2rad(angles), abs(actual) / scale, style, label=method)
                ax.plot(np.deg2rad(angles), abs(expected) / scale, color="black", linewidth=0.8, label="Analytical")
                ax.set_title(f"{family}{args.axes[-1]} — {case}")
                ax.set_ylim(0, 1.1)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
        fig.suptitle("2-D NTFF at 1 GHz: independent formulations and line-source reference")
        fig.tight_layout(rect=(0, 0.06, 1, 0.96))
        fig.savefig(args.output / "patterns.png", dpi=180)
        plt.close(fig)
        manifest["status"] = "passed"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

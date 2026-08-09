# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
#
# This file is part of gprMax.

"""Benchmark the Cython circular matched-modal FIR against its old NumPy form.

The benchmark uses the public :class:`CausalModalFIR` step so Python dispatch
and validation overhead are included. Timing is intentionally kept out of the
normal pytest suite; the accompanying test only checks this harness with a tiny
case and makes no speed assertion.

Run from the repository root, for example::

    python -m testing.benchmarking.benchmark_matched_modal_fir
    python -m testing.benchmarking.benchmark_matched_modal_fir \
        --case 4,4096,4096 --minimum-speedup 1.2
"""

import argparse
import json
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

import gprMax
from gprMax.matched_eigenmode_ports import CausalModalFIR


DEFAULT_CASES = ((1, 256, 16384), (4, 4096, 4096), (2, 16384, 2048))


class _NumpyShiftFIR:
    """Frozen reference for the pre-circular ``CausalModalFIR.step``."""

    backend = "numpy_shift"

    def __init__(self, kernels):
        self.kernels = np.ascontiguousarray(kernels).copy()
        self.dtype = self.kernels.dtype
        self.mode_count, self.tap_count = self.kernels.shape
        self._history = np.zeros(
            (self.mode_count, max(0, self.tap_count - 1)), dtype=self.dtype
        )

    def step(self, samples):
        raw_samples = np.asarray(samples)
        if raw_samples.ndim == 0 and self.mode_count == 1:
            raw_samples = raw_samples.reshape(1)
        if raw_samples.shape != (self.mode_count,):
            raise ValueError("samples must have one value per mode")
        current = np.ascontiguousarray(raw_samples, dtype=self.dtype)
        if self.tap_count == 1:
            return np.zeros(self.mode_count, dtype=self.dtype)
        translated = np.sum(
            self.kernels[:, 1:] * self._history,
            axis=1,
            dtype=self.dtype,
        )
        if self._history.shape[1] > 1:
            self._history[:, 1:] = self._history[:, :-1]
        self._history[:, 0] = current
        return np.ascontiguousarray(translated, dtype=self.dtype)


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _parse_case(value):
    try:
        parts = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "case must be modes,taps,steps"
        ) from exc
    if len(parts) != 3 or any(part < 1 for part in parts):
        raise argparse.ArgumentTypeError(
            "case must contain three positive integers: modes,taps,steps"
        )
    return parts


def _evaluate(fir_type, kernels, samples):
    fir = fir_type(kernels)
    return np.vstack([fir.step(sample) for sample in samples]), fir.backend


def _time_once(fir_type, kernels, samples):
    fir = fir_type(kernels)
    last = np.zeros(kernels.shape[0], dtype=kernels.dtype)
    start = perf_counter()
    for sample in samples:
        last = fir.step(sample)
    elapsed = perf_counter() - start
    checksum = np.sum(last)
    return elapsed, checksum, fir.backend


def _median_mad(values):
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    return median, mad


def _modal_fir_mac_count(mode_count, tap_count, step_count):
    """Return the number of causal modal taps actually visited during a run."""
    history_length = max(0, tap_count - 1)
    if step_count <= history_length + 1:
        taps_per_mode = step_count * (step_count - 1) // 2
    else:
        taps_per_mode = (
            history_length * step_count
            - history_length * (history_length + 1) // 2
        )
    return mode_count * taps_per_mode


def _serialise_checksum(value):
    result = complex(value)
    return {"real": result.real, "imag": result.imag}


def _case_data(seed, dtype, mode_count, tap_count, step_count):
    rng = np.random.default_rng(
        seed + 101 * mode_count + 17 * tap_count + dtype.itemsize
    )
    kernels = rng.standard_normal((mode_count, tap_count)) / np.sqrt(tap_count)
    kernels = np.ascontiguousarray(kernels, dtype=dtype)
    kernels[:, 0] = 0
    samples = np.ascontiguousarray(
        rng.standard_normal((step_count, mode_count)), dtype=dtype
    )
    return kernels, samples


def run_benchmark(args):
    """Run a paired benchmark matrix and return JSON-serialisable results."""

    cases = [tuple(case) for case in args.cases]
    for case in cases:
        if len(case) != 3 or any(value < 1 for value in case):
            raise ValueError("each case must be (modes, taps, steps) > 0")
    dtypes = [np.dtype(name) for name in args.dtypes]
    if any(
        dtype not in (np.dtype(np.float32), np.dtype(np.float64))
        for dtype in dtypes
    ):
        raise ValueError("benchmark dtypes must be float32 or float64")

    results = []
    for dtype in dtypes:
        tolerance = 3e-5 if dtype == np.dtype(np.float32) else 5e-13
        for mode_count, tap_count, step_count in cases:
            kernels, samples = _case_data(
                args.seed, dtype, mode_count, tap_count, step_count
            )
            expected, _ = _evaluate(_NumpyShiftFIR, kernels, samples)
            actual, backend = _evaluate(CausalModalFIR, kernels, samples)
            if backend != "cython":
                raise RuntimeError(
                    "matched-modal FIR Cython extension is not available; "
                    "rebuild extensions"
                )
            np.testing.assert_allclose(
                actual, expected, rtol=tolerance, atol=tolerance
            )
            max_abs_error = float(np.max(np.abs(actual - expected), initial=0))

            for _ in range(args.warmups):
                _time_once(_NumpyShiftFIR, kernels, samples)
                _time_once(CausalModalFIR, kernels, samples)

            timings = {"numpy_shift": [], "cython": []}
            checksums = {}
            for repeat in range(args.repeats):
                order = (
                    ((_NumpyShiftFIR, "numpy_shift"), (CausalModalFIR, "cython"))
                    if repeat % 2 == 0
                    else ((CausalModalFIR, "cython"), (_NumpyShiftFIR, "numpy_shift"))
                )
                for fir_type, name in order:
                    elapsed, checksum, measured_backend = _time_once(
                        fir_type, kernels, samples
                    )
                    if name == "cython" and measured_backend != "cython":
                        raise RuntimeError("benchmark left the compiled FIR path")
                    timings[name].append(elapsed)
                    checksums[name] = checksum

            numpy_median, numpy_mad = _median_mad(timings["numpy_shift"])
            cython_median, cython_mad = _median_mad(timings["cython"])
            results.append(
                {
                    "dtype": dtype.name,
                    "mode_count": mode_count,
                    "tap_count": tap_count,
                    "step_count": step_count,
                    "backend": backend,
                    "max_abs_error": max_abs_error,
                    "numpy_shift": {
                        "median_seconds": numpy_median,
                        "mad_seconds": numpy_mad,
                        "run_seconds": timings["numpy_shift"],
                        "checksum": _serialise_checksum(checksums["numpy_shift"]),
                    },
                    "cython": {
                        "median_seconds": cython_median,
                        "mad_seconds": cython_mad,
                        "run_seconds": timings["cython"],
                        "checksum": _serialise_checksum(checksums["cython"]),
                    },
                    "speedup": numpy_median / cython_median,
                    "cython_modal_taps_per_second": _modal_fir_mac_count(
                        mode_count, tap_count, step_count
                    )
                    / cython_median,
                }
            )

    minimum_speedup = getattr(args, "minimum_speedup", None)
    failures = []
    if minimum_speedup is not None:
        failures = [
            {
                "dtype": case["dtype"],
                "mode_count": case["mode_count"],
                "tap_count": case["tap_count"],
                "speedup": case["speedup"],
            }
            for case in results
            if case["tap_count"] >= 4096 and case["speedup"] < minimum_speedup
        ]

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "software": {
            "gprmax": gprMax.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "configuration": {
            "dtypes": [dtype.name for dtype in dtypes],
            "cases": [list(case) for case in cases],
            "repeats": args.repeats,
            "warmups": args.warmups,
            "seed": args.seed,
            "minimum_speedup": minimum_speedup,
        },
        "cases": results,
        "speedup_failures": failures,
    }


def _parser():
    parser = argparse.ArgumentParser(
        description="Benchmark the matched-modal Cython circular FIR."
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        type=_parse_case,
        help="modes,taps,steps; repeat for multiple cases",
    )
    parser.add_argument(
        "--dtype",
        dest="dtypes",
        action="append",
        choices=("float32", "float64"),
        help="precision to benchmark; repeat for both",
    )
    parser.add_argument("--repeats", type=_positive_int, default=7)
    parser.add_argument("--warmups", type=int, choices=range(0, 10), default=2)
    parser.add_argument("--seed", type=int, default=4821)
    parser.add_argument("--minimum-speedup", type=float)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("matched_modal_fir_benchmark_results.json"),
    )
    return parser


def main():
    args = _parser().parse_args()
    args.cases = args.cases or list(DEFAULT_CASES)
    args.dtypes = args.dtypes or ["float32", "float64"]
    if args.minimum_speedup is not None and args.minimum_speedup <= 0:
        raise SystemExit("--minimum-speedup must be positive")
    result = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Results written to {args.output}")
    if result["speedup_failures"]:
        raise SystemExit(
            f"{len(result['speedup_failures'])} case(s) missed the requested speedup"
        )


if __name__ == "__main__":
    main()

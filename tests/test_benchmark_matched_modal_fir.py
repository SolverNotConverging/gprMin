"""Configuration and execution checks for the matched-modal FIR benchmark."""

import argparse
import json
from types import SimpleNamespace

import pytest

from testing.benchmarking.benchmark_matched_modal_fir import (
    _modal_fir_mac_count,
    _parse_case,
    run_benchmark,
)


def test_matched_modal_fir_benchmark_case_parser():
    assert _parse_case("4,4096,2048") == (4, 4096, 2048)
    for value in ("4,4096", "4,0,100", "one,2,3"):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_case(value)


@pytest.mark.parametrize(
    ("mode_count", "tap_count", "step_count", "expected"),
    [
        (2, 1, 50, 0),
        (3, 5, 3, 9),
        (3, 5, 5, 30),
        (3, 5, 8, 66),
    ],
)
def test_matched_modal_fir_benchmark_counts_only_available_history_taps(
    mode_count, tap_count, step_count, expected
):
    assert _modal_fir_mac_count(mode_count, tap_count, step_count) == expected


def test_matched_modal_fir_benchmark_executes_compiled_public_path():
    args = SimpleNamespace(
        cases=[(2, 32, 64)],
        dtypes=["float64"],
        repeats=1,
        warmups=0,
        seed=918,
        minimum_speedup=None,
    )

    result = run_benchmark(args)

    assert result["configuration"] == {
        "dtypes": ["float64"],
        "cases": [[2, 32, 64]],
        "repeats": 1,
        "warmups": 0,
        "seed": 918,
        "minimum_speedup": None,
    }
    assert result["speedup_failures"] == []
    assert len(result["cases"]) == 1
    case = result["cases"][0]
    assert case["backend"] == "cython"
    assert case["dtype"] == "float64"
    assert (case["mode_count"], case["tap_count"], case["step_count"]) == (
        2,
        32,
        64,
    )
    assert case["max_abs_error"] < 1e-12
    assert case["numpy_shift"]["median_seconds"] > 0
    assert case["cython"]["median_seconds"] > 0
    assert case["speedup"] > 0
    assert case["cython_modal_taps_per_second"] > 0
    expected_macs = _modal_fir_mac_count(2, 32, 64)
    assert (
        case["cython_modal_taps_per_second"]
        * case["cython"]["median_seconds"]
        == pytest.approx(expected_macs)
    )
    json.dumps(result)

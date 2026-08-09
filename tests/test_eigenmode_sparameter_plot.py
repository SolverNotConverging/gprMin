import csv
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

from testing.regression.eigenmode_sources.cases.bending_waveguide.plot_bend_comparison import (
    plot_comparison,
)
from testing.regression.eigenmode_sources.cases.plot_sparameters import plot_tree
from testing.regression.eigenmode_sources.cases.validate_sparameters import (
    validate_tree,
)
from testing.regression.eigenmode_sources.plot_snapshots import snapshot_paths


def _load_example_plotter():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "features"
        / "eigenmode_ports"
        / "example_1_straight_waveguide"
        / "plot_results.py"
    )
    spec = importlib.util.spec_from_file_location("eigenmode_example_plotter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_matched_example_plotter():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "features"
        / "eigenmode_ports"
        / "example_4_matched_waveguide"
        / "plot_results.py"
    )
    spec = importlib.util.spec_from_file_location(
        "matched_eigenmode_example_plotter", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_snapshot(path: Path, time_ns: float):
    with h5py.File(path, "w") as output:
        output.attrs["time"] = time_ns * 1e-9
        output.attrs["dx_dy_dz"] = (0.001, 0.001, 0.001)
        output["Ez"] = np.full((3, 2, 1), time_ns)


def _write_matched_example_output(stem: Path):
    csv_path = stem.with_name(stem.name + "_sparameters.csv")
    fieldnames = (
        "frequency_hz",
        "source_port",
        "source_mode",
        "destination_port",
        "destination_mode",
        "S_magnitude_db",
        "valid",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for frequency, destination_port, magnitude_db, valid in (
            (11.5e9, 1, -18.0, 1),
            (10.0e9, 1, -22.0, 1),
            (11.5e9, 2, -1.5, 1),
            (10.0e9, 2, -0.8, 1),
            (10.75e9, 2, 100.0, 0),
        ):
            writer.writerow(
                {
                    "frequency_hz": frequency,
                    "source_port": 1,
                    "source_mode": 1,
                    "destination_port": destination_port,
                    "destination_mode": 1,
                    "S_magnitude_db": magnitude_db,
                    "valid": valid,
                }
            )

    with h5py.File(stem.with_suffix(".h5"), "w") as output:
        ports = output.create_group("eigenmode_ports")
        for port_number, boundary_index in ((1, 0), (2, 120)):
            port = ports.create_group(f"port{port_number}")
            port.attrs["Matched"] = True
            port.attrs["MatchDepthCells"] = 10
            port.attrs["MatchedBoundaryIndex"] = boundary_index
            port.attrs["ModeIndices"] = (1,)
            port.attrs["MatchedFormulation"] = "Alimenti2000NumericalModalTranslation"

    snapshot_dir = stem.with_name(stem.name + "_snaps")
    snapshot_dir.mkdir()
    _write_snapshot(snapshot_dir / "late_name.h5", 0.8)
    _write_snapshot(snapshot_dir / "early_name.h5", 1.6)
    _write_snapshot(snapshot_dir / "middle_name.h5", 1.2)


def _write_case(root, source_mode, primary_transmission_db=-1, case_name=None):
    case_name = case_name or f"mode{source_mode}"
    case_dir = root / case_name
    case_dir.mkdir()
    path = case_dir / f"{case_name}_sparameters.csv"
    fieldnames = (
        "frequency_hz",
        "source_port",
        "source_mode",
        "destination_port",
        "destination_mode",
        "S_real",
        "S_imag",
        "S_magnitude",
        "S_magnitude_db",
        "S_phase_deg",
        "coefficient_magnitude_squared",
        "valid",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for frequency in (1e9, 2e9):
            for destination_port, destination_mode, magnitude_db in (
                (1, source_mode, -30),
                (
                    2,
                    1,
                    primary_transmission_db if source_mode == 1 else -40,
                ),
                (
                    2,
                    2,
                    -40 if source_mode == 1 else primary_transmission_db,
                ),
            ):
                writer.writerow(
                    {
                        "frequency_hz": frequency,
                        "source_port": 1,
                        "source_mode": source_mode,
                        "destination_port": destination_port,
                        "destination_mode": destination_mode,
                        "S_real": 1,
                        "S_imag": 0,
                        "S_magnitude": 10 ** (magnitude_db / 20),
                        "S_magnitude_db": magnitude_db,
                        "S_phase_deg": 0,
                        "coefficient_magnitude_squared": 10 ** (magnitude_db / 10),
                        "valid": 1,
                    }
                )


def test_modal_sparameter_plot_writes_one_combined_plot_per_csv(tmp_path):
    _write_case(tmp_path, 1)
    _write_case(tmp_path, 2)

    outputs = plot_tree(tmp_path)

    assert {path.name for path in outputs} == {
        "mode1_sparameters_plot.png",
        "mode2_sparameters_plot.png",
    }
    assert all(path.stat().st_size > 0 for path in outputs)


def test_example_snapshots_are_sorted_by_physical_time_and_can_be_capped(
    tmp_path,
):
    plotter = _load_example_plotter()
    stem = tmp_path / "guide"
    snapshot_dir = tmp_path / "guide_snaps"
    snapshot_dir.mkdir()
    _write_snapshot(snapshot_dir / "guide_1000ps.h5", 1.0)
    _write_snapshot(snapshot_dir / "guide_1600ps.h5", 1.6)
    _write_snapshot(snapshot_dir / "guide_400ps.h5", 0.4)

    snapshots = plotter.read_field_snapshots(stem, maximum_time_ns=1.0)

    assert [snapshot[0] for snapshot in snapshots] == pytest.approx([0.4, 1.0])


def test_matched_waveguide_example_plotter_consumes_synthetic_outputs(tmp_path, monkeypatch):
    plotter = _load_matched_example_plotter()
    stem = tmp_path / "matched_waveguide"
    sparameter_plot = tmp_path / "matched_waveguide_sparameters.png"
    field_plot = tmp_path / "matched_waveguide_field_propagation.png"
    _write_matched_example_output(stem)
    monkeypatch.setattr(plotter, "OUTPUT_STEM", stem)
    monkeypatch.setattr(plotter, "SPARAMETER_PLOT_PATH", sparameter_plot)
    monkeypatch.setattr(plotter, "FIELD_PLOT_PATH", field_plot)

    plotter.main()

    traces = plotter.read_sparameters(stem)
    assert set(traces) == {1, 2}
    assert traces[1][:, 0].tolist() == [10.0, 11.5]
    assert traces[2][:, 1].tolist() == [-0.8, -1.5]
    snapshots = plotter.read_field_snapshots(stem)
    assert [snapshot[0] for snapshot in snapshots] == pytest.approx(
        [0.8, 1.2, 1.6]
    )
    assert sparameter_plot.stat().st_size > 0
    assert field_plot.stat().st_size > 0


def test_regression_snapshot_plot_ignores_stale_generated_files(tmp_path):
    case_dir = tmp_path / "guide"
    snapshot_dir = case_dir / "guide_snaps"
    snapshot_dir.mkdir(parents=True)
    (case_dir / "guide.in").write_text(
        "#snapshot: 0 0 0 1 1 inf 1 1 1 1e-9 xy_center_current.h5\n",
        encoding="utf-8",
    )
    _write_snapshot(snapshot_dir / "xy_center_current.h5", 1.0)
    _write_snapshot(snapshot_dir / "xy_center_stale.h5", 2.0)

    paths = snapshot_paths(case_dir, "xy")

    assert [path.name for path in paths] == ["xy_center_current.h5"]


def test_straight_waveguide_sparameter_validator_accepts_expected_response(
    tmp_path,
):
    straight_root = tmp_path / "straight_waveguide"
    straight_root.mkdir()
    _write_case(straight_root, 1, primary_transmission_db=0.1)

    messages = validate_tree(tmp_path)

    assert len(messages) == 1
    assert "mean S21=0.100 dB" in messages[0]


def test_curved_bend_comparison_and_validator_expect_large_radius_improvement(
    tmp_path,
):
    bend_root = tmp_path / "bending_waveguide"
    for polarisation in ("2d_tm", "2d_te"):
        polarisation_root = bend_root / polarisation
        polarisation_root.mkdir(parents=True)
        for case_name, s21_db in (
            ("small_bend", -5),
            ("medium_bend", -2),
            ("large_bend", -0.5),
        ):
            _write_case(
                polarisation_root,
                source_mode=1,
                primary_transmission_db=s21_db,
                case_name=case_name,
            )

    messages = validate_tree(tmp_path)
    output = plot_comparison(bend_root)

    assert len(messages) == 2
    assert all("curved bends" in message for message in messages)
    assert output.name == "bend_radius_sparameters_comparison.png"
    assert output.stat().st_size > 0


def test_curved_bend_validator_rejects_small_radius_improvement(tmp_path):
    polarisation_root = tmp_path / "bending_waveguide" / "2d_tm"
    polarisation_root.mkdir(parents=True)
    for case_name, s21_db in (
        ("small_bend", -1),
        ("medium_bend", -0.75),
        ("large_bend", -0.5),
    ):
        _write_case(
            polarisation_root,
            source_mode=1,
            primary_transmission_db=s21_db,
            case_name=case_name,
        )

    with pytest.raises(AssertionError, match="at least 2 dB"):
        validate_tree(tmp_path)

from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax
import gprMax.matched_eigenmode_ports as matched_module


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _rectangular_waveguide_scene(
    *,
    lower_depth=5,
    upper_depth=5,
    lower_port_position=0.005,
    pml_thickness=0,
    fill_conductivity=0,
    heterogeneous_buffer=False,
    thin_wire_buffer=False,
    symmetry_face=None,
    overlapping_ports=False,
    include_upper_port=True,
):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.03, 0.02, 0.012)))
    scene.add(gprMax.PMLThickness(thickness=pml_thickness))
    if symmetry_face is not None:
        scene.add(gprMax.SymmetryBoundary(face=symmetry_face, type="pmc"))
    scene.add(gprMax.TimeWindow(time=0.12e-9))
    scene.add(
        gprMax.Waveform(
            wave_type="contsine",
            amp=1,
            freq=20e9,
            id="mode",
        )
    )

    if fill_conductivity:
        scene.add(
            gprMax.Material(
                er=2,
                se=fill_conductivity,
                mr=1,
                sm=0,
                id="lossy_fill",
            )
        )
        scene.add(
            gprMax.Box(
                p1=(0, 0, 0),
                p2=(0.03, 0.02, 0.012),
                material_id="lossy_fill",
            )
        )
    if heterogeneous_buffer:
        scene.add(gprMax.Material(er=2, se=0, mr=1, sm=0, id="buffer_insert"))
        scene.add(
            gprMax.Box(
                p1=(0, 0.006, 0.004),
                p2=(0.003, 0.014, 0.008),
                material_id="buffer_insert",
            )
        )

    scene.add(
        gprMax.Box(
            p1=(0, 0, 0),
            p2=(0.03, 0.001, 0.012),
            material_id="pec",
        )
    )
    if thin_wire_buffer:
        scene.add(
            gprMax.ThinWire(
                p1=(0, 0.010, 0.006),
                p2=(lower_port_position + 0.001, 0.010, 0.006),
                radius=0.1e-3,
            )
        )
    scene.add(
        gprMax.Box(
            p1=(0, 0.019, 0),
            p2=(0.03, 0.02, 0.012),
            material_id="pec",
        )
    )
    scene.add(
        gprMax.Box(
            p1=(0, 0.001, 0),
            p2=(0.03, 0.019, 0.001),
            material_id="pec",
        )
    )
    scene.add(
        gprMax.Box(
            p1=(0, 0.001, 0.011),
            p2=(0.03, 0.019, 0.012),
            material_id="pec",
        )
    )

    scene.add(
        gprMax.EigenmodeBand(
            id="band",
            fmin=20e9,
            fmax=20e9,
            points=1,
        )
    )
    scene.add(
        gprMax.EigenmodePort(
            port=1,
            p1=(lower_port_position, 0.001, 0.001),
            p2=(lower_port_position, 0.019, 0.011),
            direction="+",
            modes=(1,),
            anchors=(20e9,),
            plot_fields=False,
        )
    )
    if include_upper_port:
        scene.add(
            gprMax.EigenmodePort(
                port=2,
                p1=(
                    lower_port_position if overlapping_ports else 0.025,
                    0.001,
                    0.001,
                ),
                p2=(
                    lower_port_position if overlapping_ports else 0.025,
                    0.019,
                    0.011,
                ),
                direction="+" if overlapping_ports else "-",
                modes=(1,),
                anchors=(20e9,),
                plot_fields=False,
            )
        )
    scene.add(gprMax.EigenmodeMatch(port=1, depth_cells=lower_depth))
    if include_upper_port:
        scene.add(gprMax.EigenmodeMatch(port=2, depth_cells=upper_depth))
    scene.add(
        gprMax.EigenmodeExcitation(
            port=1,
            mode=1,
            waveform="mode",
            plot_waveform=False,
        )
    )
    scene.add(gprMax.Rx(p1=(0.015, 0.01, 0.006)))
    return scene


@pytest.mark.parametrize("cpu_precision", ["single", "double"])
def test_3d_one_mode_active_and_passive_matched_ports_run(
    tmp_path,
    cpu_precision,
):
    output = tmp_path / f"matched_rectangular_waveguide_{cpu_precision}"
    gprMax.run(
        scenes=[_rectangular_waveguide_scene()],
        n=1,
        outputfile=output,
        cpu_precision=cpu_precision,
        hide_progress_bars=True,
    )

    with h5py.File(output.with_suffix(".h5"), "r") as handle:
        receiver = handle["rxs/rx1"]
        field_peak = max(
            np.max(np.abs(receiver[component][...]))
            for component in ("Ex", "Ey", "Ez")
        )
        assert field_peak > 0

        lower = handle["eigenmode_ports/port1"]
        upper = handle["eigenmode_ports/port2"]
        assert bool(lower.attrs["Matched"])
        assert bool(upper.attrs["Matched"])
        assert lower.attrs["MatchedBoundaryIndex"] == 0
        assert upper.attrs["MatchedBoundaryIndex"] == 30
        assert lower.attrs["MatchDepthCells"] == 5
        assert upper.attrs["MatchDepthCells"] == 5
        assert lower.attrs["MatchedFormulation"] == (
            "PowerAdjointModalAdmittanceADE"
        )
        assert upper.attrs["MatchedFormulation"] == (
            "PowerAdjointModalAdmittanceADE"
        )
        np.testing.assert_array_equal(lower.attrs["ModeIndices"], (1,))
        np.testing.assert_array_equal(upper.attrs["ModeIndices"], (1,))


def test_order_zero_rational_runtime_runs_on_a_fixed_profile_waveguide(
    tmp_path,
    monkeypatch,
):
    original_synthesis = matched_module.synthesize_passive_admittance

    def fixed_profile_model(*args, **kwargs):
        return original_synthesis(
            np.asarray((1.0,)),
            np.asarray((1.0,)),
            candidate_orders=(0,),
            direct=1.0,
            maximum_relative_error=1e-12,
            maximum_reflection_error=1e-12,
        )

    monkeypatch.setattr(
        matched_module,
        "synthesize_passive_admittance",
        fixed_profile_model,
    )
    monkeypatch.setattr(matched_module, "ENABLE_RATIONAL_ADMITTANCE_RUNTIME", True)
    output = tmp_path / "matched_fixed_profile_rational"

    gprMax.run(
        scenes=[_rectangular_waveguide_scene()],
        n=1,
        outputfile=output,
        hide_progress_bars=True,
    )

    with h5py.File(output.with_suffix(".h5"), "r") as handle:
        port = handle["eigenmode_ports/port1"]
        assert bool(port.attrs["MatchedRationalRuntimeEnabled"])
        assert port.attrs["MatchedRationalShadowStatus"] == "certified-enabled"
        assert len(port.attrs["MatchedRationalShadowPoleReal"]) == 0
        assert bool(port.attrs["MatchedRationalShadowFinalPassive"])


def test_disabled_shadow_fit_failure_does_not_abort_matched_boundary(
    tmp_path,
    monkeypatch,
):
    def fail_synthesis(*args, **kwargs):
        raise RuntimeError("deliberate shadow failure")

    def forbidden_ade(*args, **kwargs):
        raise AssertionError("disabled runtime must not construct a rational ADE")

    monkeypatch.setattr(
        matched_module,
        "synthesize_passive_admittance",
        fail_synthesis,
    )
    monkeypatch.setattr(matched_module, "RationalModalAdmittanceADE", forbidden_ade)
    output = tmp_path / "matched_shadow_failure_isolated"

    gprMax.run(
        scenes=[_rectangular_waveguide_scene()],
        n=1,
        outputfile=output,
        hide_progress_bars=True,
    )

    with h5py.File(output.with_suffix(".h5"), "r") as handle:
        port = handle["eigenmode_ports/port1"]
        assert not bool(port.attrs["MatchedRationalRuntimeEnabled"])
        assert port.attrs["MatchedRationalShadowStatus"] == "failed-disabled"
        assert "deliberate shadow failure" in port.attrs[
            "MatchedRationalShadowFailure"
        ]


def test_matched_eigenmode_hash_command_runs_end_to_end(tmp_path):
    inputfile = tmp_path / "matched_port.in"
    inputfile.write_text(
        "\n".join(
            (
                "#title: Matched eigenmode hash command",
                "#dx_dy_dz: 0.001 0.001 0.001",
                "#domain: 0.03 0.02 0.012",
                "#time_window: 0.12e-9",
                "#pml_cells: 0",
                "#waveform: contsine 1 20e9 mode",
                "#box: 0 0 0 0.03 0.001 0.012 pec",
                "#box: 0 0.019 0 0.03 0.02 0.012 pec",
                "#box: 0 0.001 0 0.03 0.019 0.001 pec",
                "#box: 0 0.001 0.011 0.03 0.019 0.012 pec",
                "#eigenmode_band: band 20e9 20e9 1",
                "#eigenmode_port: 1 0.005 0.001 0.001 0.005 0.019 0.011 + 1 20e9",
                "#eigenmode_match: 1 5",
                "#eigenmode_excitation: 1 1 mode",
                "#rx: 0.015 0.010 0.006",
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "matched_port_hash"

    gprMax.run(
        inputfile=inputfile,
        n=1,
        outputfile=output,
        hide_progress_bars=True,
    )

    with h5py.File(output.with_suffix(".h5"), "r") as handle:
        port = handle["eigenmode_ports/port1"]
        assert bool(port.attrs["Matched"])
        assert port.attrs["MatchDepthCells"] == 5
        assert port.attrs["MatchedFormulation"] == (
            "PowerAdjointModalAdmittanceADE"
        )
        assert max(
            np.max(np.abs(handle[f"rxs/rx1/{component}"][...]))
            for component in ("Ex", "Ey", "Ez")
        ) > 0


def test_matched_waveguide_example_builds_geometry_from_exact_hash_input(tmp_path):
    source = (
        REPOSITORY_ROOT
        / "examples"
        / "features"
        / "eigenmode_ports"
        / "example_4_matched_waveguide"
        / "matched_waveguide.in"
    )
    inputfile = tmp_path / source.name
    contents = source.read_text(encoding="utf-8")
    inputfile.write_text(contents, encoding="utf-8")
    commands = [line.strip() for line in contents.splitlines()]

    assert "#pml_cells: 0" in commands
    assert "#domain: 0.300 0.012 0.006" in commands
    assert "#dx_dy_dz: 0.0005 0.0005 0.0005" in commands
    assert "#time_window: 18e-9" in commands
    assert "#eigenmode_band: microstrip_band 4.3e9 4.7e9 21" in commands
    assert "#rx: 0.150 0.006 0.003" in commands
    for expected_box in (
        "#box: 0 0 0 0.300 0.012 0.0005 pec",
        "#box: 0 0 0.0005 0.300 0.012 0.0020 substrate",
        "#box: 0 0.005 0.0020 0.300 0.007 0.0025 pec",
    ):
        assert expected_box in commands
    assert not any(command.startswith("#domain_mode:") for command in commands)
    assert [
        command for command in commands if command.startswith("#eigenmode_port:")
    ] == [
        "#eigenmode_port: 1 0.003 0 0.0005 0.003 0.012 0.006 + 1 auto",
        "#eigenmode_port: 2 0.297 0 0.0005 0.297 0.012 0.006 - 1 auto",
    ]
    assert [
        command for command in commands if command.startswith("#eigenmode_match:")
    ] == [
        "#eigenmode_match: 1 6",
        "#eigenmode_match: 2 6",
    ]
    snapshot_commands = [
        command for command in commands if command.startswith("#snapshot:")
    ]
    assert len(snapshot_commands) == 8
    assert all(
        command.startswith("#snapshot: 0 0.006 0 0.300 ")
        for command in snapshot_commands
    )
    snapshot_times_ns = [
        float(command.split()[-2]) * 1e9 for command in snapshot_commands
    ]
    assert snapshot_times_ns == pytest.approx(
        [2.7, 3.3, 3.9, 4.5, 5.4, 6.6, 8.5, 15.0]
    )
    assert [command.split()[-1] for command in snapshot_commands] == [
        "matched_microstrip_2700ps.h5",
        "matched_microstrip_3300ps.h5",
        "matched_microstrip_3900ps.h5",
        "matched_microstrip_4500ps.h5",
        "matched_microstrip_5400ps.h5",
        "matched_microstrip_6600ps.h5",
        "matched_microstrip_8500ps.h5",
        "matched_microstrip_15000ps.h5",
    ]

    gprMax.run(
        inputfile=inputfile,
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "matched_waveguide_geometry",
        hide_progress_bars=True,
    )

    assert (tmp_path / "matched_waveguide_Port1_Mode1.png").stat().st_size > 0
    assert (tmp_path / "matched_waveguide_Port2_Mode1.png").stat().st_size > 0
    assert (
        tmp_path / "matched_waveguide_EigenmodeExcitation.png"
    ).stat().st_size > 0


def test_matched_port_requires_exact_distance_to_domain_face(tmp_path):
    with pytest.raises(ValueError, match="declares depth_cells=4.*5 cell"):
        gprMax.run(
            scenes=[_rectangular_waveguide_scene(lower_depth=4)],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "wrong_depth",
            hide_progress_bars=True,
        )


def test_matched_port_rejects_pml_on_its_domain_face(tmp_path):
    with pytest.raises(ValueError, match="set that face's PML thickness to zero"):
        gprMax.run(
            scenes=[
                _rectangular_waveguide_scene(
                    pml_thickness=(1, 0, 0, 0, 0, 0),
                )
            ],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "matched_pml_overlap",
            hide_progress_bars=True,
        )


def test_matched_port_rejects_transverse_pml_overlap(tmp_path):
    with pytest.raises(ValueError, match="intersects a transverse PML slab"):
        gprMax.run(
            scenes=[
                _rectangular_waveguide_scene(
                    pml_thickness=(0, 2, 0, 0, 0, 0),
                )
            ],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "matched_transverse_pml_overlap",
            hide_progress_bars=True,
        )


def test_matched_port_rejects_opposite_longitudinal_pml_overlap(tmp_path):
    with pytest.raises(ValueError, match="opposite longitudinal PML slab"):
        gprMax.run(
            scenes=[
                _rectangular_waveguide_scene(
                    lower_depth=25,
                    lower_port_position=0.025,
                    pml_thickness=(0, 0, 0, 10, 0, 0),
                    include_upper_port=False,
                )
            ],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "matched_opposite_pml_overlap",
            hide_progress_bars=True,
        )


def test_matched_port_requires_interior_reference_plane(tmp_path):
    with pytest.raises(ValueError, match="reference plane to be strictly inside"):
        gprMax.run(
            scenes=[
                _rectangular_waveguide_scene(
                    lower_depth=30,
                    lower_port_position=0.030,
                    include_upper_port=False,
                )
            ],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "matched_outer_reference_plane",
            hide_progress_bars=True,
        )


def test_matched_port_rejects_symmetry_on_its_domain_face(tmp_path):
    with pytest.raises(ValueError, match="cannot share domain face x0"):
        gprMax.run(
            scenes=[_rectangular_waveguide_scene(symmetry_face="x0")],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "matched_symmetry_overlap",
            hide_progress_bars=True,
        )


def test_matched_port_rejects_lossy_fill(tmp_path):
    with pytest.raises(ValueError, match="lossless, nondispersive section materials"):
        gprMax.run(
            scenes=[_rectangular_waveguide_scene(fill_conductivity=0.1)],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "lossy_match",
            hide_progress_bars=True,
        )


def test_matched_port_rejects_heterogeneous_longitudinal_buffer(tmp_path):
    with pytest.raises(ValueError, match="longitudinally uniform material cross-section"):
        gprMax.run(
            scenes=[_rectangular_waveguide_scene(heterogeneous_buffer=True)],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "heterogeneous_match",
            hide_progress_bars=True,
        )


def test_matched_port_rejects_thin_wire_update_stencil(tmp_path):
    with pytest.raises(ValueError, match="thin-wire custom update stencils"):
        gprMax.run(
            scenes=[_rectangular_waveguide_scene(thin_wire_buffer=True)],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "thin_wire_match",
            hide_progress_bars=True,
        )


def test_matched_port_rejects_overlapping_apertures(tmp_path):
    with pytest.raises(ValueError, match="both write.*Matched port apertures"):
        gprMax.run(
            scenes=[_rectangular_waveguide_scene(overlapping_ports=True)],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "overlapping_matches",
            hide_progress_bars=True,
        )

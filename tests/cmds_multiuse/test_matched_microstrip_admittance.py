import h5py
import numpy as np
import pytest

import gprMax
import gprMax.matched_eigenmode_ports as matched_module


def _shielded_microstrip_scene(*, time_window):
    """Small uniform lossless microstrip with no absorbing grid cells."""

    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(0.5e-3, 0.5e-3, 0.5e-3)))
    scene.add(gprMax.Domain(p1=(0.024, 0.012, 0.006)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=time_window))
    scene.add(gprMax.Material(er=4.4, se=0, mr=1, sm=0, id="substrate"))
    scene.add(
        gprMax.Box(
            p1=(0, 0, 0),
            p2=(0.024, 0.012, 0.0005),
            material_id="pec",
        )
    )
    scene.add(
        gprMax.Box(
            p1=(0, 0, 0.0005),
            p2=(0.024, 0.012, 0.0020),
            material_id="substrate",
        )
    )
    scene.add(
        gprMax.Box(
            p1=(0, 0.005, 0.0020),
            p2=(0.024, 0.007, 0.0025),
            material_id="pec",
        )
    )
    scene.add(
        gprMax.EigenmodeBand(
            id="microstrip_band",
            fmin=4e9,
            fmax=5e9,
            points=21,
        )
    )
    scene.add(
        gprMax.EigenmodePort(
            port=1,
            p1=(0.003, 0, 0.0005),
            p2=(0.003, 0.012, 0.006),
            direction="+",
            modes=(1,),
            anchors="auto",
            plot_fields=False,
        )
    )
    scene.add(
        gprMax.EigenmodePort(
            port=2,
            p1=(0.021, 0, 0.0005),
            p2=(0.021, 0.012, 0.006),
            direction="-",
            modes=(1,),
            anchors="auto",
            plot_fields=False,
        )
    )
    scene.add(gprMax.EigenmodeMatch(port=1, depth_cells=6))
    scene.add(gprMax.EigenmodeMatch(port=2, depth_cells=6))
    scene.add(
        gprMax.EigenmodeExcitation(
            port=1,
            mode=1,
            waveform="auto",
            plot_waveform=False,
        )
    )
    scene.add(gprMax.Rx(p1=(0.012, 0.006, 0.003)))
    return scene


def _electric_trace(handle):
    receiver = handle["rxs/rx1"]
    components = [
        np.asarray(receiver[name][...], dtype=np.float64)
        for name in ("Ex", "Ey", "Ez")
    ]
    return np.sqrt(sum(component**2 for component in components))


@pytest.mark.integration
def test_constant_admittance_matches_lossless_microstrip_without_pml(tmp_path):
    output = tmp_path / "matched_microstrip_admittance"
    gprMax.run(
        scenes=[_shielded_microstrip_scene(time_window=5e-9)],
        n=1,
        outputfile=output,
        hide_progress_bars=True,
    )

    with h5py.File(output.with_suffix(".h5"), "r") as handle:
        port1 = handle["eigenmode_ports/port1"]
        port2 = handle["eigenmode_ports/port2"]
        for port, boundary_index in ((port1, 0), (port2, 48)):
            assert bool(port.attrs["Matched"])
            assert port.attrs["MatchedFormulation"] == (
                "PowerAdjointModalAdmittanceADE"
            )
            assert port.attrs["MatchedBoundaryIndex"] == boundary_index
            assert port.attrs["MatchDepthCells"] == 6
            np.testing.assert_allclose(
                port.attrs["MatchedNormalizedAdmittance"],
                (1.0,),
            )
            time_constant = np.asarray(
                port.attrs["MatchedModalHalfCellTimeConstant"]
            )
            assert time_constant.shape == (1,)
            assert np.all(np.isfinite(time_constant))
            assert np.all(time_constant > 0)
            anchor_frequencies = np.asarray(port.attrs["AnchorFrequencies"])
            sampled_admittance = np.asarray(
                port.attrs["MatchedFixedBasisAdmittanceReal"]
            ) + 1j * np.asarray(port.attrs["MatchedFixedBasisAdmittanceImag"])
            assert anchor_frequencies.size >= 9
            assert sampled_admittance.shape == anchor_frequencies.shape
            assert np.all(np.isfinite(sampled_admittance))
            centre = np.argmin(np.abs(anchor_frequencies - 4.5e9))
            assert sampled_admittance[centre] == pytest.approx(1.0 + 0.0j)
            assert np.max(port.attrs["MatchedFixedBasisElectricResidual"]) < 0.03
            assert np.max(port.attrs["MatchedFixedBasisMagneticResidual"]) < 0.03
            assert port.attrs["MatchedRationalShadowStatus"] == "certified-disabled"
            assert not bool(port.attrs["MatchedRationalRuntimeEnabled"])
            assert "disabled" in port.attrs["MatchedRationalRuntimeRejection"]
            assert bool(port.attrs["MatchedRationalShadowFinalPassive"])
            assert port.attrs[
                "MatchedRationalShadowMinimumRealAdmittance"
            ] > port.attrs["MatchedRationalShadowRequiredPassivityMargin"]
            assert len(port.attrs["MatchedRationalShadowPoleReal"]) == 2
            discrete_poles = np.asarray(
                port.attrs["MatchedRationalShadowDiscretePoleReal"]
            ) + 1j * np.asarray(port.attrs["MatchedRationalShadowDiscretePoleImag"])
            assert np.all(np.abs(discrete_poles) < 1)
            assert len(port.attrs["MatchedRationalShadowTrainingFrequencies"]) >= 8
            assert len(port.attrs["MatchedRationalShadowValidationFrequencies"]) >= 1

        frequency = np.asarray(port1["frequency"][...])
        inband = (frequency >= 4.1e9) & (frequency <= 4.9e9)
        valid = (
            np.asarray(port1["valid_S"][0], dtype=bool)
            & np.asarray(port2["valid_S"][0], dtype=bool)
            & inband
        )
        assert np.count_nonzero(valid) >= 15
        s11 = np.asarray(port1["S"][0, valid], dtype=np.complex128)
        s21 = np.asarray(port2["S"][0, valid], dtype=np.complex128)
        assert np.max(np.abs(s11)) < 0.18
        assert np.min(np.abs(s21)) > 0.98
        np.testing.assert_allclose(
            np.abs(s11) ** 2 + np.abs(s21) ** 2,
            1.0,
            atol=0.05,
        )

        trace = _electric_trace(handle)
        assert np.all(np.isfinite(trace))
        assert np.max(trace) > 0


def test_experimental_scalar_rational_runtime_rejects_microstrip_profile_residual(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(matched_module, "ENABLE_RATIONAL_ADMITTANCE_RUNTIME", True)
    output = tmp_path / "matched_microstrip_rational_admittance"
    with pytest.raises(
        ValueError,
        match=r"fixed-basis E/H residual.*higher-dimensional discrete DtN",
    ):
        gprMax.run(
            scenes=[_shielded_microstrip_scene(time_window=5e-9)],
            n=1,
            geometry_only=True,
            outputfile=output,
            hide_progress_bars=True,
        )
    assert not output.with_suffix(".h5").exists()


@pytest.mark.integration
@pytest.mark.slow
def test_constant_admittance_microstrip_has_no_late_time_growth(tmp_path):
    output = tmp_path / "matched_microstrip_admittance_long"
    gprMax.run(
        scenes=[_shielded_microstrip_scene(time_window=12e-9)],
        n=1,
        outputfile=output,
        hide_progress_bars=True,
    )

    with h5py.File(output.with_suffix(".h5"), "r") as handle:
        trace = _electric_trace(handle)

    assert np.all(np.isfinite(trace))
    peak = float(np.max(trace, initial=0.0))
    assert peak > 0

    block_length = trace.size // 12
    block_rms = np.asarray(
        [
            np.sqrt(np.mean(block**2))
            for block in np.array_split(trace, 12)
        ]
    )
    assert block_length > 0
    assert np.all(np.isfinite(block_rms))
    assert np.max(block_rms[-2:]) < 0.02 * peak
    assert block_rms[-1] <= max(2 * block_rms[-2], 1e-10 * peak)

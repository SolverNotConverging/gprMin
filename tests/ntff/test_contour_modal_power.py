"""Independent modal power arithmetic and enclosure failure cases."""

from types import SimpleNamespace
import numpy as np
import pytest
from tests.ntff.test_contour import monitor
from gprMax.ntff.surfaces import COMPONENT_OFFSETS


def power_monitor():
    m = monitor("TE", 2)
    nf = len(m.frequencies)
    a = np.tile(np.array([1.0, 1j])[:, None], (1, nf))
    b = 0.2 * a
    w = np.tile(np.array([[1.0, -0.25j], [0.25j, 2.0]])[None], (nf, 1, 1))
    port = SimpleNamespace(
        owner=SimpleNamespace(port_index=1),
        power_matrix=w,
        electric_gram=w.copy(),
        power_matrix_valid=np.ones(nf, bool),
        result=SimpleNamespace(
            frequency=m.frequencies.copy(), incident=a, outgoing=b, power_wave_valid=np.ones((2, nf), bool)
        ),
    )
    m.grid.eigenmodeports = [port]
    m._result = dict(radiation_intensity=np.ones((nf, 4)), radiated_power_per_length=np.ones(nf))
    return m, port


def test_coherent_modal_power_includes_cross_terms():
    m, _ = power_monitor()
    result = m.result
    np.testing.assert_allclose(result["incident_power_per_port_per_length"], 3.5)
    np.testing.assert_allclose(result["accepted_power_per_length"], 3.5 * 0.96)
    np.testing.assert_allclose(result["gain"], 2 * np.pi / (3.5 * 0.96))
    assert np.all(result["gain_valid"])


def test_modal_frequency_matches_single_precision_representation():
    m, port = power_monitor()
    m.frequencies = np.linspace(22e9, 23e9, len(m.frequencies))
    port.result.frequency = m.frequencies.astype(np.float32)
    assert np.all(m.result["gain_valid"])


def test_net_flux_retains_forward_backward_interference():
    m, port = power_monitor()
    port.result.outgoing[:] = 0.2j * port.result.incident
    port.electric_gram += 0.3j * np.eye(2)[None]
    np.testing.assert_allclose(m.result["accepted_power_per_length"], 3.12)


@pytest.mark.parametrize("case", ["window", "frequency", "evanescent", "zero", "other_source"])
def test_modal_gain_masks_unsupported_normalization(case):
    m, port = power_monitor()
    if case == "window":
        m.window_name = "hann"
    elif case == "frequency":
        port.result.frequency *= 1.001
    elif case == "evanescent":
        port.result.power_wave_valid[1] = False
    elif case == "zero":
        port.result.outgoing[:] = port.result.incident
    else:
        m.grid.hertziandipoles = [object()]
    assert not np.any(m.result["gain_valid"])
    assert np.all(np.isnan(m.result["gain"]))


def test_modal_aperture_must_be_strictly_enclosed():
    m = monitor("TM", 2)
    g = m.grid
    g.pmls = {"thickness": dict.fromkeys(("x0", "xmax", "y0", "ymax", "z0", "zmax"), 0)}
    ids = np.ones((6, g.nx + 1, g.ny + 1, g.nz + 1), np.uint32)
    lookup = {c: i for i, c in enumerate(COMPONENT_OFFSETS)}
    port = SimpleNamespace(
        normal_axis=0,
        plane_index=int((m.lower[0] + m.upper[0]) // 2),
        transverse_axes=(1, 2),
        transverse_start=(m.lower[1] + 3, 0),
        transverse_stop=(m.upper[1] - 3, 1),
    )
    g.eigenmodeports = [SimpleNamespace(owner=port)]
    assert m.validate_materials(ids, lookup) == 1
    port.plane_index = m.lower[0] + 1
    with pytest.raises(ValueError, match="modal aperture"):
        m.validate_materials(ids, lookup)


@pytest.mark.parametrize("family", ["TE", "TM"])
def test_adding_contour_does_not_change_antenna_fields(tmp_path, family):
    import h5py
    import gprMax
    from testing.validation.ntff_2d.matched_antenna import antenna_scene

    histories = []
    for enabled in (False, True):
        scene, _ = antenna_scene(family, steps=500)
        if not enabled:
            scene.grid_objects = [obj for obj in scene.grid_objects if not isinstance(obj, gprMax.NTFFContour)]
        scene.add(gprMax.Rx(p1=(0.02, 0.04, np.inf), id="rear_radiation"))
        path = tmp_path / str(enabled)
        gprMax.run(scenes=[scene], outputfile=path, hide_progress_bars=True, log_level=30)
        with h5py.File(path.with_suffix(".h5")) as f:
            histories.append(np.stack([f["rxs/rx1/" + c][...] for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")]))
    np.testing.assert_array_equal(*histories)

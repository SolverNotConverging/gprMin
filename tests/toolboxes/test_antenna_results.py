import csv

import h5py
import numpy as np

from toolboxes.AntennaGUI.results import read_sparameters, read_farfield, discover_results


def test_full_circle_cuts_use_opposite_azimuth_and_close_seam():
    from toolboxes.AntennaGUI.results import polar_cut

    theta, phi = np.meshgrid([0, 90, 180], [0, 90, 180, 270], indexing="ij")
    values = np.array([[0, 0, 0, 0], [1, 2, 3, 4], [5, 5, 5, 5]])
    angle, cut = polar_cut(theta.ravel(), phi.ravel(), values.ravel(), "theta", 0)
    np.testing.assert_allclose(np.rad2deg(angle), [0, 90, 180, 270, 360])
    np.testing.assert_allclose(cut, [0, 1, 5, 3, 0])
    angle, cut = polar_cut(theta.ravel(), phi.ravel(), values.ravel(), "phi", 90)
    np.testing.assert_allclose(np.rad2deg(angle), [0, 90, 180, 270, 360])
    np.testing.assert_allclose(cut, [1, 2, 3, 4, 1])


def test_active_sparameter_validity(tmp_path):
    path = tmp_path / "antenna_active_sparameters.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["frequency_hz", "port", "mode", "active_S_real", "active_S_imag", "coefficient_valid", "power_wave_valid"]
        )
        writer.writerows([[2e9, 1, 2, 3, 4, 1, 0], [1e9, 1, 2, 0, 1, 0, 0]])
    traces = read_sparameters(path)
    assert list(traces) == ["Active S · port 1 mode 2"]
    data = next(iter(traces.values()))
    np.testing.assert_array_equal(data["frequency"], [1e9, 2e9])
    np.testing.assert_array_equal(data["value"], [1j, 3 + 4j])
    np.testing.assert_array_equal(data["coefficient_valid"], [False, True])
    assert not data["power_wave_valid"].any()
    assert discover_results(tmp_path)[0]["name"] == "Active S-parameters"


def test_farfield_invalid_normalization_is_masked(tmp_path):
    path = tmp_path / "antenna.h5"
    group = "ntff/surface/frequency/band/far_field/pattern"
    with h5py.File(path, "w") as f:
        g = f.create_group(group)
        g["theta"] = [0.0, 90.0]
        g["phi"] = [0.0, 0.0]
        g["fields/gain_dbi"] = [[1.0, 2.0]]
        g["fields/radiation_efficiency"] = [0.9]
        g["port_power/gain_valid"] = [False]
    assert len(discover_results(tmp_path)) == 1
    for q in ("gain_dbi", "radiation_efficiency"):
        result = read_farfield(path, group, q, 0)
        assert not result["valid"]
        assert np.isnan(result["value"]).all()

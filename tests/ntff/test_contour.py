"""Independent cylindrical-wave validation of the 2-D contour transform."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.constants import epsilon_0, mu_0
from scipy.special import hankel2

import gprMax
from gprMax.ntff.contour import NTFFContourMonitor, contour_directions, evaluate_contour
from gprMax.ntff.surfaces import COMPONENT_OFFSETS


def monitor(family, axis, dl=0.002, formulation="equivalent_current"):
    size = [round(0.24 / dl)] * 3
    size[axis] = 1 if family == "TM" else 2
    grid = SimpleNamespace(nx=size[0], ny=size[1], nz=size[2], dl=(dl, dl, dl), dt=1e-12, iterations=10)
    lower, upper = [round(0.04 / dl)] * 3, [round(0.20 / dl)] * 3
    lower[axis], upper[axis] = 0, size[axis]
    return NTFFContourMonitor(
        grid,
        f"2D {family}{'xyz'[axis]}",
        lower,
        upper,
        (0.7e9, 1.1e9),
        np.arange(0, 360, 5),
        "test",
        formulation=formulation,
    )


def cylindrical_fields(points, source, k, axis, family, omega, epsilon=epsilon_0, mu=mu_0):
    delta = points - source
    delta[:, axis] = 0
    radius = np.linalg.norm(delta, axis=1)
    scalar = hankel2(0, k * radius)
    gradient = (-k * hankel2(1, k * radius))[:, None] * delta / radius[:, None]
    axial = np.eye(3)[axis]
    if family == "TM":
        return scalar[:, None] * axial, np.cross(axial, gradient) / (1j * omega * mu)
    return -np.cross(axial, gradient) / (1j * omega * epsilon), scalar[:, None] * axial


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("family", ("TE", "TM"))
@pytest.mark.parametrize("formulation", ("ksir", "equivalent_current"))
def test_hankel_complex_amplitude_phase_and_yee_convergence(family, axis, formulation):
    errors = []
    for dl in (0.004, 0.002):
        m = monitor(family, axis, dl, formulation)
        m.configure_background = lambda _: None
        m.epsilon, m.mu = epsilon_0 * 2.3, mu_0 * 1.2
        m.impedance = np.sqrt(m.mu / m.epsilon)
        m.wavenumbers = 2 * np.pi * m.frequencies * np.sqrt(m.mu * m.epsilon)
        source = np.array([0.11, 0.12, 0.13])
        for component, (nodes, _, _) in m._node_maps.items():
            points = (nodes + COMPONENT_OFFSETS[component]) * dl
            accumulator = m._accumulators[component]
            for fi, (freq, k) in enumerate(zip(m.frequencies, m.wavenumbers)):
                e, h = cylindrical_fields(points, source, k, axis, family, 2 * np.pi * freq, m.epsilon, m.mu)
                accumulator.inside_dft[fi] = (e if component[0] == "E" else h)[:, "xyz".index(component[1])]
            accumulator._next_iteration = m.iterations
        m.finalise()
        directions = contour_directions(m.angles, axis)
        expected = (np.sqrt(2 / (np.pi * m.wavenumbers)) * np.exp(0.25j * np.pi))[:, None] * np.exp(
            1j * m.wavenumbers[:, None] * ((source - m.origin) @ directions.T)
        )
        error = np.max(abs(m.result["scalar"] - expected) / abs(expected))
        errors.append(error)
        assert_allclose(m.result["directivity"], 1, atol=0.007)
        expected_power = 2 / m.wavenumbers * (1 / m.impedance if family == "TM" else m.impedance)
        assert_allclose(m.result["radiated_power_per_length"], expected_power, rtol=0.01 * (dl / 0.004) ** 2)
        assert_allclose(m.result["magnetic"], np.cross(directions, m.result["electric"]) / m.impedance, atol=1e-12)
    assert errors[1] < 0.003
    assert errors[1] < errors[0] * 0.3


def test_contour_cancels_external_hankel_source():
    m = monitor("TM", 2, dl=0.001)
    k = np.array([20.0])
    source = np.array([0.3, 0.12, 0])
    delta = m.points - source
    delta[:, 2] = 0
    r = np.linalg.norm(delta, axis=1)
    field = hankel2(0, k[0] * r)[None, :]
    derivative = (-k[0] * hankel2(1, k[0] * r) * np.sum(delta / r[:, None] * m.normals, axis=1))[None, :]
    far = evaluate_contour(
        m.points, m.normals, m.lengths, field, derivative, k, contour_directions(m.angles, 2), m.origin
    )
    assert np.max(abs(far)) < 2e-6


def scene(family, axis, dl=0.002, duration=3e-9, formulation="equivalent_current"):
    domain = [0.16] * 3
    domain[axis] = np.inf
    lower, upper, source = [0.04] * 3, [0.12] * 3, [0.08] * 3
    lower[axis] = upper[axis] = source[axis] = np.inf
    model = gprMax.Scene()
    pml = [10] * 6
    pml[axis] = pml[axis + 3] = 0
    source_class = gprMax.HertzianDipole if family == "TM" else gprMax.MagneticDipole
    for item in (
        gprMax.DomainMode(mode=family),
        gprMax.Domain(p1=domain),
        gprMax.Discretisation(p1=(dl, dl, dl)),
        gprMax.TimeWindow(time=duration),
        gprMax.PMLThickness(thickness=pml),
        gprMax.OMPThreads(n=1),
        gprMax.Waveform(wave_type="ricker", amp=1 if family == "TM" else dl, freq=1e9, id="pulse"),
        source_class(p1=source, polarisation="xyz"[axis], waveform_id="pulse"),
    ):
        model.add(item)
    output = gprMax.NTFFContour(
        lower, upper, "contour", (0.8e9, 1e9, 1.2e9), np.arange(0, 360, 10), formulation=formulation
    )
    model.add(output)
    return model, output


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("family", ("TE", "TM"))
@pytest.mark.parametrize("formulation", ("ksir", "equivalent_current"))
def test_fdtd_line_source_and_hdf5(tmp_path, family, axis, formulation):
    model, output = scene(family, axis, formulation=formulation)
    path = tmp_path / f"{family}{axis}"
    gprMax.run(scenes=[model], outputfile=path, n=1, gpu=None)
    assert_allclose(output.result["directivity"], 1, atol=0.02)
    assert np.all(np.isfinite(output.result["scalar"]))
    assert np.all(output.result["radiated_power_per_length"] > 0)
    expected = analytic_line_sources(output._monitor)
    assert np.max(abs(output.result["scalar"] - expected) / abs(expected)) < 0.02
    with h5py.File(path.with_suffix(".h5")) as data:
        group = data["ntff_contours/contour"]
        assert group["scalar"].shape == (3, 36)
        assert_allclose(group["scalar"][:], output.result["scalar"])
        assert group.attrs["mode"] == f"2D {family}{'xyz'[axis]}"


def analytic_line_sources(m):
    """Continuum line-source solution, using actual impressed-current DFTs.

    No fitted amplitude or phase and no NTFF result enters this reference.
    Electric sources carry I*dl; magnetic sources carry a magnetic moment.
    """
    tm = m.geometry.polarisation == "TM"
    sources = m.grid.hertziandipoles if tm else m.grid.magneticdipoles
    result = np.zeros((len(m.frequencies), len(m.angles)), dtype=complex)
    for source in sources:
        values = (source.waveformvalues_halfdt if tm else source.waveformvalues_wholedt)[: m.iterations]
        times = (np.arange(len(values)) + (0.5 if tm else 0)) * m.dt
        spectrum = m.dt * (np.exp(-2j * np.pi * m.frequencies[:, None] * times) @ values)
        spectrum *= source.dl / m.grid.dl[m.axis] if tm else 1 / m.grid.dl[m.axis]
        position = (np.asarray(source.coord) + COMPONENT_OFFSETS[m.scalar_component]) * m.grid.dl
        phase = np.exp(1j * m.wavenumbers[:, None] * ((position - m.origin) @ contour_directions(m.angles, m.axis).T))
        prefactor = -2j * np.pi * m.frequencies * (m.mu if tm else m.epsilon)
        green = np.exp(-0.25j * np.pi) / np.sqrt(8 * np.pi * m.wavenumbers)
        result += (prefactor * spectrum * green)[:, None] * phase
    return result


@pytest.mark.parametrize("frequencies", [(0,), (-1,), (1e13,), (np.nan,)])
def test_invalid_frequencies(frequencies):
    m = monitor("TM", 2)
    with pytest.raises(ValueError):
        NTFFContourMonitor(m.grid, "2D TMz", m.lower, m.upper, frequencies, (0,), "bad")


@pytest.mark.parametrize("family", ("TE", "TM"))
@pytest.mark.parametrize("formulation", ("ksir", "equivalent_current"))
def test_reject_invalid_materials_pml_and_external_source(family, formulation):
    m = monitor(family, 2, formulation=formulation)
    g = m.grid
    g.pmls = {"thickness": dict.fromkeys(("x0", "xmax", "y0", "ymax", "z0", "zmax"), 0)}
    lookup = {component: index for index, component in enumerate(COMPONENT_OFFSETS)}
    ids = np.ones((6, g.nx + 1, g.ny + 1, g.nz + 1), dtype=np.uint32)
    assert m.validate_materials(ids, lookup) == 1
    material = SimpleNamespace(numID=1, er=2.0, mr=1.0, se=0, sm=0, poles=0)
    m.configure_background([material])
    material.se = 0.1
    with pytest.raises(ValueError, match="lossless"):
        m.configure_background([material])
    ids[lookup[m.scalar_component], 5, 5, m.geometry.live_index] = 2
    with pytest.raises(ValueError, match="exterior"):
        m.validate_materials(ids, lookup)
    ids.fill(1)
    g.pmls["thickness"]["x0"] = 30
    with pytest.raises(ValueError, match="PML"):
        m.validate_materials(ids, lookup)
    g.pmls["thickness"]["x0"] = 0
    g.hertziandipoles = [SimpleNamespace(coord=(5, 5, 0), polarisation="z")]
    with pytest.raises(ValueError, match="enclose"):
        m.validate_materials(ids, lookup)


@pytest.mark.parametrize("formulation", ("ksir", "equivalent_current"))
def test_incomplete_record_is_not_published(formulation):
    m = monitor("TE", 1, formulation=formulation)
    with pytest.raises(ValueError, match="incomplete"):
        m.finalise()
    with pytest.raises(RuntimeError, match="after"):
        _ = m.result


@pytest.mark.parametrize("formulation", ("ksir", "equivalent_current"))
def test_empty_radiation_has_undefined_directivity(formulation):
    m = monitor("TM", 2, formulation=formulation)
    m.surface_material_id = 1
    m.configure_background([SimpleNamespace(numID=1, er=1, mr=1, se=0, sm=0, poles=0)])
    for accumulator in m._accumulators.values():
        accumulator._next_iteration = m.iterations
    m.finalise()
    assert np.all(m.result["scalar"] == 0)
    assert np.all(m.result["radiated_power_per_length"] == 0)
    assert np.all(np.isnan(m.result["directivity"]))

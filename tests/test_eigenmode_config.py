from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.sources as sources_module
from gprMax.eigenmode_config import (
    EigenmodeBandSpec,
    EigenmodeBandpassWaveform,
    EigenmodeMatchSpec,
    EigenmodePortSpec,
    sampled_waveform_spectrum,
)
from gprMax.sources import (
    EigenmodeAnchorMismatchError,
    EigenmodeSource,
    initialise_eigenmode_ports,
)
from gprMax.waveforms import Waveform


def test_automatic_bandpass_tracks_requested_band_and_avoids_dc_nyquist():
    dt = 0.2e-12
    sample_count = 2048
    waveform = EigenmodeBandpassWaveform(
        band_id='wg',
        fmin=45e9,
        fmax=65e9,
        amplitude=1.0,
        dt=dt,
        sample_count=sample_count,
        spectral_threshold=1e-3,
    )

    _, frequencies, spectrum = sampled_waveform_spectrum(waveform, dt, sample_count)
    magnitude = np.abs(spectrum)
    magnitude /= np.max(magnitude)
    passband = (frequencies >= 45e9) & (frequencies <= 65e9)

    assert np.min(magnitude[passband]) > 0.25
    assert magnitude[0] < 1e-3
    assert magnitude[-1] < 1e-3
    assert waveform.significant_low < 45e9
    assert waveform.significant_high > 65e9
    assert waveform.chi < 0.35 * sample_count * dt
    assert np.argmax(np.abs(waveform.samples)) * dt == pytest.approx(
        waveform.chi,
        abs=dt,
    )


def test_bad_custom_waveform_recommends_automatic_bandpass():
    waveform = Waveform()
    waveform.type = 'gaussiandot'
    waveform.amp = 1.0
    waveform.freq = 55e9
    grid = SimpleNamespace(dt=0.2e-12, iterations=2048)
    band = EigenmodeBandSpec(id='wg', fmin=45e9, fmax=65e9, points=81)

    with pytest.raises(ValueError, match='waveform=\'auto\''):
        band.resolve_spectrum(grid, waveform, generated_waveform=False)


def _port(port, anchors):
    return EigenmodePortSpec(
        port=port,
        p1=(0.0, 0.0, 0.0),
        p2=(0.0, 1.0, 1.0),
        normal='x',
        direction='+',
        normal_axis=0,
        transverse_axes=(1, 2),
        invariant_axis=None,
        modes=(1,),
        anchors=anchors,
        plot_fields=None,
    )


def test_all_auto_ports_cover_the_same_significant_spectrum():
    band = EigenmodeBandSpec(id='wg', fmin=45e9, fmax=65e9, points=81)
    band.significant_range = (32e9, 78e9)
    source = _port(1, 'auto')
    receiver = _port(2, 'auto')

    source.resolve_anchors(band, is_source=True)
    receiver.resolve_anchors(band, is_source=False)

    assert source.resolved_anchors[0] == 32e9
    assert source.resolved_anchors[-1] == 78e9
    assert receiver.resolved_anchors == source.resolved_anchors


def test_matched_auto_ports_use_dense_synthesis_anchors_and_exact_centre():
    band = EigenmodeBandSpec(id='wg', fmin=45e9, fmax=65e9, points=81)
    band.significant_range = (32e9, 78e9)
    ordinary = _port(1, 'auto')
    matched = _port(2, 'auto')
    matched.match = EigenmodeMatchSpec(port=2, depth_cells=10)

    ordinary.resolve_anchors(band, is_source=True)
    matched.resolve_anchors(band, is_source=False)

    assert len(matched.resolved_anchors) >= 9
    assert len(matched.resolved_anchors) > len(ordinary.resolved_anchors)
    assert matched.resolved_anchors[0] == 32e9
    assert matched.resolved_anchors[-1] == 78e9
    assert matched.resolved_anchors.count(55e9) == 1


def test_explicit_multiple_anchors_require_coverage_but_single_is_allowed():
    band = EigenmodeBandSpec(id='wg', fmin=45e9, fmax=65e9, points=81)
    band.significant_range = (32e9, 78e9)

    with pytest.raises(ValueError, match='Suggested coverage anchors'):
        _port(1, (45e9, 55e9, 65e9)).resolve_anchors(band, is_source=True)

    single = _port(1, (55e9,))
    single.resolve_anchors(band, is_source=True)
    assert single.resolved_anchors == (55e9,)


def test_matched_explicit_anchors_always_include_exact_band_centre():
    band = EigenmodeBandSpec(id='wg', fmin=45e9, fmax=65e9, points=81)
    band.significant_range = (45e9, 65e9)
    port = _port(1, (45e9, 65e9))
    port.match = EigenmodeMatchSpec(port=1, depth_cells=10)

    port.resolve_anchors(band, is_source=True)

    assert port.resolved_anchors == (45e9, 55e9, 65e9)


def test_matched_near_centre_anchor_is_replaced_by_exact_centre():
    band = EigenmodeBandSpec(id='wg', fmin=45e9, fmax=65e9, points=81)
    band.significant_range = (45e9, 65e9)
    near_centre = np.nextafter(55e9, np.inf)
    port = _port(1, (45e9, near_centre, 65e9))
    port.match = EigenmodeMatchSpec(port=1, depth_cells=10)

    port.resolve_anchors(band, is_source=True)

    assert port.resolved_anchors == (45e9, 55e9, 65e9)


def test_matched_explicit_anchors_must_cover_required_spectrum():
    band = EigenmodeBandSpec(id='wg', fmin=45e9, fmax=65e9, points=81)
    band.significant_range = (40e9, 70e9)
    port = _port(1, (55e9,))
    port.match = EigenmodeMatchSpec(port=1, depth_cells=10)

    with pytest.raises(ValueError, match='verification and boundary-operator synthesis'):
        port.resolve_anchors(band, is_source=True)


def test_auto_anchor_mode_mismatch_falls_back_to_band_centre(monkeypatch):
    source = EigenmodeSource(None)
    source.plane_index = 1
    source.port_index = 3
    source.frequency = 45e9
    source.frequencies = (45e9, 55e9, 65e9)
    source.anchor_policy = 'auto'
    source.fallback_frequency = 55e9
    calls = []

    def fail_broadband(grid, frequencies):
        raise EigenmodeAnchorMismatchError('mode mismatch')

    source._solve_broadband_eigenmode = fail_broadband
    source._extract_frequency_dependent_materials = lambda grid: calls.append('extract')
    source._solve_eigenmode = lambda grid: calls.append('solve')
    source._prepare_single_frequency_injection = lambda grid: calls.append('prepare')
    source._register_port_monitor = lambda grid: calls.append('monitor')
    warnings = []
    monkeypatch.setattr(sources_module.logger, 'warning', warnings.append)

    source.grid_init(SimpleNamespace())

    assert source.frequencies == (55e9,)
    assert calls == ['extract', 'solve', 'prepare', 'monitor']
    assert 'may be inaccurate toward frequencies far from this anchor' in warnings[0]


def test_explicit_anchor_mode_mismatch_remains_an_error():
    with pytest.raises(
        EigenmodeAnchorMismatchError,
        match='single explicit frequency anchor',
    ):
        EigenmodeSource._check_anchor_overlap(
            0.2,
            45e9,
            65e9,
            1,
            'Eigenmode port 1',
        )


class _CoordinatedPort:
    def __init__(
        self,
        port,
        frequencies,
        failure=None,
        match_depth_cells=None,
        anchor_policy='auto',
    ):
        self.port_index = port
        self.frequency = frequencies[0]
        self.frequencies = frequencies
        self.anchor_policy = anchor_policy
        self.requested_anchor_policy = anchor_policy
        self.resolved_anchor_policy = anchor_policy
        self.fallback_frequency = 55e9
        self.spectrum_coverage_policy = 'error'
        self.port_monitor = None
        self.match_depth_cells = match_depth_cells
        self.failure = failure
        self.attempts = []

    def grid_init(self, grid):
        frequencies = tuple(self.frequencies)
        self.attempts.append(frequencies)
        if self.failure is not None and self.failure(frequencies):
            raise EigenmodeAnchorMismatchError(
                'test tracking failure',
                first_frequency=self.failure.first,
                second_frequency=self.failure.second,
                mode_index=2,
                overlap=0.2,
                context=f'Eigenmode port {self.port_index}',
            )
        grid.eigenmodeports.append(self)


def _failure(first, second, predicate):
    predicate.first = first
    predicate.second = second
    return predicate


def test_guard_band_tracking_failure_trims_all_auto_ports(monkeypatch):
    anchors = (32e9, 45e9, 55e9, 65e9, 78e9)
    failure = _failure(32e9, 45e9, lambda values: values[0] == 32e9)
    source = _CoordinatedPort(1, anchors, failure)
    receiver = _CoordinatedPort(2, anchors)
    grid = SimpleNamespace(
        eigenmodesources=[source],
        eigenmodereceivers=[receiver],
        eigenmodeports=[],
        eigenmodeband=EigenmodeBandSpec(
            id='wg',
            fmin=45e9,
            fmax=65e9,
            points=81,
            significant_range=(32e9, 78e9),
        ),
    )
    warnings = []
    monkeypatch.setattr(sources_module.logger, 'warning', warnings.append)

    initialise_eigenmode_ports(grid)

    expected = (45e9, 55e9, 65e9, 78e9)
    assert source.frequencies == expected
    assert receiver.frequencies == expected
    assert source.spectrum_coverage_policy == 'allow'
    assert source.resolved_anchor_policy == 'auto_broadband_guard_trimmed'
    assert receiver.resolved_anchor_policy == 'auto_broadband_guard_trimmed'
    assert 'endpoint modal profile' in warnings[0]


def test_in_band_tracking_failure_falls_back_all_auto_ports(monkeypatch):
    anchors = (32e9, 45e9, 55e9, 65e9, 78e9)
    failure = _failure(45e9, 55e9, lambda values: len(values) > 1)
    source = _CoordinatedPort(1, anchors)
    receiver = _CoordinatedPort(2, anchors, failure)
    grid = SimpleNamespace(
        eigenmodesources=[source],
        eigenmodereceivers=[receiver],
        eigenmodeports=[],
        eigenmodeband=EigenmodeBandSpec(
            id='wg',
            fmin=45e9,
            fmax=65e9,
            points=81,
            significant_range=(32e9, 78e9),
        ),
    )
    warnings = []
    monkeypatch.setattr(sources_module.logger, 'warning', warnings.append)

    initialise_eigenmode_ports(grid)

    assert source.frequencies == (55e9,)
    assert receiver.frequencies == (55e9,)
    assert source.resolved_anchor_policy == 'auto_single_fallback'
    assert receiver.resolved_anchor_policy == 'auto_single_fallback'
    assert 'All automatic eigenmode ports' in warnings[0]


def test_matched_auto_port_tracking_failure_cannot_fall_back():
    anchors = (32e9, 45e9, 55e9, 65e9, 78e9)
    failure = _failure(45e9, 55e9, lambda values: len(values) > 1)
    source = _CoordinatedPort(1, anchors, match_depth_cells=10)
    receiver = _CoordinatedPort(2, anchors, failure)
    grid = SimpleNamespace(
        eigenmodesources=[source],
        eigenmodereceivers=[receiver],
        eigenmodeports=[],
        eigenmodeband=EigenmodeBandSpec(
            id='wg',
            fmin=45e9,
            fmax=65e9,
            points=81,
            significant_range=(32e9, 78e9),
        ),
    )

    with pytest.raises(ValueError, match='verification points'):
        initialise_eigenmode_ports(grid)


def test_matched_explicit_tracking_failure_does_not_suggest_one_anchor():
    anchors = (40e9, 55e9, 70e9)
    failure = _failure(40e9, 55e9, lambda values: True)
    source = _CoordinatedPort(
        1,
        anchors,
        failure,
        match_depth_cells=10,
        anchor_policy='explicit',
    )
    grid = SimpleNamespace(
        eigenmodesources=[source],
        eigenmodereceivers=[],
        eigenmodeports=[],
        eigenmodeband=EigenmodeBandSpec(
            id='wg',
            fmin=45e9,
            fmax=65e9,
            points=81,
            significant_range=(40e9, 70e9),
        ),
    )

    with pytest.raises(ValueError, match='do not replace them with one') as error:
        initialise_eigenmode_ports(grid)

    assert 'if a constant modal basis' not in str(error.value)

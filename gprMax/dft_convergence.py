"""Stop pulsed local FDTD runs when every available spectrum is stable.

Checks use complex DFT differences, never ratios such as S11 alone. Device
readbacks occur only at checkpoints and do not download the Yee field arrays.
The configured time window remains the allocation and execution upper bound.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DFTConvergenceSettings:
    enabled: bool = True
    rtol: float = 1e-3
    check_interval: int = 0
    stable_checks: int = 3
    minimum_time: float = 0.0

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("dft_convergence must be a boolean")
        if not np.isfinite(self.rtol) or not 0 < self.rtol < 1:
            raise ValueError("dft_rtol must be finite and between zero and one")
        for name, minimum in (("check_interval", 0), ("stable_checks", 2)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"DFT {name} must be an integer >= {minimum}")
        if not np.isfinite(self.minimum_time) or self.minimum_time < 0:
            raise ValueError("dft_min_time must be finite and non-negative")


class SpectralStability:
    """Require every complex frequency row to remain stable over a full span.

    Rows may contain many surface samples or modes. Each row is compared in
    its own L2 norm. Weak bins are retained, with a 1e-12 relative floor to
    avoid division by zero. An entirely silent set never declares convergence.
    """

    def __init__(self, rtol, stable_checks):
        self.rtol = rtol
        self.stable_checks = stable_checks
        self.history = []
        self.maximum_error = float("inf")

    def observe(self, spectra):
        current = tuple(np.asarray(values, dtype=np.complex128).copy() for values in spectra)
        if not current or any(not np.all(np.isfinite(x)) for x in current):
            self.history.clear()
            self.maximum_error = float("inf")
            return False
        if self.history and any(a.shape != b.shape for a, b in zip(current, self.history[-1])):
            raise ValueError("DFT convergence spectra changed shape during a run")
        if self.history and len(current) != len(self.history[-1]):
            raise ValueError("DFT convergence monitor set changed during a run")
        self.history.append(current)
        self.history = self.history[-(self.stable_checks + 1):]
        if len(self.history) < self.stable_checks + 1:
            return False
        maximum = 0.0
        active = False
        for index, now in enumerate(current):
            now = now.reshape(now.shape[0], -1)
            norm = np.linalg.norm(now, axis=1)
            peak = float(np.max(norm, initial=0))
            active |= peak > 0
            for previous in self.history[:-1]:
                before = previous[index].reshape(now.shape)
                before_norm = np.linalg.norm(before, axis=1)
                floor = max(peak, float(np.max(before_norm, initial=0))) * 1e-12
                scale = np.maximum(np.maximum(norm, before_norm), floor)
                difference = np.linalg.norm(now - before, axis=1)
                if not (np.all(np.isfinite(scale)) and np.all(np.isfinite(difference))):
                    self.maximum_error = float("inf")
                    return False
                error = np.divide(difference, scale, out=np.zeros_like(scale), where=scale > 0)
                maximum = max(maximum, float(np.max(error, initial=0)))
        self.maximum_error = maximum
        return active and maximum <= self.rtol


def _trace_ports(grid):
    ports = list(getattr(grid, "port_monitors", ()))
    for name in ("transmissionlines", "magneticfrillsources"):
        ports.extend(
            source.port_output for source in getattr(grid, name, ())
            if getattr(source, "port_output", None) is not None
        )
    return ports


def _sources(grid):
    return [source for name in (
        "voltagesources", "hertziandipoles", "magneticdipoles", "transmissionlines",
        "magneticfrillsources", "networkterminals", "eigenmodesources", "discreteplanewaves",
    ) for source in getattr(grid, name, ())]


def _source_end(grid, rtol):
    """Inspect the entire prepared drive, including delayed and repeated pulses."""
    end = 0
    for source in _sources(grid):
        drives = []
        for name in ("waveformvalues_wholedt", "waveformvalues_halfdt", "waveform_whole", "waveform_half"):
            value = getattr(source, name, None)
            if value is not None:
                drives.append(np.asarray(value))
        for name in ("broadband_e_envelopes", "broadband_h_envelopes"):
            value = getattr(source, name, None)
            if value is not None:
                drives.append(np.moveaxis(value, -1, 0))
        if not drives and hasattr(source, "_waveform_value"):
            for offset in (0, source._magnetic_modal_time_offset(grid)):
                drives.append(np.asarray([
                    source._waveform_value(i * grid.dt + offset, grid)
                    if source._source_is_active(i * grid.dt + offset) else 0
                    for i in range(grid.iterations)
                ]))
        if not drives:
            # Unknown source envelopes cannot safely authorize an early stop.
            return grid.iterations
        for drive in drives:
            amplitude = np.max(np.abs(drive.reshape(drive.shape[0], -1)), axis=1)
            if not np.all(np.isfinite(amplitude)):
                return grid.iterations
            peak = np.max(amplitude, initial=0)
            active = np.flatnonzero(amplitude > peak * min(1e-6, rtol * 1e-3))
            if active.size:
                end = max(end, int(active[-1]) + 1)
    return end


def _device_histories(updates, backend):
    """Refresh recorded terminal/receiver histories without finalising monitors."""
    from gprMax.receivers import dtoh_rx_array
    from gprMax.sources import dtoh_transmission_line_outputs, dtoh_magnetic_frill_source_outputs
    from gprMax.network_ports import dtoh_rational_network_outputs

    grid = updates.grid
    def read(name):
        value = getattr(updates, name)
        return value if backend == "metal" else value.get()
    if getattr(grid, "port_monitors", ()) and grid.rxs:
        currents = read("rxcurrents_dev") if updates.nrxcurrent else None
        dtoh_rx_array(read("rxs_dev"), read("rxcoords_dev"), grid, currents)
    if grid.transmissionlines:
        dtoh_transmission_line_outputs(read("tl_Vtotal_dev"), read("tl_Itotal_dev"), grid)
    if grid.magneticfrillsources:
        dtoh_magnetic_frill_source_outputs(
            read("frill_Vinc_dev"), read("frill_Vtotal_dev"), read("frill_Itot_dev"), grid
        )
    if grid.networkterminals:
        dtoh_rational_network_outputs(read("rn_voltage_dev"), read("rn_current_dev"), grid)


def _port_spectra(port, count, maximum):
    """Use the original native FFT bins throughout convergence comparisons."""
    if hasattr(port, "terminal"):
        terminal = port.terminal
        histories = [0.5 * (terminal.voltage[:count] + terminal.voltage[1:count + 1]),
                     terminal.network_current[:count], terminal.generator_voltage[:count]]
        length = maximum
    elif hasattr(port, "receiver"):
        integer_voltage = -port.dl * port.receiver.outputs[port.component][:count]
        if port.hard_source:
            histories = [integer_voltage, port.receiver.outputs[f"I{port.source.polarisation}"][:count]]
            length = maximum
        else:
            histories = [0.5 * (integer_voltage[:-1] + integer_voltage[1:]),
                         port.source.waveformvalues_halfdt[:count - 1]]
            length = maximum - 1
    else:
        source = port.source
        names = ("Vinc", "Iinc", "Vtotal", "Itotal") if hasattr(source, "Itotal") else ("Vinc", "Vtotal", "Itot")
        histories = [getattr(source, name)[:count] for name in names]
        length = maximum
    return [np.fft.rfft(values, n=length)[port._frequency_slice][:, None] for values in histories]


class DFTConvergence:
    def __init__(self, updates, settings, backend):
        self.updates, self.grid, self.settings, self.backend = updates, updates.grid, settings, backend
        grid = self.grid
        self.maximum = getattr(grid, "iterations", 0)
        self.ports = _trace_ports(grid)
        self.ntff = list(getattr(grid, "ntff_monitors", ()))
        self.modal = list(getattr(grid, "eigenmodeports", ()))
        self.sar = list(getattr(grid, "sar_monitors", ()))
        self.enabled = False
        self.reason = "disabled" if not settings.enabled else "no_dft_bins"
        self.checks = 0
        self.stability = SpectralStability(settings.rtol, settings.stable_checks)
        if not settings.enabled or not (self.ports or self.ntff or self.modal or self.sar):
            return
        self.reason = "maximum_time_window"
        if getattr(updates, "is_distributed", False) or hasattr(updates, "updaters"):
            self.reason = "unsupported_coupled_grid"
        elif any(not hasattr(monitor, "frequencies") or not hasattr(monitor, "_accumulators") for monitor in self.ntff):
            self.reason = "time_domain_ntff_requires_full_window"
        elif any(getattr(monitor, "window_name", "rectangular") != "rectangular" for monitor in self.ntff + self.sar):
            self.reason = "fixed_taper_requires_full_window"
        elif any(getattr(monitor, "incident_surface_file", None) is not None for monitor in self.ntff):
            self.reason = "incident_reference_requires_full_window"
        elif self.ports:
            from gprMax import config

            if getattr(config.sim_config, "study", None) is not None:
                self.reason = "study_requires_fixed_native_fft"
        if self.reason != "maximum_time_window":
            logger.info("DFT early stopping inactive: %s", self.reason)
            return
        for port in self.ports:
            if not port.prepared:
                port.prepare(grid)
        # A checkpoint spans at least half a cycle of explicit DFT frequencies.
        # Native FFT bins depend on the maximum duration; comparing them uses
        # fixed zero-padding, with a default checkpoint of 2% of that duration.
        frequencies = [np.asarray(m.frequencies) for m in self.ntff + self.sar]
        frequencies.extend(np.asarray(m.frequency) for m in self.modal)
        positive = np.concatenate([f[f > 0] for f in frequencies]) if frequencies else np.empty(0)
        period_steps = math.ceil(0.5 / (positive.min() * grid.dt)) if positive.size else 0
        self.interval = settings.check_interval or max(64, math.ceil(self.maximum * 0.02), period_steps)
        # Wait two domain crossing times after the final significant source
        # sample. This guards against declaring a quiet, not-yet-reached monitor
        # converged. Users can impose a longer minimum for delayed responses.
        extent = np.asarray(grid.dl) * np.asarray((grid.nx, grid.ny, grid.nz))
        er_mr = max((max(1.0, float(np.max(m.er)) * float(np.max(m.mr))) for m in grid.materials), default=1.0)
        crossing = math.ceil(2 * np.linalg.norm(extent) * math.sqrt(er_mr) / (299792458 * grid.dt))
        self.minimum = max(_source_end(grid, settings.rtol) + crossing,
                           math.ceil(settings.minimum_time / grid.dt), self.interval)
        self.enabled = True
        logger.info("DFT convergence: tolerance %.3g, checkpoint %d steps, %d stable checks; earliest check %d",
                    settings.rtol, self.interval, settings.stable_checks, self.minimum)

    def _spectra(self, count):
        spectra = []
        device = self.backend in ("cuda", "opencl", "metal")
        if device:
            if self.ports:
                _device_histories(self.updates, self.backend)
            if self.modal:
                from gprMax.eigenmode_device import finalise_device_eigenmode_monitors
                finalise_device_eigenmode_monitors(self.modal, self.backend)
            combined = getattr(self.updates, "ntff_collector", None)
            collector = getattr(combined, "frequency", None)
            if self.ntff and collector is None:
                raise RuntimeError("DFT convergence cannot omit the device NTFF accumulators")
            if collector is not None:
                for record in collector.records:
                    a, b, c, d = collector._download(record)
                    spectra.extend([(a + 1j * b).reshape(record.shape), (c + 1j * d).reshape(record.shape)])
                for _, records in collector.incident_records:
                    for record in records:
                        a, b, _, _ = collector._download(record)
                        spectra.append((a + 1j * b).reshape(record.shape))
            collector = getattr(self.updates, "sar_collector", None)
            if self.sar and collector is None:
                raise RuntimeError("DFT convergence cannot omit the device SAR accumulators")
            if collector is not None:
                for record in collector.records:
                    for component in record.components:
                        a, b = collector._download(record, component)
                        spectra.append((a + 1j * b).reshape(record.nfrequencies, component.nedges))
        else:
            for monitor in self.ntff:
                for accumulator in monitor._accumulators.values():
                    spectra.extend([accumulator.inside_dft, accumulator.outside_dft])
                if getattr(monitor, "_incident_electric", None) is not None:
                    spectra.append(monitor._incident_electric)
            for monitor in self.sar:
                spectra.extend(monitor.accumulators.values())
        for monitor in self.modal:
            spectra.extend([monitor.electric_dft, monitor.magnetic_dft])
        for port in self.ports:
            spectra.extend(_port_spectra(port, count, self.maximum))
        return spectra

    def should_stop(self, count):
        if not self.enabled or count < self.minimum or count % self.interval:
            return False
        self.checks += 1
        if self.stability.observe(self._spectra(count)):
            self.reason = "dft_converged"
            logger.info("All DFT bins converged at step %d/%d (maximum relative change %.3g)",
                        count, self.maximum, self.stability.maximum_error)
            return True
        return False

    def finish(self, count):
        grid = self.grid
        grid.completed_iterations = count
        grid.dft_convergence_result = {
            "StopReason": self.reason, "Converged": self.reason == "dft_converged",
            "RelativeTolerance": self.settings.rtol, "Checks": self.checks,
            "MaximumRelativeChange": self.stability.maximum_error,
            "MaximumIterations": self.maximum, "CompletedIterations": count,
            "MaximumTimeWindow": getattr(grid, "timewindow", 0.0),
            "StableChecksRequired": self.settings.stable_checks,
            "CheckInterval": getattr(self, "interval", 0),
            "EarliestCheckIteration": getattr(self, "minimum", 0),
        }
        grid._dft_restore = []
        if self.enabled and self.reason != "dft_converged":
            logger.info("Maximum time window reached at step %d; all-bin DFT convergence was not established", count)
        if count >= self.maximum:
            return
        def replace(obj, name, value):
            grid._dft_restore.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)
        for monitor in self.ntff:
            replace(monitor, "iterations", count)
            for accumulator in monitor._accumulators.values():
                replace(accumulator, "iterations", count)
        for monitor in self.sar:
            replace(monitor, "grid_iterations", count)
            replace(monitor, "_source_samples", monitor._source_samples[:count])
            replace(monitor, "_source_window", monitor._source_window[:count])


def restore_monitor_lengths(grid):
    for obj, name, value in reversed(getattr(grid, "_dft_restore", ())):
        setattr(obj, name, value)
    grid._dft_restore = []


@contextmanager
def completed_output_window(grid):
    """Present only executed samples to output writers; preserve allocations.

    Device finalisation must precede this context because its packed arrays
    retain their original maximum sizes. The temporary views also make native
    port FFT axes reflect the actual record duration rather than padded zeros.
    """
    maximum = grid.iterations
    count = getattr(grid, "completed_iterations", maximum)
    changes = []
    def replace(obj, name, value):
        changes.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)
    try:
        if count < maximum:
            replace(grid, "iterations", count)
            replace(grid, "timewindow", (count - 1) * grid.dt)
            for receiver in grid.rxs:
                replace(receiver, "outputs", {key: value[:count] for key, value in receiver.outputs.items()})
            for source in _sources(grid):
                for name in (
                    "Vinc", "Iinc", "Vtotal", "Itotal", "Itot", "voltage", "network_current", "generator_voltage"
                ):
                    value = getattr(source, name, None)
                    if isinstance(value, np.ndarray) and value.ndim == 1 and value.size in (maximum, maximum + 1):
                        replace(source, name, value[:count + (value.size == maximum + 1)])
            replace(grid, "snapshots", [snap for snap in grid.snapshots if snap.iteration < count])
            for port in _trace_ports(grid):
                port.prepared = False
        yield
    finally:
        for obj, name, value in reversed(changes):
            setattr(obj, name, value)
        restore_monitor_lengths(grid)
        if count < maximum:
            for port in _trace_ports(grid):
                port.prepared = False

DFT convergence and maximum time windows
=======================================

The configured ``TimeWindow`` (or ``#time_window``) is the maximum duration.
By default, a local pulsed simulation can stop earlier when every available
frequency bin of every DFT monitor is stable. Runs without DFT bins keep
their full duration. Automatic checks include modal ports, frequency-domain
NTFF surface and incident-field DFTs, SAR DFTs, voltage-source/network ports,
and automatic transmission-line and magnetic-frill port spectra.

The default relative tolerance is 0.001 (0.1%). Each frequency row is compared
using the L2 norm of its complex change, including phase, across all its
surface samples or modes. Every monitored row must satisfy the tolerance
against all checkpoints in the last three comparison intervals. Weak bins
are included, with a relative floor of 1e-12 of the largest row norm in the
same accumulator; a completely silent set cannot trigger convergence.
This is a numerical stopping criterion, not a guarantee of physical accuracy.

Checks begin after the last significant sample of every prepared excitation,
plus two estimated domain crossing times. A significant source sample exceeds
1e-6 of that drive's peak, or ``dft_rtol * 1e-3`` if smaller. The entire source
record is inspected, including delayed and repeated pulses. Continuous drives
and unknown source envelopes keep running to the maximum. Set a longer minimum
time when the model has known delayed responses, slow propagation, or resonances.

The automatic checkpoint interval is at least 64 timesteps, 2% of the maximum
record, and half a period at the lowest explicitly monitored positive frequency.
Port histories that normally undergo an FFT after the run are compared on a
fixed frequency grid using zero-padding to the maximum record length. Final
native port FFTs use the actual completed record and its frequency resolution;
explicitly configured modal, NTFF and SAR frequencies stay unchanged.

Controls
--------

For a fixed-duration run::

    python -m gprMax model.in --no-dft-convergence

To set the tolerance, checkpoint interval, and minimum simulated time::

    python -m gprMax model.in --dft-rtol 0.001 --dft-check-interval 256 --dft-min-time 2e-9

``--dft-stable-checks`` controls the number of stable comparison intervals
(default three, minimum two). The Python API accepts the equivalent controls::

    gprMax.run(
        scenes=[scene],
        dft_convergence=True,
        dft_rtol=0.001,
        dft_check_interval=256,
        dft_stable_checks=3,
        dft_min_time=2e-9,
    )

Outputs and supported execution
-------------------------------

Receiver and port histories contain only executed samples. Future snapshots
are omitted. HDF5 ``Iterations`` records the actual length, and the
``dft_convergence`` group's attributes preserve the maximum iterations and time
window, actual completed iterations, tolerance, number of checks, largest
relative change, convergence status and stop reason. Reaching the maximum
without convergence is reported as ``maximum_time_window``.

CPU, CUDA, OpenCL and Metal local runs use the same criterion. GPU checkpoints
read DFT accumulators and recorded output histories, and synchronize those
readbacks; they do not copy the full Yee field arrays. This implementation
uses periodic blocking checkpoints rather than asynchronous GPU reductions.
DFT storage still uses the maximum allocation size, so early stopping saves
execution time rather than reducing the initial GPU memory requirement.

Fixed non-rectangular tapers, time-domain NTFF monitors, MPI domain decomposition
and coupled subgrid runs retain the full window and record the reason that early
stopping is inactive. A Hann taper is defined over the full configured duration
and cannot be truncated while preserving that transform. MPI task farming of
independent local models can use convergence normally.

Studies with native FFT ports retain a fixed record so their cases share the
same frequency axis. Incident-surface subtraction also retains the full window;
generate its saved reference with ``--no-dft-convergence`` to preserve the same
sampling signature in both runs.

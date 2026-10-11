.. _ntff-2d:

Two-dimensional NTFF contours
=============================

``NTFFContour`` transforms a closed rectangular contour in a 2-D TE or TM
model into a one-dimensional angular far-field pattern. All three invariant
axes are supported. Choose ``formulation='ksir'`` (alias ``'ramahi'``) for
the scalar Kirchhoff/Ramahi integral, or ``'equivalent_current'`` (default)
for the conventional Love-current integral.

Usage
-----

For a TMz model, add either or both outputs:

.. code-block:: python

    import numpy as np
    import gprMax

    for method in ('ksir', 'equivalent_current'):
        scene.add(gprMax.NTFFContour(
            p1=(0.04, 0.04, np.inf),
            p2=(0.12, 0.12, np.inf),
            id=method,
            frequencies=(0.8e9, 1e9, 1.2e9),
            angles=np.arange(0, 360, 5),
            formulation=method,
        ))

The hash syntax is:

.. code-block:: none

    #ntff_contour: x1 y1 z1 x2 y2 z2 id angle_start angle_stop angle_step f1 [f2 ...] [window] [formulation]

Angles are in degrees, and the stop angle is included. The interval must be
an integer number of positive steps. Frequencies must be positive and below
the temporal Nyquist limit. ``window`` is ``rectangular`` (default) or
``hann``; the optional formulation follows the optional window. Use ``inf``
for both invariant coordinates. ``origin`` is available in the Python API
and defaults to the rectangle centre.
Omitting ``angles`` in Python requests 0 through 359 degrees in one-degree steps.

The angle increases from +u towards +v, where (u,v,invariant) is right-handed:
yz for invariant x, zx for invariant y, and xy for invariant z. The
:download:`complete TMz example <../../examples/features/ntff_2d/tmz_contour.in>`
requests both formulations. Switching ``#domain_mode`` to ``TE`` and using a
z-polarised ``#magnetic_dipole`` gives the corresponding TEz example.

The contour must enclose the impressed sources. Every sampling stencil must
lie outside PML and in one homogeneous, isotropic, lossless, nondispersive
background; the exterior must have the same material. Objects inside the
contour may differ. Modal apertures must also be enclosed, with at least one
cell between their coupling stencil and the contour. Plane-wave excitation, open contours,
layered exteriors, geometry reuse/studies and finite-distance or transient
2-D transforms are not supported by this interface.

Matched-port antennas
---------------------

Runnable inputs are provided for
:download:`TMz <../../examples/features/ntff_2d/matched_tmz.in>` and
:download:`TEz <../../examples/features/ntff_2d/matched_tez.in>`:

.. code-block:: console

    python -m gprMax examples/features/ntff_2d/matched_tmz.in -gpu
    python -m gprMax examples/features/ntff_2d/matched_tez.in -gpu

``EigenmodePort`` with ``VirtualWaveguide`` provides a bidirectional matched
feed for TE and TM antennas on CPU and CUDA. The port, its finite conducting
feed structure, and every radiator must lie inside the contour. The contour
must not cross a feedline. The auxiliary waveguide and its terminating PML
are outside the physical radiation domain. Reduced-mode rear
clearing is restricted to the layer next to the aperture instead
of extending through the homogeneous exterior to the domain boundary.
The finite feed should include its physical rear PEC backing and side walls;
see ``testing/validation/ntff_2d/matched_antenna.py`` for both polarizations.
CUDA virtual guides currently require nondispersive cross-sections without
surface-impedance walls; those restrictions are unchanged from 3-D.

Both contour methods return the same modal power and gain datasets:
``incident_power_per_port_per_length``, ``reflected_power_per_port_per_length``,
``accepted_power_per_port_per_length``, ``accepted_power_per_length``,
``port_ids``, ``port_power_valid``, ``gain_valid``, ``gain``, and
``radiation_efficiency``. For each port, incident and outgoing power are
:math:`a^\dagger W a` and :math:`b^\dagger W b`, including coherent cross-mode
terms in the modal power matrix :math:`W`. Accepted power is the net modal
Poynting flux :math:`\mathrm{Re}[(a-b)^\dagger G_E(a+b)]`, summed over enclosed
ports. It reduces to incident minus outgoing power for power-orthogonal
lossless modes; the full expression retains forward/backward interference
for complex modal impedances. In two dimensions :math:`G(\phi)=2\pi U(\phi)/P'_a`
and radiation efficiency is :math:`P'_r/P'_a`.

Power normalization requires the rectangular DFT window and matching modal
band frequencies. All retained modal power coordinates must be valid and
the accepted power must be positive to define gain. Missing frequencies,
evanescent coordinates, other impressed drives, and negligible accepted
power produce false validity masks and undefined gain, rather than a fitted
normalization. Include enough modes to resolve power carried by the feed.
The saved powers use the raw DFT normalization, with units W s²/m; ratios
such as efficiency and gain are dimensionless. A matched feed absorbs waves
returning from an antenna; it does not imply that the antenna has zero S11.

Formulations and normalization
------------------------------

With time dependence :math:`e^{i\omega t}`, the outgoing 2-D Green function is
:math:`G=-iH_0^{(2)}(kr)/4`. Define

.. math::

    C(k)=\frac{e^{-i\pi/4}}{\sqrt{8\pi k}},\qquad
    F(\alpha)=\lim_{r\to\infty}\sqrt{r}\,e^{ikr}\psi(r,\alpha).

Here :math:`\psi=E_a` for TM and :math:`\psi=H_a` for TE, with invariant axis
:math:`a`. The scalar Ramahi/KSIR transform evaluates

.. math::

    F=C(k)\oint_C e^{ik\hat{s}\cdot(\mathbf r'-\mathbf r_0)}
    \left[ik(\hat{n}\cdot\hat{s})\psi-\partial_n\psi\right]\,d\ell.

Inside/outside samples of the same scalar component give the normal
derivative and their average gives the contour field. Bilinear interpolation
accounts for Yee staggering on the live plane; the invariant storage
thickness is never an integration weight.

The conventional method instead forms :math:`\mathbf J=\hat n\times\mathbf H`
and :math:`\mathbf M=-\hat n\times\mathbf E` at common contour midpoints. It
evaluates the independent vector integral

.. math::

    \mathbf F_E=-ikC(k)\oint_C e^{ik\hat{s}\cdot(\mathbf r'-\mathbf r_0)}
    \left[\eta\big(\mathbf J-\hat{s}(\hat{s}\cdot\mathbf J)\big)
    -\hat{s}\times\mathbf M\right]\,d\ell,
    \qquad \mathbf F_H=\hat{s}\times\mathbf F_E/\eta.

Both methods accumulate :math:`\Delta t\sum_n f(t_n)e^{-i\omega t_n}`.
Electric fields are sampled at :math:`n\Delta t`, magnetic fields at
:math:`(n+1/2)\Delta t`. Complex fields retain the waveform spectrum;
there is no implicit source normalization. For physical monochromatic
amplitudes, divide by the appropriate source spectrum before interpreting
power in watts. With the raw transient DFT, squared-field quantities retain
the corresponding seconds-squared factor.

For monochromatic fields the angular power per invariant length is
:math:`U=|F|^2/(2\eta)` for TM and :math:`U=\eta|F|^2/2` for TE.
The total is :math:`P'=\int_0^{2\pi}U\,d\alpha` in W/m, and 2-D directivity is
:math:`D=2\pi U/P'`. The total uses a separate full-circle quadrature even
when requested angles cover only a cut. Zero radiated power produces
undefined (NaN) directivity. These cylindrical-wave quantities differ from
a cut through a 3-D antenna pattern.

Acceleration and output
-----------------------

CPU collection uses the same Cython/OpenMP DFT accumulator as 3-D NTFF.
CUDA, OpenCL and Metal retain raw contour-node DFTs on the device until
finalization. Both angular evaluations reuse the existing OpenMP kernels
with line weights and the cylindrical Green prefactor. NumPy fallbacks are
retained. Collection follows the solver precision; Metal uses single precision.

MPI CPU domain decomposition assigns every sampled Yee node to one rank,
then gathers phasors onto rank zero after time stepping. There is no
additional contour communication inside the timestep loop. Split either or
both in-plane axes; the invariant axis must have one rank.

HDF5 results are under ``/ntff_contours/<id>/``:

* ``frequencies``, ``angles``, ``origin``, ``points``, ``normals`` and ``lengths``;
* ``scalar`` with shape (frequency, angle), and ``electric``/``magnetic`` with
  shape (frequency, angle, Cartesian component);
* ``radiation_intensity``, ``directivity``, and ``radiated_power_per_length``;
* ``surface_dft`` with component phasors and, for KSIR, ``normal_derivative``.

Attributes record the formulation, polarization/axis, precision, backend,
MPI size, material constants, time window, phase convention and normalization.
The same field arrays are available through the Python object's ``result``
dictionary after solving (on the coordinator for MPI).

Analytical validation
---------------------

``tests/ntff/test_contour.py`` checks translated Hankel waves in a non-vacuum
background, absolute complex amplitude, per-length power, field orientation,
second-order mesh convergence and cancellation of an exterior source. FDTD
line-source tests cover TE/TM along all axes. Accelerator and MPI tests
compare complete solves to CPU outputs.

CPU, CUDA and OpenCL have been exercised in single and double precision on
Windows, and MPI CPU has been compared with serial runs for every TE/TM axis
and both in-plane decomposition directions. Metal shares the device collector
and has a hardware parity test, but has not been hardware-qualified for this
2-D feature on the Windows development host.

Run the independent FDTD validation driver with:

.. code-block:: console

    python -m testing.validation.ntff_2d.validate_contour --output contour_validation

It runs single lines and opposite-phase line pairs in all six modes at
0.8, 1.0 and 1.2 GHz on 4 and 2 mm meshes. Analytical complex fields use
the impressed-current DFT and outgoing Hankel asymptotic, without fitted
amplitude or phase. Each new output directory contains a manifest with source
hashes, error measurements, HDF5 results and ``patterns.png``. ``--backend``
selects CPU, CUDA, OpenCL or Metal; use ``--precision single`` for Metal.

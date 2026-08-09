=======================
Eigenmode port examples
=======================

These four numbered examples form the beginner tutorial in
``docs/source/eigenmode_port.rst``:

``example_1_straight_waveguide``
    Start here. Inspect the two physical guided modes, calculate multimode S11
    and S21, and learn how artificial PEC-boundary modes can appear.

``example_2_curved_waveguide``
    Repeat the workflow for a tight bend and observe reflection and conversion
    from the launched mode into other monitored modes.

``example_3_antenna_and_farfield``
    Feed a pyramidal horn through a rectangular-waveguide eigenmode port and
    calculate S11, a 3D directivity pattern, and E-/H-plane directivity, gain,
    and realized gain.

``example_4_matched_waveguide``
    Terminate both ends of an air-filled parallel-plate waveguide with matched
    modal boundaries. There is deliberately no PML. Plot very low S11 and
    near-0 dB S21, then follow the guided pulse until the output matched face
    absorbs it.

Run every command below from the repository root.

Example 1
=========

.. code-block:: console

    python -m gprMax examples/features/eigenmode_ports/example_1_straight_waveguide/straight_waveguide.in --geometry-only
    python -m gprMax examples/features/eigenmode_ports/example_1_straight_waveguide/straight_waveguide.in -outputfile examples/features/eigenmode_ports/example_1_straight_waveguide/straight_waveguide
    python examples/features/eigenmode_ports/example_1_straight_waveguide/plot_results.py

Example 2
=========

.. code-block:: console

    python -m gprMax examples/features/eigenmode_ports/example_2_curved_waveguide/curved_waveguide.in --geometry-only
    python -m gprMax examples/features/eigenmode_ports/example_2_curved_waveguide/curved_waveguide.in -outputfile examples/features/eigenmode_ports/example_2_curved_waveguide/curved_waveguide
    python examples/features/eigenmode_ports/example_2_curved_waveguide/plot_results.py

Example 3
=========

The 3D horn is more expensive. Eigenmode sources currently require the CPU
solver, so do not add ``-gpu`` to this example command.

.. code-block:: console

    python -m gprMax examples/features/eigenmode_ports/example_3_antenna_and_farfield/horn_antenna.in --geometry-only
    python -m gprMax examples/features/eigenmode_ports/example_3_antenna_and_farfield/horn_antenna.in -outputfile examples/features/eigenmode_ports/example_3_antenna_and_farfield/horn_antenna
    python examples/features/eigenmode_ports/example_3_antenna_and_farfield/plot_results.py

Example 4
=========

This compact 2D example is CPU-only and intentionally sets every PML thickness
to zero:

.. code-block:: console

    python -m gprMax examples/features/eigenmode_ports/example_4_matched_waveguide/matched_waveguide.in --geometry-only
    python -m gprMax examples/features/eigenmode_ports/example_4_matched_waveguide/matched_waveguide.in -outputfile examples/features/eigenmode_ports/example_4_matched_waveguide/matched_waveguide
    python examples/features/eigenmode_ports/example_4_matched_waveguide/plot_results.py

The initial matched-boundary formulation is deliberately narrower than an
ordinary eigenmode port. Its buffer aperture must have one homogeneous,
lossless, nondispersive fill and fixed, effectively real modal profiles with a
real scalar cutoff. A conventional microstrip air/substrate cross-section is
therefore not a valid matched aperture. For microstrip, lossy or dispersive
materials, or frequency-dependent/complex modes, use an ordinary eigenmode
port, continue a uniform feed into a longitudinal PML, and verify convergence
of the termination reflection. A match absorbs only the modes listed by its
port; omitted guided, evanescent, or radiation content can reflect.

Generated CSV, HDF5, VTK-HDF, modal-field, snapshot, and result-plot files are
ignored by Git and can be recreated by rerunning the examples. The larger
validation matrix remains under ``testing/regression/eigenmode_sources``.

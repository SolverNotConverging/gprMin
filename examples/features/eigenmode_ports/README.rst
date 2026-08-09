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
    Terminate both ends of a lossless, shielded 3D microstrip with single-mode
    modal-admittance ADE boundaries. There is deliberately no PML. Plot S11
    and S21, then follow the quasi-TEM pulse until the passive matched face
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

This long 3D example is CPU-only and intentionally sets every PML thickness
to zero. Its 300 mm guide and 18 ns time window make the quasi-TEM packet's
travel, absorption at the passive end, and late ring-down visible:

.. code-block:: console

    python -m gprMax examples/features/eigenmode_ports/example_4_matched_waveguide/matched_waveguide.in --geometry-only
    python -m gprMax examples/features/eigenmode_ports/example_4_matched_waveguide/matched_waveguide.in -outputfile examples/features/eigenmode_ports/example_4_matched_waveguide/matched_waveguide
    python examples/features/eigenmode_ports/example_4_matched_waveguide/plot_results.py

The matched boundary is deliberately narrower than an ordinary eigenmode port.
It supports one mode on a 3D CPU/main-grid model. The matched section must be
longitudinally uniform, and every finite material must be positive, lossless,
and nondispersive; ideal PEC/PMC constraints are allowed. The fixed
centre-frequency E/H profile must remain effectively real and stable over a
relatively narrow band. Lossy, dispersive, multimode, strongly
frequency-dependent, complex-profile, or radiating cases should use an
ordinary eigenmode port followed by a longitudinal PML. A match absorbs only
the retained guided mode. Perturbing the guide can create radiation or omitted
modes, so such a model needs PML.

Generated CSV, HDF5, VTK-HDF, modal-field, snapshot, and result-plot files are
ignored by Git and can be recreated by rerunning the examples. The larger
validation matrix remains under ``testing/regression/eigenmode_sources``.

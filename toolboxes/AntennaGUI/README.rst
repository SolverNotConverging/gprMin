Antenna Studio
==============

Optional PySide6/PyVista desktop antenna editor. OpenCascade constructs CAD
solids; the existing gprMax voxelizer and solver preprocess the model. The GUI
is not imported by the normal solver package.

Installation and launch
-----------------------

From the repository root, create a separate supported Python environment::

    conda env create -f toolboxes/AntennaGUI/environment.yml
    conda activate gprmax-gui
    python -m pip install -e . --no-build-isolation
    python -m toolboxes.AntennaGUI

An existing compatible environment can install Python GUI dependencies with
``pip install -e ".[gui]"``. Install ``pythonocc-core`` through conda-forge as
well; it is not a pip dependency. Building the solver from source requires the
same compiler prerequisites as the rest of gprMax. The desktop development
environment uses Python 3.12; do not upgrade another solver environment in place.

An optional project filename can be passed after the module name. Start with
``File > Examples > Waveguide-fed horn`` or ``Probe-fed patch``. The latter uses
illustrative substrate values and is an integration example, not a calibrated
antenna design.

Modeling workflow
-----------------

* Add boxes/cylinders or a planar sketch, then extrude or revolve a sketch.
  Profiles may be dimensioned rectangles/circles/polylines or a sequence of
  drawn line/circular-arc segments. Arc drawing takes a midpoint and endpoint
  after the preceding segment. The final straight segment closes the profile.
* Use union/subtract/intersect on selected CAD solids. Operations retain the
  target material. Source features remain in the history; enabled consuming
  features replace them in the physical model.
* Edit dimensions in the properties panel and press Apply. Use Edit undo/redo
  to restore a document revision. Right-click a mesh to select its body.
* Use Geometry > Drag / rotate selected body for a VTK transform widget. The
  actor moves immediately; CAD rebuild starts when the mouse is released.
  Numeric transforms are also available. Capturing a surface plane makes a
  detached datum plane, not an associative CAD-face reference.
* Import STEP solids with their original occurrence placements. STL imports
  require explicit source units and a closed surface; they support placement
  but not solid booleans. CAD references are copied into a project's assets
  directory when saving. Names, material references and object tags survive
  renaming/reopening.
* Assign materials in the properties panel. Material and body identities are
  independent. Later physical bodies win on overlap; unoccupied voxels are
  transparent. An explicit air body can replace previously occupied media.
* Rectangular PEC sheets and planar CAD-face PEC sheets are final overlays after
  all solids. Zero-thickness sheets must align with a global coordinate plane
  when building for gprMax; use finite thickness for inclined or non-PEC faces. Curved sheets,
  general sketch constraints, fillets, lofts and assembly constraints are not
  supported by this release.

Coordinates and dimensions are stored in SI units. Length fields are displayed
in mm, times in ns and frequency bands in GHz; explicit modal anchors use Hz.
Port planes must align with the global axes. Simulation settings specify the
common modal frequency band and manual domain/mesh/time values.

Local placement and face extrusion
----------------------------------

Use **Geometry > Local coordinates** to enable a local system or align it by
right-clicking a CAD point, edge or planar face. Point selection snaps to the
nearest CAD vertex, retaining the current orientation. Edge selection places
the origin on the nearest edge and aligns U with its local tangent. Face
selection puts the origin at the picked point and W along the outward normal.
The red/green/blue U/V/W arrows show the active frame. Numerical origin and
orthonormal U/V vectors can also be edited under **Local coordinate system**.

New boxes, cylinders and sketches capture the active coordinate frame; their
property coordinates stay relative to that frame. Changing the active frame
does not move existing objects. Global coordinates and local positions may be
negative. Box sizes, cylinder heights and rectangular sketch dimensions can
also be signed; radii must remain positive. Imported geometry and numeric
transforms continue to use global CAD coordinates.

Choose **Select face for extrusion**, right-click a planar CAD face and enter
a signed thickness in mm. Positive thickness follows the outward normal;
negative thickness goes inward; zero creates a PEC sheet. The original solid
remains, and the new feature can receive its own material or be combined with
the original using a boolean operation. Face references use CAD topology indices
and check orientation during regeneration; reselect faces after topology-changing
edits. STL meshes have no CAD topology for this operation.

Automatic model build
---------------------

Set the requested maximum frequency (and lower band edge) in Simulation, then
press **Build model**. **Automatic build settings** exposes all heuristics:

* The cell size is the smallest of the maximum cell size, the shortest used
  material wavelength at fmax divided by 20, and one third of the smallest
  solid bounding dimension. Thin gaps and local details still need mesh review.
* The device and configured monitor bounds receive an air gap of at least 15
  cells or a quarter free-space wavelength at fmin, whichever is larger, then
  10 PML cells on every side. PML is inside the final domain. The orange outline
  shows its inner boundary. This is an engineering starting point, not a
  guaranteed reflection tolerance.
* CAD coordinates are preserved. A translated solver copy starts at (0,0,0),
  with the offset recorded in ``build_report.json``. The discretized preview
  is translated back to CAD coordinates. Ports and monitors snap to the mesh;
  their movement is reported. ``solver_settings.json`` records the actual
  simulation, port and monitor settings in solver coordinates.
* Automatic run time includes 6/(fmax-fmin), the latest launch delay, and five
  round trips along the domain diagonal at the slowest used dielectric wave
  speed. Change the round-trip count or disable Automatic time for a manual
  time window. Resonant structures may require longer runs and decay checks.
* A configurable cell budget blocks excessive allocations before voxelization.

Run, mode preview and export use the same automatic preparation when enabled.
Existing example projects keep their validated manual meshes until Build model
is selected; new projects default to automatic preparation. Disable Enabled in
Automatic build settings to use the manual simulation values.

These defaults follow the resolution and boundary-clearance principles in the
`gprMax modeling guidance <https://gprmanual.readthedocs.io/en/latest/gprmodelling.html>`_.

Preprocessing and modes
-----------------------

The workflow toolbar separates three operations:

* **Preview solver geometry** builds CAD, voxelizes and runs geometry-only
  preprocessing without requiring ports or excitation.
* **Inspect port modes** builds the configured ports, uses their excitation,
  and opens an interactive modal basis viewer. Unrelated NTFF/snapshot requests
  are omitted from this operation.
* **Validate export** runs complete geometry-only preprocessing including all
  NTFF and snapshot requests, without field time stepping.

Failed CAD or solver jobs retain the last successful viewport. Jobs run in
separate processes, can be cancelled, and are associated with immutable project
revisions. Results from older revisions are discarded. Logs include solver
warnings, including cutoff and anchor tracking diagnostics.

The mode viewer selects physical port, anchor and mode. It displays E/H
magnitudes or complex component real/imaginary/phase maps and transverse vector
arrows. E and H share a display-phase rotation. These are normalized modal
basis fields, not the excited time-domain simulation fields. Below-cutoff and
power-invalid anchors stay labeled as such.

Virtual waveguides remain experimental and require a uniform cross-section on
both sides of the reference plane, including the material stencil. Alternatively
model a guide continuing into PML. An eigenmode port alone is not an absorbing
termination. Conventional NTFF requires its enclosing sampling surface to be in
a supported homogeneous lossless background, clear of material interfaces and
PML. Gain output associates all physical ports and uses rectangular windows.

Use the Design/Solver geometry toggles, transparency, and View > Voxel
cross-section to inspect the actual discretized model. Automatic thin-solid
expansion is disabled. Bodies that disappear block validated export; subcell
dimensions and fully overwritten bodies produce diagnostics. This does not
detect every unresolved local feature or electrical short.

Exports
-------

Export creates a fresh directory beneath the selected destination, preprocesses
the complete model, and writes ``run_antenna.py``, paired ``geometry.h5`` and
``materials.json``, and ``export_manifest.json``. Geometry-only output and mode
artifacts are included for inspection. A sheet-only model has no geometry HDF5.

Run the result independently, from any working directory::

    python /path/to/export/run_antenna.py --geometry-only
    python /path/to/export/run_antenna.py
    python /path/to/export/run_antenna.py --gpu 0 --output /path/to/results/model

The exported script requires gprMax but not the GUI, VTK or OpenCascade. Move
the export directory as a unit. Material properties and run settings can be
edited outside the GUI. Geometry and voxel spacing are compiled properties;
change them in the authoring project and export again. The JSON authoring
document is the editable source, not arbitrary Python script parsing.

Field snapshot regions/times/components can be configured for export and local
runs. Time-domain field animation remains a later stage.

Running and viewing results
---------------------------

Choose **Run simulation** on the toolbar or press F5 to run the current model
on the CPU in an isolated process. The jobs panel shows solver output. Cancel
job stops the active process and retains all earlier completed results. Starting
another job while a solver job is active does not interrupt it. Edits made during
a run are excluded from that run; its results are labeled **older model**.

The left panel contains a **Results** tree grouped by operation and completion
time. Double-click a result (or select it and use **Open result**) to open:

* **Modal fields** after a mode preview, validation, export, or full simulation.
  These are the port basis fields, not the simulated antenna's time-domain field.
* **S-parameters** after a full simulation: trace selection, magnitude in dB or
  linear units, phase, real and imaginary parts. The default view excludes invalid
  power-wave samples; coefficient validity can be selected separately. A single
  source provides one matrix column; simultaneous sources produce active
  S-parameters, not a complete passive S matrix.
* **Far field** for each configured NTFF monitor: frequency and quantity
  selection, angular maps, full-circle polar theta cuts at fixed phi and phi
  cuts at fixed theta, and a rotatable 3D surface. Theta cuts join the selected
  azimuth with its opposite azimuth to cover 360 degrees; an unsampled opposite
  azimuth is interpolated periodically. Phi cuts and 3D surfaces close the
  0/360-degree seam. The 3D radius is normalized linear magnitude/power, while color
  preserves the selected quantity. Invalid power-normalized samples are omitted.

Save the project before running to put results in a sibling ``<project>_results``
directory. The sibling ``<project>.results.json`` index restores the left-panel
history when the project is reopened. It references output folders by absolute
path; retain those folders. Unsaved-project runs use a temporary work directory.
**Simulation > Open completed result folder** can register a retained result
folder again, including one moved to another location. Plot toolbars support
zoom, pan and image export. Select a result to open its output folder.

Developer interfaces
--------------------

``project`` owns versioned persistence and cheap validation. ``cad`` regenerates
OpenCascade shapes and tessellated arrays. ``compiler`` turns the document into
paired voxel/material artifacts and an ordered constructor representation used
by both live Scene creation and Python export. ``worker`` is the isolated job
entry point, with an atomic ``completed.json`` result. ``app`` owns Qt/VTK only.

The optional Python API output request::

    scene.add(gprMax.EigenmodeFieldOutput(filename="port_modes", ports=(1, 2)))

writes ``port_modes.modes.h5`` during model build, including geometry-only runs.
It supports serial 3D main-grid ports. It does not require time-domain result
finalization and does not change existing PNG behavior. It is currently a Python
API request, not a hash-command input-file extension.

The HDF5 schema stores tracked complex Cartesian E/H banks by port, anchor and
selected mode, with per-component native Yee coordinate vectors, effective
indices, normalization/axis metadata, and validity masks. Arrays represent the
positive-normal basis; launch direction is separate metadata. The GUI reader
averages staggered components onto a common transverse cell-center display grid
without modifying the stored fields.

Run the focused integration suite in the GUI environment::

    python -m pytest tests/toolboxes/test_antenna_gui.py tests/toolboxes/test_antenna_gui_widgets.py

Qt tests require a functioning native OpenGL context. Core authoring tests do
not import Qt. CAD/widget tests skip only when their optional dependencies are
absent. Portable binary installers, MPI/subgrid modal-field export, and automated
full S-matrix studies are outside the current desktop release.

To measure native rendering on the target machine, run::

    python -m toolboxes.AntennaGUI.benchmark benchmark.json

This reports GPU-synchronized cached-scene rendering timings and process memory
for approximately one million triangles across 64 actors at 1920 x 1080.
It is a rendering benchmark, not a guarantee of interactive application speed.

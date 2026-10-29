Conformal FDTD development
==========================

Implementation status
---------------------

The internal ``gprMax.conformal`` package retains ordered continuous Scene
geometry, compiles cut edges and connected face regions, and advances a paired
sparse curl operator in Python and Cython. Opaque sheets have independent field
degrees of freedom and SIBC histories on their two sides. The enlargement
planner aggregates compatible open neighbours and checks the resulting timestep.
Production Scenes still use the existing Yee solver. Production stepping,
finite sheet tips, staircase patch rebuilding, source/PML/output integration,
and conformal eigenmode ports remain outstanding. A CUDA operator backend
keeps sparse fields and local SIBC histories on the device; it is not yet
connected to the production solver.

Dielectric averaging remains the existing grid builder's responsibility.
``compile_topology`` consumes its averaged bulk material IDs, permittivity,
permeability, and conductivities. Physical PEC/SIBC geometry supplies the cut
measures and boundary states; it does not replace dielectric averaging.
The supplied bulk grid must be built before conductor masks are applied.
Regression tests compare the uncut operator against the native Cython Yee
update and verify that PEC/SIBC cuts leave the averaged material records
unchanged. Bulk magnetic and SIBC loss use one implicit midpoint solve.

The local ``dft_convergence`` branch was merged into ``master`` at ``0611ed78``
before this work. Its completed-record handling must remain part of the
conformal output path.

Geometry contract
-----------------

Continuous geometry is the source of truth. Resolve object ordering, Boolean
ownership, material assignment, and surface orientation before constructing
field degrees of freedom. Preserve exact primitive coordinates and imported
surface connectivity. A voxel-only import retains its declared voxel geometry;
subcell surfaces cannot be reconstructed from material occupancy alone.

``geometry.cut_cell`` currently accepts planes spanning a Cartesian cell. It
returns oriented closed polyhedra in physical coordinates. Each PEC or SIBC
wall carries a geometric identifier and a side. A zero-thickness opaque sheet
splits the retained domain into separate regions. Opposing sides have independent
ownership even where their coordinates coincide. SIBC model identifiers can be
shared; field traces and ADE histories must remain separate. A diagonal normal
uses the same construction as an axis-aligned normal.

``scene_geometry`` preserves last-object-wins ownership for supported boxes,
spheres, ellipsoids, cylinders, plates, and triangles. ``sections`` uses a common
faceted representation for curved edge and face intersections with a requested
surface error. The topology compiler currently rejects finite sheet tips that
need additional internal transparent edges. Unsupported primitives remain
explicit diagnostics. Small regions remain in the geometry for the subsequent
stability decision.

Discrete operator
-----------------

The reference system uses the paired equations

.. math::

   M_h \dot h = -C e, \qquad M_e \dot e = C^T h.

The compiler must derive the oriented curl matrix and positive Hodge masses
from the same retained topology. Tests cover both prescribed matrices and
operators compiled from continuous geometry. Their transpose pairing gives a discrete energy identity. The
leapfrog invariant at :math:`e^n,h^{n-1/2}` is

.. math::

   W^n = \tfrac12 e^{nT}M_e e^n
       + \tfrac12 h^{n-1/2,T}M_h h^{n-1/2}
       - \tfrac{\Delta t}{2}h^{n-1/2,T}C e^n.

For :math:`A=M_h^{-1/2}CM_e^{-1/2}`, the implementation accepts timesteps
strictly below :math:`2/\sqrt{\lVert A\rVert_1\lVert A\rVert_\infty}`.
This sufficient bound can be conservative. A small face-area threshold alone
does not establish stability.

The implicit surface solve reuses ``SurfaceImpedanceModel.discretise``.
Cut-face boundary circulations supply magnetic trace ports under the
good-conductor approximation of vanishing surface-normal H. Positive weights
give a passive resistive contribution; each sheet side owns its ADE state.
Resistive ports satisfy the discrete Poynting balance in regression tests,
including simultaneous bulk magnetic loss. Axis-aligned and diagonal opaque
sheet tests check isolation. Physical reflection coefficients, convergence,
and dispersive surface energy still require acceptance validation.

Enlargement and fallback
------------------------

``CurlOperator.aggregate_magnetic`` combines connected magnetic control
volumes through a prolongation matrix :math:`P`:

.. math::

   C_c=P^TC, \qquad M_{h,c}=P^TM_hP.

Shared internal circulations cancel and magnetic mass is conserved. The caller
must supply adjacency certified by geometry. Aggregation requires an open
shared face and matching connected-domain labels. An opaque sheet cannot be
crossed to obtain a larger cell. The coarse operator is subjected to the same
timestep check. ``plan_enlargement`` selects connected groups automatically,
restricts bulk loss and magnetic flux, and transports each SIBC trace through
the prolongation while keeping its ADE history separate. Source/output and
eigenmode-port projections through the complete enlarged production
discretisation remain to be done.

The production selection order is:

1. Construct conformal topology, adding independent boundary degrees of freedom
   for opaque sheets and disconnected retained regions.
2. Test the assembled operator at the requested timestep.
3. Enlarge unstable small control volumes using compatible open neighbours and
   retest the resulting operator.
4. Rebuild a consistent staircase patch if a thin gap remains unresolved or
   enlargement cannot produce a valid stable patch. Report its location and
   reason in the run provenance.

A fallback patch must update both sides of its interface, including curl
incidence, masses, SIBC ownership, and source/output projections. Mixing an
independently clamped magnetic coefficient with an unchanged electric update
would break the paired operator. Geometry failures that also prevent a valid
staircase boundary require a diagnostic rather than silently losing a sheet.

Remaining integration and validation
------------------------------------

Connect the compiled sparse degrees of freedom and enlarged operators to the
dense Yee exterior, source and receiver interpolation, PML, modal ports,
NTFF, geometry reuse, and completed-record output. Store boundary provenance,
extra-DOF counts, enlargement groups, and fallback reasons with each run.

CPU acceptance requires staircase recovery for grid-aligned geometry, rotated
PEC cavity eigenfrequencies, PEC scattering, passive SIBC reflection and loss,
opaque-sheet isolation, thin-gap fallback, and long-time stability across mesh
refinement. Compare geometry error separately from time-discretisation error.
The existing voxel SIBC checks should remain until the production conformal
path can represent and validate their formerly rejected cases.

CUDA follows validated CPU integration. Upload identical topology and
coefficients, keep extra field and ADE states resident on the device, preserve
update ordering, and compare single/double precision fields, power, convergence
stopping, and fallback provenance against Cython. Internal kernel agreement
alone is insufficient to enable the production conformal feature.

The internal CUDA regression currently checks diagonal PEC/SIBC boundaries,
bulk electric and magnetic loss, resistive and one-/two-state passive surface
models, nonzero initial ADE histories, and enlarged magnetic cells against
Cython in both field precisions. General coupled magnetic trace matrices
require a device implicit solve; the current local kernel rejects them.

Method references
-----------------

* `A Conformal Thin Boundary Model for FDTD
  <https://eprints.whiterose.ac.uk/id/eprint/131594/>`_ gives relevant context
  for conformal surface models. This implementation's requested sheet model
  is opaque, with independent fields on each side.
* `Uniform stable conformal convolutional perfectly matched layer for enlarged
  cell technique conformal finite-difference time-domain method
  <https://cpb.iphy.ac.cn/article/doi/10.1088/1674-1056/24/2/024101>`_
  identifies conformal PML as a part of the enlarged-cell formulation that
  requires its own validation.

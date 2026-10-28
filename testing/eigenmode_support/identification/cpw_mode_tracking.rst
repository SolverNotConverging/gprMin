CPW odd/even tracking check
==========================

Run from the repository root::

   python -m testing.eigenmode_support.identification.cpw_mode_tracking --output results/cpw_mode_tracking

The experiment exercises the production ``track_solver_bank`` and
``align_groups`` routines with real full-vector Yee-grid eigenfields. It does
not exercise automatic anchor refinement, FDTD injection, port interpolation,
S-parameters, or an independent open-boundary solver. The experiment verifies the production operation-based symmetry pipeline.

Geometry and identification
---------------------------

The finite-ground CPW has a 1 mm centre strip, two 0.5 mm slots, 2 mm ground
strips, and 0.25 mm metal thickness. The lower half of the 12 by 8 mm port
window has relative permittivity 4; the upper half is air. The base mesh is
0.25 mm. In the original geometry the metal occupies y=0 through 0.25 mm,
so its centre is 0.125 mm above the aperture centre. All three conductors are isolated from a surrounding PEC enclosure.
That enclosure introduces an additional common/enclosure branch, retained as
a competing candidate rather than silently excluded. This is a deliberately
coarse diagnostic model, not a dimensioned microwave design.

Even/odd names refer to conductor-potential symmetry: the ordinary CPW mode
has equal ground potentials; the odd slotline mode has opposite ground
potentials. See Ponchak, Papapolymerou, and Tentzeris, IEEE TMTT 53(2), 2005,
DOI 10.1109/TMTT.2004.840571:
https://tentzeris.ece.gatech.edu/TMTT_2005_Asym_CPW.pdf.

Two checks independent of the assignment score identify the branches:

* Full polar-electric/axial-magnetic vector reflection about the centre strip
  gives parity +1 for even and -1 for odd. The horizontal electric component
  alone has the opposite reflection sign.
* Slot voltages integrated in directions from the centre towards the grounds
  have normalized phase correlation +1 for even and -1 for odd. At the reference
  anchor the even CPW branch has much stronger slot voltage than the competing
  even enclosure branch. All solver modes are normalized to forward power.

Results
-------

For nine anchors from 2 through 10 GHz, candidate order is randomly permuted
at every non-reference anchor and every candidate receives a random complex
phase. Six candidates are solved and the three leading reference channels
are tracked. Mode 2 is odd, mode 3 is even CPW, and mode 1 is the enclosure
branch for this reference bank only; these numbers are not general labels.

The substrate case passes: both parity and slot-voltage correlation stay at
+1/-1 within 1e-8; minimum squared E/H overlap is 0.9932337 and maximum combined
residual is 1.124e-8, below the default 1e-7 limit. All tracked profiles have
valid forward power. Additional regression tests repeat order/phase scrambling
with two seeds and a coarser three-anchor bank.

An offset homogeneous-air control tests nearly degenerate TEM channels at 6--10 GHz.
Random unitary mixtures are applied only to the three numerically degenerate
channels; their discrete Maxwell residuals are recomputed after manipulation.
The extra TEM channel is introduced by the enclosing PEC conductor. Automatic
detection now selects ``cs_u`` and the certified source-basis channels have
pure +1/-1 mirror parity. The single left/right mirror is sufficient even
though the metal is offset vertically. A transposed real-Maxwell control
also checks horizontal ``cs_v`` detection and pure final parity.

The detector tests primitive operations once per anchor and constructs the
supported groups from their common intersection. It supports all subgroups
of the transverse square-grid point group. The two-dimensional even sector
still requires additional physical information to distinguish ordinary CPW
and common channels; parity alone does not uniquely choose that basis.

A centered homogeneous-air control restores C2v. The metal occupies
-0.125 through +0.125 mm. A 12 by 8.25 mm window with 33 vertical cells
represents both metal faces symmetrically at the original 0.25 mm spacing;
centering a single-cell-thick slab on an even-cell grid would misrepresent
its thickness or mask symmetry. The dielectric is homogeneous air throughout.

After independent random degenerate rotations, candidate permutations, and
complex phases, automatic detection and alignment both select C2v. All three
final channels have pure left/right parity (+1 or -1), including exactly one
odd channel. Thus degeneracy alone does not cause odd/even separation to
fail: the centered C2v case succeeds with the current implementation.
The two even channels share a symmetry sector, so parity alone still does
not uniquely distinguish the ordinary CPW channel from the enclosure channel.
Regression tests cover this centered result with two random seeds.

At 6 GHz the convergence probes for the original substrate geometry give:

=================  ==========  ==========
Case               Even neff   Odd neff
=================  ==========  ==========
Base mesh          1.361684    1.462080
0.125 mm mesh      1.404794    1.489540
16 by 12 mm window 1.361653    1.467037
=================  ==========  ==========

Halving the grid changes indices by 3.17% and 1.88%; enlarging the aperture
changes them by 0.0023% and 0.34%. These are sensitivity probes, not a completed
convergence study. Tracking success does not establish propagation-constant
accuracy or remove finite-window effects.

``report.json`` stores all anchor diagnostics and ``tracking.png`` compares
continuity and parity for the substrate, offset degenerate, and centered
degenerate cases.

# Rational positive-real modal admittance matched port

## Purpose

This file is the living implementation plan and engineering log for replacing
the centre-frequency constant modal admittance in the matched eigenmode port
with a fitted, globally passive rational characteristic admittance.

The implementation must improve finite-band impedance matching without
weakening the stability guarantee of the existing raw-Yee power-adjoint
boundary. Accuracy is required over the declared and verified spectrum;
outside that spectrum the boundary may reflect, but it must never generate
energy.

This work is a new local modal-admittance formulation. It is not the earlier
modal-translation FIR formulation and does not reintroduce a longitudinal
convolution kernel.

## Current status

| Layer | Status | Production use |
|---|---|---|
| Fixed-gauge E/H admittance samples | Implemented and tested | Diagnostics |
| Deterministic scalar vector fitting | Implemented and tested | Shadow only |
| Global scalar positive-real search/repair | Implemented; independent KYP cross-check pending | Shadow only |
| Real-state trapezoidal rational ADE | Implemented and tested | Experimental switch only |
| Historical experimental 5 ns and 12 ns scalar microstrip stability | Passed before the fixed-profile release gate | Diagnostic only |
| Match improvement over constant ADE | Failed; both remain about -16.2 dB worst S11 | Blocks default enablement |
| One-way raw-Yee terminal oracle | Implemented; exposes a roughly 2% FDFD/time-domain ratio gap | Test/diagnostics |
| Scalar fixed-profile runtime gate | Implemented; the microstrip's roughly 2% residual is rejected | Mandatory |
| Scalar fitted model as default | Blocked | Disabled |
| Hybrid/MIMO discrete DtN extension | Planned after scalar failure isolation | Not implemented |

## Non-negotiable design decisions

1. Retain the existing raw-Yee magnetic extraction and electric
   reconstruction. They are an exact power-adjoint pair and are the foundation
   of the coupled FDTD-boundary energy argument.
2. Retain the ordinary FDTD cells between the reference plane and the outer
   boundary. They provide the physical propagation delay and numerical
   dispersion; the fitted load is local to the outer boundary.
3. Fit only the characteristic modal admittance. Keep the terminal Yee
   half-cell storage as an analytic positive term rather than asking a
   finite-band fit to infer it.
4. Use a fixed centre-frequency modal voltage/current gauge when forming
   frequency samples. Independently power-normalizing every anchor would make
   every sampled admittance equal to one and would destroy the information to
   be fitted.
5. Stable poles are necessary but not sufficient. Every accepted model must
   pass a global positive-real certificate after fitting, after passivity
   enforcement, after serialization, and in the runtime precision.
6. Passivity enforcement is mandatory and cannot be disabled by the user.
7. Use a passive trapezoidal/Tustin realization aligned with the Yee time
   staggering. Do not use an ad hoc explicit IIR update.
8. Preserve the present constant-admittance model as the exact order-zero
   regression case.
9. Version 1 remains scalar, one-mode, 3D, longitudinally uniform, lossless,
   nondispersive, and fixed-profile. Multimode/MIMO and varying spatial bases
   are later work.

## Mathematical model

The terminal model is

\[
Y_{\mathrm{term}}(s) = sE_h + Y_c(s),
\qquad
Y_c(s) = D + \sum_{k=1}^{K}\frac{r_k}{s-p_k},
\]

with

- \(\operatorname{Re}p_k < 0\);
- real poles/residues or complex-conjugate pairs;
- \(E_h > 0\), the known terminal Yee half-cell storage;
- \(D=Y_\infty>0\), the passive high-frequency completion of the
  characteristic admittance;
- \(\operatorname{Re}Y_c(j\omega)\ge\epsilon\) globally.

For the first implementation the modal coordinates are normalized so that
\(Y_c(j\omega_c)=1\), and the default high-frequency completion is
\(Y_\infty=1\). The rational residual therefore tends to zero outside the
fitted band. The total terminal admittance still contains \(sE_h\); this term
is required to control the Yee half-cell and the Nyquist limit.

### Active and passive boundary equations

Let \(V=a+b\), let \(Q\) be the direction-normalized magnetic coefficient,
and define outward current \(I_{\mathrm{out}}=-Q\). The boundary equation is

\[
E_h\dot V + Y_c(s)(V-2a) = I_{\mathrm{out}}.
\]

For a passive boundary, \(a=0\). When \(K=0\), \(D=1\), and \(E_h=\tau\),
this reduces exactly to the existing recurrence

\[
\tau\dot V+V=2a-Q.
\]

The incident waveform must therefore pass through the same fitted
characteristic-admittance dynamics. Retaining an unfiltered \(2a\) term would
make the load passive but the active generator frequency-mismatched.

## Fixed-coordinate admittance samples

Let \(F_0\) be the flattened native tangential-E basis at the exact centre
frequency, let \(J_0\) be its raw-Yee magnetic power covector, and let

\[
W=F_0J_0^T>0.
\]

After jointly phase-aligning every anchor E/H pair \((E_i,J_i)\) to the centre
pair, define

\[
v_i=\frac{E_iJ_0^T}{W},
\qquad
i_i=\frac{F_0J_i^T}{W},
\qquad
Y_i=\frac{i_i}{v_i}.
\]

This construction must be invariant to an arbitrary joint complex scaling of
an anchor E/H pair, must give \(Y(\omega_c)=1\), and must retain changes in the
relative E/H magnitude.

Also record separate residuals

\[
\frac{\lVert E_i-v_iF_0\rVert}{\lVert E_i\rVert},
\qquad
\frac{\lVert J_i-i_iJ_0\rVert}{\lVert J_i\rVert}.
\]

A scalar admittance fit cannot compensate for a frequency-dependent spatial
profile. Excessive residual or failed mode tracking is a hard rejection.

## Sampling and model-order policy

### Automatic anchors

1. Begin with approximately 9--13 deterministic Chebyshev/log-spaced modal
   solves over the significant source spectrum plus a guard band.
2. Keep the exact centre frequency as a mandatory solve and basis frequency.
3. Split anchors into fitting and interleaved validation sets.
4. Fit candidate pole counts \(K=0,2,4,\ldots\), where a conjugate pair counts
   as two poles.
5. Require an overdetermined fit, initially
   \(N_{\mathrm{train}}\ge 2K+4\), plus independent validation anchors.
6. Add an eigenmode solve near the largest validation/model disagreement and
   repeat.
7. Select the smallest order meeting the fit, holdout, mismatch, stability,
   and passivity requirements.
8. Initially cap the model at 8--12 poles. Reject a band that needs a more
   fragile model.

### Explicit anchors

- A single explicit frequency can only use \(K=0\). It cannot identify a
  genuine rational model.
- Multiple explicit anchors may use only an order supported by their number
  and coverage.
- If the requested order is underdetermined, reject with the required anchor
  count and recommended frequencies; never invent poles from one solve.

## Vector-fitting pipeline

1. Scale frequency by \(\omega_c\) and scale admittance to order-one values.
2. Add the negative-frequency conjugates required by a real time-domain model.
3. Seed stable negative-real and lightly damped conjugate poles
   logarithmically through the fitted and transition bands.
4. Run relaxed vector-fitting/Sanathanan--Koerner pole relocation.
5. Reflect unstable relocated poles into the open left half-plane.
6. Enforce conjugate pairing and a minimum damping margin.
7. Iterate until both pole movement and weighted response error converge.
8. With the poles fixed, solve for residues and \(D\) under realness and
   reciprocity constraints.
9. Drop negligible residues, merge nearly duplicate poles, and refit.
10. Evaluate weighted RMS error, worst holdout error, and the
    reflection-equivalent mismatch

    \[
    \left|\frac{Y_{\mathrm{fit}}-Y_{\mathrm{sample}}}
    {Y_{\mathrm{fit}}+Y_{\mathrm{sample}}}\right|.
    \]

The initial pole locations are deterministic numerical seeds. The final
\(p_k\) are vector-fitted. The final \(r_k\) and \(D\) are obtained by a
fixed-pole constrained solve.

## Global positive-real verification and enforcement

For the scalar model require

\[
\operatorname{Re}Y_c(j\omega)\ge\epsilon
\quad\text{for every real }\omega.
\]

A dense sweep is useful diagnostics but is not a certificate. The accepted
model must pass a continuous-time Hamiltonian/KYP positive-real test.

Enforcement procedure:

1. Keep the stable fitted poles fixed.
2. Locate every passivity-violation interval and its most-negative value.
3. Perturb residues and \(D\), preserving conjugacy and realness.
4. Minimize weighted change from the unconstrained fit subject to linear
   constraints \(\operatorname{Re}Y(j\omega_q)\ge\epsilon\) at the violating
   frequencies.
5. Repeat violation discovery and constrained correction until the global
   Hamiltonian/KYP certificate passes.
6. Require \(D\ge\epsilon\) and \(E_h\ge0\) independently.
7. If the correction exceeds the allowed in-band error, refine anchors or
   increase model order. Do not silently accept a badly distorted model.
8. Repeat certification after coefficient serialization and in every runtime
   precision.

The scalar fixed-pole constraints are linear in the real residue/direct-term
parameters, so version 1 can use NumPy/SciPy without introducing a mandatory
semidefinite-programming dependency.

## Real state-space ADE

Realize the proper part as

\[
\dot x=A x+B(V-2a),
\qquad
Y_c(V-2a)=Cx+D(V-2a).
\]

Real poles use scalar states; conjugate pole pairs use real 2-by-2 blocks.
Apply trapezoidal/Tustin integration. Each pole maps to

\[
z_k=\frac{1+p_k\Delta t/2}{1-p_k\Delta t/2},
\]

which remains inside the unit circle when \(\operatorname{Re}p_k<0\).

The coupled voltage/state solve must be derived from the midpoint boundary
power equation rather than evaluated explicitly. Runtime cost is \(O(K)\)
state and work per port per step, with no growing FIR history.

Prewarp physical modal frequencies before continuous fitting when the final
model will be used through Tustin:

\[
\Omega_i=\frac{2}{\Delta t}\tan\left(\frac{\omega_i\Delta t}{2}\right).
\]

## Code organization

- `gprMax/modal_admittance.py`
  - model/sample dataclasses;
  - vector fitting, order selection, fit diagnostics, and the Yee-staggered
    characteristic-data transform.
- `gprMax/modal_admittance_passivity.py`
  - scalar global positive-real search;
  - fixed-pole residue/direct-term repair;
  - passivity certificate and repair diagnostics.
- `gprMax/modal_admittance_ade.py`
  - real state-space conversion;
  - Tustin/prewarp helpers;
  - coupled passive voltage/state update.
- `gprMax/matched_eigenmode_ports.py`
  - retain geometry/material validation and raw-Yee field coupling;
  - add anchor admittance extraction and fitted-model construction;
  - retain the scalar constant-admittance timestep as the default while the
    rational path remains experimental.
- `gprMax/cython/matched_eigenmode.pyx`
  - later real-state update and allocation-free raw-H projection/E write.
- `gprMax/eigenmode_ports.py`
  - HDF5 fit samples, coefficients, errors, and passivity diagnostics.

The public `#eigenmode_match: <port> <depth_cells>` syntax remains valid.
Automatic order selection is the default. An advanced explicit order may be
added later, but passivity enforcement will not have a disable switch and
users will not directly enter poles/residues.

## Metadata to record

- formulation and fitter version;
- basis and synthesis frequency ranges;
- anchor frequencies and complex sampled admittances;
- electric and magnetic fixed-basis residuals;
- continuous poles, residues, and direct term;
- half-cell storage coefficient;
- algebraic pole count and real-state count;
- train, holdout, maximum, and reflection-equivalent errors;
- pole stability margin;
- raw and final passivity margins;
- passivity correction norm and algorithm;
- Tustin/prewarp settings and discrete poles;
- passivity certificate storage matrix when available.

## Validation and release gates

### Fast mathematical tests

- recover known passive one-real-pole and RLC/conjugate-pair responses;
- deterministic fit under scaling, noise, and nonuniform frequency samples;
- compare transfer functions rather than pole identity for nearly cancelling
  models;
- detect RHP poles, negative resistance, broken conjugacy, and narrow
  unsampled passivity violations;
- correct a mildly nonpassive fit and reject a strongly inconsistent target;
- verify continuous and discrete positive-real certificates;
- verify order-zero equivalence with the current recurrence.

### Modal-coordinate tests

- arbitrary joint complex rescaling of anchor E/H leaves \(Y_i\) unchanged;
- exact centre sample is one;
- both port directions and all three axes give identical admittance;
- analytic TEM/TE fixtures give the expected characteristic admittance;
- changing profile, cutoff crossing, degeneracy, or poor Gram conditioning is
  rejected.

### ADE and coupled-grid energy tests

- impulse, chirp, random, and alternating-sign/Nyquist drives;
- zero-input state decay and reset;
- real-block conjugate realization and pole-order invariance;
- source-off ADE storage plus Yee grid energy does not grow beyond scaled
  roundoff;
- boundary work equals the energy removed from the Yee grid;
- single and double precision, Courant sweeps, all axes and both directions.

### End-to-end tests

- analytic uniform guide against an ordinary eigenmode-port-plus-PML
  reference;
- PML-free microstrip with independent holdout frequencies;
- meaningful improvement over the current approximately -16 dB worst-case
  microstrip S11, initially targeting -25 to -30 dB without power imbalance;
- at least 50--100 ns source-off microstrip stability;
- a broadband/Nyquist stress excitation that may reflect but never grows;
- mesh and timestep refinement;
- the 300 mm, 18 ns Example 4 as a slow/nightly propagation-and-absorption
  regression.

No rational matched port becomes the default until the global passivity
certificate and coupled Yee-plus-ADE energy gates pass.

## Initial and future scope

Version 1 keeps these restrictions:

- one isolated propagating mode;
- 3D CPU main-grid execution;
- longitudinally uniform buffer;
- positive scalar, lossless, nondispersive materials plus supported ideal
  conductors;
- an effectively real and stable E/H spatial profile across the fitted band;
- no unrepresented radiation, important evanescent content, or mode crossing.

Future MIMO work will power-whiten the modal Gram and fit

\[
Y(s)=D+\sum_k\frac{R_k}{s-p_k}
\]

with common poles, reciprocal matrix residues, and a matrix KYP/passivity
certificate. A changing spatial basis or radiating aperture requires a larger
reduced spatial DtN model and is outside this implementation.

## Revised hybrid-guide phase after the scalar experiment

The scalar experiment established that a passive pole-residue ADE is
implementable, but also that fitting a single E/H amplitude ratio is not enough
to make the tested hybrid microstrip accurately transparent. The next phase
must therefore be a reduced *matrix* discrete Dirichlet-to-Neumann model, not a
higher-order scalar curve fit.

1. Form a reduced tangential boundary basis from the retained guided mode,
   centre-to-anchor profile residuals, and selected evanescent port modes. Use
   a power-weighted SVD/POD and retain only directions above a declared energy
   threshold.
2. Build raw-Yee electric synthesis and magnetic extraction operators as exact
   adjoints. Power-whiten their Gram so the reduced boundary supply rate is
   simply \(v^T i\).
3. At every synthesis frequency compute the full reduced map
   \(I_i=Y_iV_i\). The columns must come from independent boundary solutions;
   independently normalized eigenmodes are not enough to identify off-diagonal
   coupling.
4. Use the fully discrete temporal frequency and longitudinal Yee phase when
   forming the target. Keep any analytically known boundary mass/storage matrix
   outside the proper rational fit.
5. Fit a common-pole reciprocal model

   \[
   Y_c(s)=D+\sum_k\frac{R_k}{s-p_k},
   \qquad D=D^T,\quad R_k=R_k^T,
   \]

   using interleaved held-out frequency solves. Increase reduced basis size
   before increasing pole order when spatial residual dominates.
6. Require a matrix positive-real certificate
   \((Y(j\omega)+Y(j\omega)^H)/2\succeq\epsilon I\) globally. The release
   implementation needs a Hamiltonian/KYP certificate and a deterministic
   passivity-enforcement solve; sampled singular-value clipping is not an
   acceptable substitute.
7. Realize the certified model as a real MIMO state-space ADE and solve the
   small coupled boundary-voltage system implicitly at each E update. Include
   the ADE certificate storage in the coupled Yee energy regression.
8. Re-run the one-way oracle, passive-end reflection, both-end S parameters,
   Nyquist/noise stress, and at least 50--100 ns ring-down. Only then consider
   replacing the current scalar default.

This phase is intentionally not started by silently fitting the observed
two-percent one-way error: doing so would make the test geometry part of the
model-construction loop and would still leave the larger hybrid boundary
residual unexplained.

## Engineering log

### 2026-08-09: plan established

Achieved:

- froze the mathematical target as a fitted characteristic admittance plus
  analytic Yee half-cell storage;
- defined fixed-coordinate anchor samples that retain impedance variation;
- selected relaxed vector fitting with adaptive model order;
- made global positive-real certification and enforcement release blockers;
- derived the active-source relationship requiring the incident waveform to
  pass through the same fitted characteristic admittance;
- defined the Tustin real-state ADE and staged validation plan;
- preserved the current order-zero constant-admittance boundary as the
  regression oracle.

Known risks and unresolved work:

- the exact scalar Hamiltonian/KYP verifier and constrained residue correction
  still need implementation and numerical conditioning tests;
- automatic anchor refinement requires coordination with the existing modal
  solve/tracking lifecycle;
- fixed-coordinate admittance samples must be verified against analytic guides
  before they are allowed to drive a boundary;
- the passivity proof must include the actual raw-Yee coupling and runtime
  precision, not only the standalone rational model;
- high-order fits, near-cutoff modes, and rapidly changing profiles may need to
  be rejected rather than approximated.

Next implementation milestone:

1. standalone rational model and evaluation;
2. scalar vector fitting and synthetic recovery tests;
3. exact positive-real verifier and enforcement loop;
4. standalone Tustin ADE with energy tests;
5. shadow-mode extraction of microstrip anchor admittance;
6. boundary integration only after the preceding gates pass.

### 2026-08-09: fixed-coordinate sampling implemented in shadow mode

Achieved:

- added a reusable fixed-power-gauge sampling function that returns complex
  modal voltage, current, admittance, and separate electric/magnetic spatial
  residuals;
- made the projection invariant to arbitrary joint complex scaling and phase
  of each eigenmode anchor pair;
- integrated diagnostic sample extraction into matched-boundary preparation
  without changing the production order-zero timestep;
- added an exact centre-admittance check, which fails initialization if the
  fixed coordinate does not produce \(Y(\omega_c)=1\);
- added tests for complex admittance recovery, gauge invariance, out-of-span
  profile residuals, malformed arrays, non-positive power pairing, and zero
  projected voltage;
- ran the new sampling tests together with the existing modal-admittance
  boundary tests: 30 tests passed.

Observed limitations:

- sample extraction is currently diagnostic-only and is intentionally not
  allowed to alter the boundary update;
- the existing automatic anchor set has not yet been densified for rational
  identification;
- the extracted microstrip samples still need an end-to-end comparison with a
  reference characteristic admittance before enabling a fitted load;
- separate profile residual rejection thresholds have not yet replaced the
  established joint-overlap guard.

Microstrip shadow-run result:

- both the 5 ns scattering regression and the 12 ns late-growth regression
  still pass with diagnostic extraction enabled;
- six automatic anchors spanning approximately 3.23--5.77 GHz produced
  fixed-coordinate characteristic-admittance magnitudes of only
  0.998172--1.00234 at both ends;
- this variation is much smaller than the existing approximately -16 dB
  reflection. Consequently, frequency variation of the scalar characteristic
  admittance alone may not be the dominant mismatch mechanism;
- before claiming that a rational fit improves the microstrip port, measure
  the one-way numerical terminal relation using the exact raw-Yee boundary
  operators and separate terminal-load error from active-source, staggering,
  finite-window, and S-parameter-reference-plane error;
- a rational fitter that accurately reproduces these nearly constant samples
  but does not improve S11 is a valid negative result and must not be hidden by
  empirical, non-passive tuning of the half-cell term.

Additional regression result:

- the existing matched-port geometry, material, API, and 3D runtime suites were
  rerun after adding shadow extraction: 24 tests passed. This confirms that the
  new diagnostics have not changed the certified order-zero field update.

### 2026-08-09: standalone rational fit, passivity, and ADE stack

Achieved:

- added a validated scalar pole-residue model with deterministic evaluation;
- added real-structured fixed-pole least squares for real poles and explicit
  conjugate pairs;
- added deterministic stable pole seeding and iterative scalar vector-fitting
  pole relocation with left-half-plane reflection, exact conjugate pairing,
  frequency/admittance scaling, and fit diagnostics;
- added global scalar positive-real analysis based on the exact rational
  polynomial for \(\operatorname{Re}Y(j\omega)\) in \(x=\omega^2\), including
  stationary points and sign intervals rather than relying on a dense sweep;
- added fixed-pole passivity repair that perturbs residues and the direct term
  through a cutting-plane sequence of constrained quadratic problems;
- added real state-space conversion for real and conjugate pole-residue terms;
- added the coupled trapezoidal rational ADE for
  \(E_h\dot V+Y_c(s)(V-2a)=I_{\mathrm{out}}\);
- verified that the zero-pole, \(D=1\) ADE reproduces the existing scalar
  matched-boundary recurrence sample-for-sample;
- added bilinear prewarp, reset, continuous/discrete response, active-source
  equilibrium, passive decay, and Nyquist stress coverage;
- ran the complete standalone sampling/fitting/passivity/ADE set: 34 tests
  passed.

Failures and remaining risks:

- the first delegated fitter implementation did not deliver a file and was
  interrupted because it blocked the passivity and ADE modules; a replacement
  core fitter was implemented and tested locally;
- the current vector fitter is the classical scalar relocation equation, not
  yet the full relaxed VF variant. It succeeds on exact synthetic real-pole
  and conjugate-pair models, but noisy and ill-conditioned modal data still
  need stress testing;
- the scalar polynomial passivity certificate catches extremely narrow
  violations missed by a 1001-point sweep, but high-order polynomial root
  conditioning remains a risk. A state-space Hamiltonian/KYP cross-check is
  still required before production acceptance;
- passivity repair presently uses SciPy SLSQP for the constrained quadratic
  problem. Determinism, scaling, and failure behavior need a broader test set;
- no fitted model is connected to the FDTD boundary yet. The production update
  remains the established order-zero model.

### 2026-08-09: dense real-anchor shadow synthesis

Achieved:

- increased automatic matched-port sampling to at least nine generated
  synthesis anchors while preserving the existing sparse policy for ordinary
  eigenmode ports;
- the tested microstrip now resolves 12 anchors over approximately
  3.23--5.77 GHz, including the exact 4.5 GHz centre;
- added smallest-order passive synthesis orchestration with independent
  validation data, mandatory fixed-pole passivity repair, correction limits,
  and explicit failure diagnostics;
- connected this pipeline in diagnostic-only shadow mode during matched-port
  initialization;
- wrote the complex fixed-basis admittance samples, separate profile
  residuals, shadow status, failure reason, and any certified shadow
  coefficients/passivity margin to HDF5;
- retained the order-zero production boundary regardless of shadow outcome.

Shadow-fit failure and interpretation:

- the microstrip samples are almost purely real and monotonic:
  \(\operatorname{Re}Y\) rises from 0.998172 to 1.002341 while
  \(\operatorname{Im}Y\) remains at numerical roundoff;
- an order-zero \(D=1\) model missed held-out samples by approximately
  \(1.54\times10^{-3}\), above the deliberately strict shadow threshold;
- the two-pole classical vector fit was non-passive and required an
  approximately 99.4% parameter change to repair, so it was correctly rejected
  and never enabled;
- the fitted two-pole model developed an extremely lightly damped pair and a
  negative real-admittance notch outside the sample band. This is precisely the
  failure that global passivity enforcement is intended to catch;
- the result suggests that independently real, frequency-varying
  characteristic-admittance samples from a lossless guide are difficult to
  approximate with a low-order strictly stable rational positive-real model.
  More importantly, their 0.2% variation still cannot explain the existing
  approximately 15% reflected amplitude;
- next work must measure the actual one-way discrete terminal relation and
  attribute the reactive mismatch before increasing model order or enabling a
  fitted boundary. Forcing a repaired model through this gate would be
  scientifically unjustified.

Verification checkpoint:

- 98 focused tests now pass, covering the new mathematical stack, fixed-basis
  sampling, existing modal-boundary energy/sign behavior, eigenmode anchor
  configuration, matched-port geometry/material validation, 5 ns microstrip
  scattering, and 12 ns microstrip late-time stability;
- `py_compile` passes for all new and modified production modules;
- `git diff --check` passes;
- at this checkpoint the co-located-data shadow fit did not pass the repair
  gate; the subsequent Yee-staggered target resolved that fit failure, as
  recorded below.

### 2026-08-09: Yee-staggered target and experimental rational runtime

Achieved:

- derived the characteristic data seen by the actual boundary recurrence from
  the co-located modal admittance:

  \[
  Y_{c,i}=
  \frac{Y_i\exp(j\beta_i\Delta w/2)}{\cos(\omega_i\Delta t/2)}
  -j\Omega_iE_h,
  \qquad
  \Omega_i=\frac{2}{\Delta t}\tan(\omega_i\Delta t/2);
  \]

- added a tested helper for this Yee spatial/time-staggering transformation;
- after the analytic half-cell term is removed, the microstrip target has a
  small physically meaningful imaginary component instead of being forced
  real at every anchor;
- a two-pole fit now passes global positive-real certification without an
  excessive repair. For the 5 ns microstrip it has approximately:
  - poles \(-1.8911\times10^9\pm j1.8181\times10^{10}\) rad/s;
  - fixed direct term \(D=1\);
  - held-out relative error \(8.24\times10^{-4}\);
  - held-out reflection-equivalent error \(4.12\times10^{-4}\);
  - global real-admittance margin 0.9978;
- connected the certified model to an explicitly experimental rational ADE
  path while leaving the default runtime disabled;
- verified the experimental path uses the fitted characteristic operator for
  both the passive load and active incident waveform;
- the experimental 5 ns scattering run and 12 ns late-time stability run both
  pass without non-finite values or exponential growth.

Performance failure:

- despite the accurate passive fit, worst in-band S11 remains approximately
  -16.20 dB, essentially identical to the constant-admittance baseline;
- centre-band S11 remains approximately \(0.0205+j0.1440\), and S21/power
  balance are also essentially unchanged;
- therefore the dominant reactive mismatch is not caused by the scalar
  characteristic-admittance frequency variation represented by these anchors;
- enabling this model by default would add complexity without a material
  accuracy benefit. The experimental switch remains off;
- the next diagnostic must compare the exact one-way raw-Yee terminal ratio
  against the active source and S-parameter decomposition, with particular
  attention to reference-plane de-staggering and generator normalization.

### 2026-08-09: monitor decomposition audit and passive-end isolation

Achieved:

- added a synthetic two-frequency monitor regression covering pure incident
  and pure outgoing waves for both port directions and both adjacent magnetic
  sampling planes;
- verified the monitor convention
  \(E=a+b\) and
  \(H=a\exp(-j\beta\delta)-b\exp(+j\beta\delta)\), including the
  accumulation kernel's direction normalization, recovers the prescribed wave
  to numerical precision;
- the monitor audit includes a full real-field `observe()` path with a poisoned
  unused H plane, so it exercises Yee-plane selection, direction sign, temporal
  DFT staggering, spatial de-staggering, Gram normalization, and final wave
  splitting rather than only the final algebra;
- the focused rational-fit, passivity, ADE, fixed-basis sampling, and monitor
  decomposition set now passes 48 tests;
- inspected the long 300 mm microstrip output independently of the short test.
  At the passive matched end, the reverse-to-forward modal ratio is about
  -19.8 dB at 4.5 GHz and ranges from about -20.2 to -19.4 dB across the
  plotted band. The same ratio is present in the short rational-ADE run;
- this passive-end observation rules out the active generator and source-port
  S-parameter normalization as the primary cause of the residual mismatch.

Findings and current blocker:

- the algebraic reference-plane split and its `magnetic_side=-1` selection are
  internally consistent; changing them would only conceal the physical
  terminal error;
- the fitted proper admittance is globally positive real and accurately
  follows the admittance inferred from the frequency-domain mode anchors, yet
  the passive terminal reflection is essentially unchanged. Therefore the
  frequency-domain anchor E/H ratio is not yet the exact one-way numerical
  characteristic relation seen by the time-domain Yee grid;
- likely missing ingredients include the fully discrete time-domain modal
  impedance (rather than the continuous-frequency FDFD ratio), hybrid-mode
  boundary staggering, or a boundary state associated with longitudinal field
  and surface-charge degrees of freedom. These must be distinguished by a
  direct one-way raw-Yee measurement before changing fit order or passivity
  enforcement;
- the active Norton forcing `2*a` is exact for the current normalized
  order-zero recurrence. With a frequency-dependent characteristic operator,
  the fitted operator must act on `V-2a`, as the experimental ADE does. A
  prescribed boundary-E incident amplitude may additionally need a staggered
  pre-emphasis, but that scale cancels from measured S11 and cannot explain the
  passive-end reflection;
- the rational runtime remains explicitly disabled by default. The next
  implementation gate is a one-way numerical terminal oracle using the same
  raw Yee projection as the boundary, followed by a fitted-model comparison on
  held-out frequencies. No coefficient should be promoted to production until
  that oracle and a coupled energy certificate both pass.

One-way numerical oracle implemented:

- added a 3D microstrip integration oracle with the active ADE source on the
  low face, a conventional PML on the remote high face, and a raw-Yee modal
  projection at an interior plane;
- the oracle DFTs integer-time E and half-time H independently and removes the
  measured half-cell propagation phase before comparing with the FDFD-derived
  fixed-basis admittance;
- across 4.1--4.9 GHz the measured colocated ratio has real part
  0.97984--0.98364 and imaginary part -0.00556-- -0.00217. Its largest absolute
  difference from the FDFD-derived target is 0.02031;
- this is a measurable modelling discrepancy, but by itself it predicts only
  about a one-percent reflection and is too small to explain the roughly
  ten-percent reverse wave at the passive boundary;
- the result narrows the blocker further: a scalar one-way characteristic
  admittance correction is necessary for high accuracy, but it is not
  sufficient for the present hybrid microstrip termination. The next model
  should expose the additional boundary degrees of freedom (or an equivalent
  MIMO discrete DtN map) rather than hiding the residual in a higher-order
  scalar fit.

### 2026-08-09: passivity-certificate audit and validation checkpoint

Audit failure found and fixed:

- the first scalar certificate scaled its pass/fail tolerance by the largest
  real-admittance value among all critical frequencies;
- a very large positive, lightly damped resonance could therefore make the
  tolerance large enough to hide a negative value at DC;
- a concrete three-pole counterexample returned a minimum real admittance of
  approximately -1 while being incorrectly labelled passive;
- the certificate now uses a conservative tolerance tied only to machine
  precision and the requested passivity margin, never to response magnitude
  elsewhere in the spectrum;
- the counterexample is a permanent regression test. An uncertain or
  ill-conditioned minimum is rejected rather than excused by a large remote
  response.

Latest verification:

- 154 focused unit/configuration tests pass after the rational-fit, global
  scalar passivity, ADE, fixed-basis, monitor, parser, and existing matched
  boundary suites are combined;
- 4 coupled microstrip/oracle integration tests pass, including the 12 ns
  no-growth run;
- the fit/passivity/ADE/sample core passes all 47 focused tests after the
  independent audit fixes;
- Python byte-code compilation and whitespace/diff checks pass for the new
  implementation;
- runtime rational use remains disabled. These results validate the
  experimental infrastructure, not a production-quality scalar microstrip
  match.

### 2026-08-09: independent fit/runtime safety audit

Additional failures found and fixed:

- the shadow diagnostic path was not completely isolated. Preprocessing or
  ADE construction could abort a normal constant-admittance run even while
  rational runtime use was disabled. The entire diagnostic preparation is now
  failure-isolated, and no rational ADE is constructed unless runtime use is
  explicitly requested;
- fitted-model arrays were only nominally read-only. They are now copied onto
  immutable byte-backed NumPy views, and the real state-space coefficients use
  the same scheme, so a certified model cannot be changed by re-enabling the
  NumPy write flag;
- the ADE constructor previously accepted any stable pole set without proving
  positive realness. The concrete stable model (D=1,p=-1,r=-2) was accepted
  and its zero-input recurrence grew by orders of magnitude. ADE construction
  now re-certifies the exact stored coefficients and rejects that model;
- the passivity-repair size metric mixed a dimensionless direct term with
  dimensional residues. It now uses the optimizer's response-weighted scaled
  coordinates and is regression-tested under a (10^6) frequency-unit
  rescaling;
- vector-fit synthesis could accept a model whose poles had not converged.
  Nonconverged candidates are now rejected, the default relocation allowance
  was raised from 20 to 50 iterations, and the final pole movement is included
  in the failure message;
- approximately conjugate input poles could pass the fitting tolerance but
  fail the stricter real-rational passivity algebra. Accepted pairs are now
  symmetrized to exact conjugates before fitting;
- least-squares columns are now normalized and rank/condition checked before
  coefficients are accepted;
- the fixed-basis zero-voltage check had an absolute scale hidden inside it.
  It is now relative to the electric/magnetic pairing and accepts a well-
  conditioned joint gauge as small as (10^{-80}) in the regression;
- HDF5 output used the wrong passivity-certificate field name and crashed
  after a completed simulation. This was reproduced, corrected, and covered
  by an end-to-end metadata regression.

Scientific release gate added:

- the scalar runtime may now be enabled only when both fixed-basis electric
  and magnetic residuals are no greater than (10^{-3});
- the tested microstrip has residuals of approximately 2%, so explicitly
  enabling the scalar fitted runtime now fails before time stepping with a
  message requesting a higher-dimensional discrete DtN model;
- this replaces the earlier 5 ns/12 ns experimental scalar-microstrip runtime
  tests. Their stable but unimproved result remains recorded above; keeping
  them as an accepted runtime path would contradict the measured failure of
  the scalar spatial model;
- a fixed-profile closed-guide integration still exercises the order-zero
  rational runtime path, while nonzero-pole dynamics and passivity are tested
  independently at the ADE level.

Intentional behavior change and remaining blockers:

- matched automatic ports now solve 9--13 verification/synthesis anchors.
  This also gives the production order-zero boundary a better local group-
  velocity estimate for its half-cell storage, so runtime-off is not a bitwise
  recreation of the earlier sparse-anchor setup;
- automatic shadow order selection can now reach four poles when enough
  anchors remain after holdout selection. It is still capped by the available
  modal samples and is not yet the adaptive add-an-anchor loop in this plan;
- accuracy is checked at interleaved holdout solves, but there is no adaptive
  between-anchor extremum search yet;
- near-duplicate pole merging, negligible-residue pruning, and an explicit
  pole-zero-cancellation metric remain unimplemented release gates;
- the floating-point real-part polynomial search is a strong scalar
  all-frequency check, not yet an independently verified mathematical
  certificate. A Hamiltonian/KYP cross-check remains mandatory before any
  fitted model is enabled in production.

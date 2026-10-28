"""CPW tracking experiment on real Yee-grid Maxwell eigenfields.

Run with python -m testing.eigenmode_support.identification.cpw_mode_tracking
--output DIR. This checks solver-bank tracking and final source-basis alignment,
not FDTD injection, S-parameters, or an open-boundary reference solution.
"""

import numpy as np
from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver
from testing.eigenmode_support.validate_fdfd_eigenmodes import _configured_solver_constants
from gprMax.eigenmode_symmetry import mirror_u
from gprMax.eigenmode_tracking import _solver_mode_fields


def cpw(f, spacing=0.00025, width=0.012, height=0.008, eps=4.0, count=6, metal_center=0.000125):
    nu, nv = round(width / spacing), round(height / spacing)
    x = (np.arange(nu) + 0.5) * spacing - width / 2
    xn = np.arange(nu + 1) * spacing - width / 2
    y = (np.arange(nv) + 0.5) * spacing - height / 2
    yn = np.arange(nv + 1) * spacing - height / 2

    def metal(x, y):
        return (
            ((abs(x) <= 0.0005 + 1e-12) | ((abs(x) >= 0.001 - 1e-12) & (abs(x) <= 0.003 + 1e-12)))
            & (y >= metal_center - 0.000125 - 1e-12)
            & (y <= metal_center + 0.000125 + 1e-12)
        )

    eu = np.broadcast_to(np.where(yn < 0, eps, 1.0), (nu, nv + 1)).astype(complex).copy()
    ev = np.broadcast_to(np.where(y < 0, eps, 1.0), (nu + 1, nv)).astype(complex).copy()
    ew = np.broadcast_to(np.where(yn < 0, eps, 1.0), (nu + 1, nv + 1)).astype(complex).copy()
    eu[metal(x[:, None], yn[None, :])] = np.inf
    ev[metal(xn[:, None], y[None, :])] = np.inf
    ew[metal(xn[:, None], yn[None, :])] = np.inf
    eu[:, [0, -1]] = np.inf
    ev[[0, -1], :] = np.inf
    ew[:, [0, -1]] = np.inf
    ew[[0, -1], :] = np.inf
    s = FDFD_2D_mode_solver(
        f, spacing, spacing, count - 1, eu, ev, ew, np.ones_like(ev), np.ones_like(eu), np.ones((nu, nv))
    )
    s.retain_tracking_operator = True
    s.calculate_diagnostics = True
    s.solve()
    return s


def stats(s, m):
    e, h = _solver_mode_fields(s, m)
    me, mh = mirror_u(s, e, h)
    vec = np.concatenate([a.ravel() for a in (*e, *(s.eta0 * a for a in h))])
    mv = np.concatenate([a.ravel() for a in (*me, *(s.eta0 * a for a in mh))])
    parity = np.vdot(vec, mv) / np.vdot(vec, vec)
    x = (np.arange(s.Nu) + 0.5) * s.du - s.Nu * s.du / 2
    j = round(s.Nv / 2)
    left = -e[0][(x > -0.001) & (x < -0.0005), j].sum() * s.du
    right = e[0][(x > 0.0005) & (x < 0.001), j].sum() * s.du
    corr = (left.conjugate() * right).real / abs(left * right)
    return [float(parity.real), float(corr), float(abs(left)), float(abs(right))]


def field_parity(electric, magnetic, impedance, action=mirror_u):
    """Full polar-E/axial-H mirror expectation, independent of tracking."""
    reflected_e, reflected_h = action(None, electric, magnetic)
    fields = (*electric, *(impedance * a for a in magnetic))
    reflected = (*reflected_e, *(impedance * a for a in reflected_h))
    return float((sum(np.vdot(a, b) for a, b in zip(fields, reflected)) / sum(np.vdot(a, a) for a in fields)).real)


def track_bank(bank, *, seed=42, mixed=False, parity_action=mirror_u):
    """Scramble candidate order/phase, then use production tracking/alignment."""
    from types import SimpleNamespace
    from gprMax.eigenmode_config import EigenmodeTrackingConfig
    from gprMax.eigenmode_tracking import track_solver_bank, align_groups, _reorder_solver_modes
    from gprMax.sources import EigenmodeSource

    rng = np.random.default_rng(seed)
    axis = np.array([s.frequency for s in bank])
    centre = len(bank) // 2
    owner = EigenmodeSource(None)
    owner.tracking = "auto"
    owner.verification = "none"
    owner.tracking_config = EigenmodeTrackingConfig()
    owner.fallback_frequency = axis[centre]
    owner._tracking_impedance = bank[0].eta0
    owner.port_index = 1
    owner.normal_axis = 2
    owner.normal = "z"
    owner.direction = "+"
    owner.transverse_axes = (0, 1)
    for k, solver in enumerate(bank):
        if mixed:
            # Homogeneous TEM modes coincide within the production tolerance.
            assert np.ptp(solver.eigenvalues[:3].real) < 1e-8
            rotation = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))[0]
            for name in ("Eu", "Ev", "Ew", "Hu", "Hv", "Hw"):
                a = getattr(solver, name)
                a[..., :3] = a[..., :3] @ rotation
            solver._normalize_modes_to_power()
        phase = np.exp(2j * np.pi * rng.random(solver.num_modes))
        for name in ("Eu", "Ev", "Ew", "Hu", "Hv", "Hw"):
            setattr(solver, name, getattr(solver, name) * phase)
        solver.eigenvectors = np.column_stack(
            [
                np.r_[solver.Eu[..., i].ravel(order="F"), solver.Ev[..., i].ravel(order="F")]
                for i in range(solver.num_modes)
            ]
        )
        solver.field_residuals = solver.tracking_field_residuals(range(solver.num_modes))
        if k != centre:
            _reorder_solver_modes(solver, rng.permutation(solver.num_modes))
    mapping = track_solver_bank(owner, axis, bank, 3)
    modes = tuple(range(1, owner._auto_tracking_mode_count + 1))
    fields = [[owner._fields_from_solver_mode(s, m) for m in modes] for s in bank]
    electric = [[p[0] for p in row] for row in fields]
    magnetic = [[p[1] for p in row] for row in fields]
    grid = SimpleNamespace(dl=np.full(3, bank[0].du))
    align_groups(owner, grid, axis, bank, modes, electric, magnetic, np.ones((len(bank), len(modes)), bool))
    parity = np.array(
        [[field_parity(e, h, owner._tracking_impedance, parity_action) for e, h in zip(er, hr)] for er, hr in zip(electric, magnetic)]
    )
    diagnostics = owner.tracking_diagnostics
    return dict(
        frequencies_GHz=(axis / 1e9).tolist(),
        neff=np.array([s.complex_neff[: len(modes)].real for s in bank]).tolist(),
        final_parity=parity.tolist(),
        candidate_indices=mapping.tolist(),
        minimum_overlap=float(diagnostics["overlaps"].min()),
        maximum_residual=float(diagnostics["residuals"].max()),
        numerical_valid=bool(diagnostics["numerical_valid"].all()),
        forward_power_valid=bool(all(s.power_valid[: len(modes)].all() for s in bank)),
        automatic_groups=owner.degenerate,
        detected_symmetry=owner.symmetry_detection["detected"],
        applied_symmetry=[group["applied"] for group in owner.symmetry_detection["groups"]],
        tracked_solver_gap_phase_correlation=([[stats(s, m)[1] for m in range(len(modes))] for s in bank]
                                             if parity_action is mirror_u else None),
    )


def run(output):
    import json
    from pathlib import Path
    from matplotlib import pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with _configured_solver_constants():
        bank = [cpw(f * 1e9) for f in range(2, 11)]
        substrate = track_bank(bank)
        parity = np.asarray(substrate["final_parity"])
        np.testing.assert_allclose(parity, np.tile([1, -1, 1], (len(bank), 1)), atol=1e-8)
        np.testing.assert_allclose(
            substrate["tracked_solver_gap_phase_correlation"], np.tile([1, -1, 1], (len(bank), 1)), atol=1e-8
        )
        assert substrate["numerical_valid"] and substrate["forward_power_valid"]
        # The even CPW branch has the strongest differential slot voltage at
        # the reference anchor; the other even branch belongs to the enclosure.
        assert sum(stats(bank[4], 2)[2:]) > sum(stats(bank[4], 0)[2:])
        homogeneous = track_bank([cpw(f * 1e9, eps=1.0) for f in (6, 7, 8, 9, 10)], mixed=True)
        assert homogeneous["numerical_valid"] and homogeneous["forward_power_valid"]
        assert homogeneous["applied_symmetry"] == ["cs_u"]
        np.testing.assert_allclose(np.abs(homogeneous["final_parity"]), 1.0, atol=1e-8)
        # An odd number of vertical cells puts the 0.25 mm metal faces at
        # +/-0.125 mm while preserving both mirrors on native Yee samples.
        centered = track_bank(
            [cpw(f * 1e9, eps=1.0, height=0.00825, metal_center=0.0) for f in (6, 7, 8, 9, 10)],
            mixed=True,
        )
        assert centered["detected_symmetry"] == "c2v"
        assert centered["applied_symmetry"] == ["c2v"]
        assert centered["numerical_valid"] and centered["forward_power_valid"]
        np.testing.assert_allclose(np.abs(centered["final_parity"]), 1.0, atol=1e-8)
        comparisons = []
        for name, options in [
            ("baseline", {}),
            ("finer_mesh", {"spacing": 0.000125}),
            ("larger_aperture", {"width": 0.016, "height": 0.012}),
        ]:
            solver = cpw(6e9, **options)
            values = [stats(solver, m) for m in range(3)]
            odd = next(m for m, v in enumerate(values) if v[0] < -0.99)
            even = max((m for m, v in enumerate(values) if v[0] > 0.99), key=lambda m: sum(values[m][2:]))
            comparisons.append(
                dict(case=name, even_neff=float(solver.real_neff[even]), odd_neff=float(solver.real_neff[odd]))
            )
    report = dict(
        substrate=substrate,
        homogeneous_degenerate=homogeneous,
        centered_homogeneous_degenerate=centered,
        convergence=comparisons,
        scope="Finite PEC enclosure; solver-bank tracking and source-basis alignment only. No FDTD or S-parameters.",
        geometry=dict(
            strip_mm=1.0,
            slot_mm=0.5,
            ground_width_mm=2.0,
            metal_thickness_mm=0.25,
            default_metal_center_mm=0.125,
            centered_control=dict(metal_center_mm=0.0, aperture_mm=[12, 8.25]),
            aperture_mm=[12, 8],
            spacing_mm=0.25,
            substrate_epsilon=4.0,
            substrate="Lower half of aperture, no backside ground. Grounds isolated from outer box.",
        ),
        labels=dict(even=3, odd=2, enclosure_branch=1),
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    axes = axes.ravel()
    frequency = substrate["frequencies_GHz"]
    neff = np.asarray(substrate["neff"])
    for m, label in [(2, "Even CPW"), (1, "Odd slotline"), (0, "Enclosure branch")]:
        axes[0].plot(frequency, neff[:, m], ".-", label=label)
        axes[1].plot(frequency, parity[:, m], ".-", label=label)
    axes[0].set(xlabel="Frequency (GHz)", ylabel="Effective index", title="Substrate CPW: tracking passes")
    axes[0].legend(fontsize=8)
    axes[1].set(
        xlabel="Frequency (GHz)", ylabel="Full-vector mirror parity", ylim=(-1.1, 1.1), title="Independent parity check"
    )
    for m in range(3):
        axes[2].plot(
            homogeneous["frequencies_GHz"], np.asarray(homogeneous["final_parity"])[:, m], ".-", label=f"Channel {m+1}"
        )
    axes[2].set(
        xlabel="Frequency (GHz)",
        ylabel="Full-vector mirror parity",
        ylim=(-1.1, 1.1),
        title="Offset degenerate TEM: Cs restores parity",
    )
    axes[2].legend(fontsize=8)
    for m in range(3):
        axes[3].plot(
            centered["frequencies_GHz"], np.asarray(centered["final_parity"])[:, m], ".-", label=f"Channel {m+1}"
        )
    axes[3].set(
        xlabel="Frequency (GHz)",
        ylabel="Full-vector mirror parity",
        ylim=(-1.1, 1.1),
        title="Centered degenerate TEM: C2v restores parity",
    )
    axes[3].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(output / "tracking.png", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            dict(
                substrate_minimum_overlap=substrate["minimum_overlap"],
                substrate_maximum_residual=substrate["maximum_residual"],
                homogeneous_final_parity=homogeneous["final_parity"],
                centered_final_parity=centered["final_parity"],
                convergence=comparisons,
            ),
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/cpw_mode_tracking")
    run(parser.parse_args().output)

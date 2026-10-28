"""Every subgroup of the transverse square Yee-grid point group."""
import numpy as np
import pytest

from gprMax.eigenmode_symmetry import (
    NATIVE_SYMMETRIES, detect_symmetry, field_actions, irreducible_blocks,
    abelian_irrep_names, _validate_material_actions,
)
from tests.test_eigenmode_auto_symmetry import sampled_port, aligned_bank
from testing.eigenmode_support.subspaces.expanded_mode_tracking import fd_solver
from testing.eigenmode_support.validate_fdfd_eigenmodes import _configured_solver_constants


def invariant_port(group):
    port = sampled_port()
    rng = np.random.default_rng(67)
    seed = rng.normal(size=(9, 9)) + 1j*rng.normal(size=(9, 9))
    orbit = [seed]
    for item in orbit:
        for action in field_actions(group).values():
            mapped = action(port, (item, item, item), (item, item, item))[0][2]
            if not any(np.array_equal(mapped, old) for old in orbit):
                orbit.append(mapped)
        assert len(orbit) <= 8
    port.eps_r_ww = 4 + sum(orbit)/len(orbit)
    return port


@pytest.mark.parametrize('group', NATIVE_SYMMETRIES)
def test_operation_pipeline_identifies_exact_subgroup(group):
    result = detect_symmetry([invariant_port(group)])
    assert result['detected'] == group
    assert set(field_actions(group)).issubset(result['operations'])


def test_pipeline_tests_each_primitive_once(monkeypatch):
    import gprMax.eigenmode_symmetry as symmetry
    calls = []
    def record(solver, actions):
        calls.extend(actions)
        return _validate_material_actions(solver, actions)
    monkeypatch.setattr(symmetry, '_validate_material_actions', record)
    assert detect_symmetry([sampled_port()])['detected'] == 'c4v'
    assert len(calls) == len(set(calls)) == 6


def test_bank_intersects_operations_and_rectangular_mesh_rejects_axis_exchange():
    assert detect_symmetry([invariant_port('c4'), invariant_port('c2v')])['detected'] == 'c2'
    port = sampled_port(nv=6)
    assert detect_symmetry([port])['candidates'] == ['c2v', 'c2', 'cs_u', 'cs_v']
    assert detect_symmetry([])['detected'] == 'none'


def test_complex_material_loss_is_not_erased():
    port = invariant_port('cs_u')
    port.eps_r_ww[1, 2] += .01j
    assert detect_symmetry([port])['detected'] == 'none'


@pytest.mark.parametrize('group', ['c4', 'c2', 'c2v_diagonal', 'cs_u', 'cs_v', 'cs_diagonal', 'cs_antidiagonal'])
def test_native_group_decomposes_real_maxwell_doublet(group):
    with _configured_solver_constants():
        bank = [fd_solver(f*1e9, (12,12), .02/12, (2,2,2), count=6) for f in (19,20,21)]
        owner, _, _, _ = aligned_bank(bank, (1,2), symmetry=group)
    record = owner.symmetry_diagnostics[0]
    assert all(len(block) == 1 for block in record['blocks'])
    assert len(record['irreps']) == 2
    assert all(row['closure'] < 1e-8 and row['field_residual'] < 1e-7 for row in record['anchors'].values())
    if group == 'c4':
        assert set(record['irreps']) == {'E+', 'E-'}
    elif group.startswith('cs_'):
        assert set(record['irreps']) == {"A'", "A''"}


def test_horizontal_cpw_detects_and_applies_cs_v():
    from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver
    from gprMax.eigenmode_symmetry import mirror_v
    from testing.eigenmode_support.identification.cpw_mode_tracking import cpw, track_bank
    with _configured_solver_constants():
        bank = []
        for frequency in (6e9, 8e9, 10e9):
            original = cpw(frequency, eps=1.)
            materials = [getattr(original, f'{prefix}_{a}{a}').T
                         for prefix in ('eps_r', 'mu_r') for a in 'vuw']
            masks = {f'{kind}_{target}_mask': getattr(original, f'{kind}_{source}_mask').T
                     for kind in ('pec', 'pmc') for target, source in zip('uvw', 'vuw')}
            solver = FDFD_2D_mode_solver(frequency, original.dv, original.du, 5, *materials, **masks)
            solver.retain_tracking_operator = True
            solver.calculate_diagnostics = True
            solver.solve()
            bank.append(solver)
        result = track_bank(bank, mixed=True, parity_action=mirror_v)
    assert result['detected_symmetry'] == 'cs_v'
    assert result['applied_symmetry'] == ['cs_v']
    np.testing.assert_allclose(abs(np.asarray(result['final_parity'])), 1., atol=1e-8)


def test_new_group_labels_and_operation_diagnostics_roundtrip_hdf5(tmp_path):
    import h5py
    import json
    from gprMax.eigenmode_tracking import write_diagnostics
    with _configured_solver_constants():
        solver = fd_solver(20e9, (12,12), .02/12, (2,2,2), count=6)
        owner, _, _, _ = aligned_bank([solver], (1,2), symmetry='cs_u')
        owner.symmetry_detection = detect_symmetry([invariant_port('cs_u')])
        with h5py.File(tmp_path/'symmetry.h5', 'w') as output:
            write_diagnostics(output, owner)
        with h5py.File(tmp_path/'symmetry.h5') as output:
            details = json.loads(output['symmetry_detection'].attrs['Details'])
            assert details['operations'] == ['mirror_u']
            assert output['symmetry_detection'].attrs['Detected'] == 'cs_u'
            assert 'symmetry_subspaces' in output

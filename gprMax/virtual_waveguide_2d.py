"""Reference CPU aperture coupling on the live plane of a reduced Yee grid.

The invariant storage layers are never treated as physical PEC side walls.
All operations are vectorized along the one physical transverse coordinate.
"""

import numpy as np


def _plane(guide, array):
    invariant = guide.reduced.invariant_axis
    selection = [slice(None)] * 3
    selection[invariant] = guide.reduced.live_index
    view = array[tuple(selection)]
    remaining = [axis for axis in range(3) if axis != invariant]
    return view if remaining[0] == guide.normal_axis else view.T


def _geometry(guide):
    transverse = next(
        axis for axis in guide.transverse_axes if axis != guide.reduced.invariant_axis
    )
    local = guide.transverse_axes.index(transverse)
    lower = (guide.u0, guide.v0)[local]
    count = (guide.nu, guide.nv)[local]
    aperture = 0 if guide.direction_sign < 0 else guide.spec.length_cells
    return transverse, lower, count, aperture


def _rear(guide, selection):
    # Detach only the neighbouring Yee layer. A full rear prism would
    # suppress exterior radiation around a finite feed. The boundary must
    # be the same whether or not a field monitor has been requested.
    lower, upper = guide.plane_index - 1, guide.plane_index + 2
    return slice(max(selection.start, lower), min(selection.stop, upper))


def couple_magnetic(guide):
    transverse, lower, count, aperture = _geometry(guide)
    normal, plane = guide.normal_axis, guide.plane_index
    for name in guide.reduced.active_magnetic:
        axis = "xyz".index(name[1].lower())
        main = _plane(guide, getattr(guide.main_grid, name))
        aux = _plane(guide, getattr(guide.aux_grid, name))
        stop = count + int(axis == transverse)
        if axis == normal:
            aux[aperture, :stop] = main[plane, lower : lower + stop]
        normal_stop = main.shape[0] - int(axis != normal)
        rear = (
            slice(0, plane)
            if guide.direction_sign > 0
            else slice(plane + int(axis == normal), normal_stop)
        )
        main[_rear(guide, rear), lower : lower + stop] = 0


def couple_electric(guide):
    transverse, lower, count, aperture = _geometry(guide)
    normal, plane = guide.normal_axis, guide.plane_index
    inside = 0 if guide.direction_sign < 0 else aperture - 1
    main_grid, aux_grid = guide.main_grid, guide.aux_grid
    for name in guide.reduced.active_electric:
        axis = "xyz".index(name[1].lower())
        main = _plane(guide, getattr(main_grid, name))
        aux = _plane(guide, getattr(aux_grid, name))
        stop = count + int(axis != transverse)
        if axis != normal:
            # Tangential E owns the aperture node. Its normal H difference
            # contains one auxiliary and one main-domain cell sample.
            magnetic = 3 - axis - normal
            main_h = _plane(guide, getattr(main_grid, "H" + "xyz"[magnetic]))
            aux_h = _plane(guide, getattr(aux_grid, "H" + "xyz"[magnetic]))
            selected = slice(1, count) if axis == guide.reduced.invariant_axis else slice(0, count)
            source_selected = slice(lower + selected.start, lower + selected.stop)
            if guide.direction_sign < 0:
                difference = aux_h[0, selected] - main_h[plane - 1, source_selected]
            else:
                difference = main_h[plane, source_selected] - aux_h[inside, selected]
            ids = _plane(guide, aux_grid.ID[axis])[aperture, selected]
            coeffs = aux_grid.updatecoeffsE[ids]
            sign = 1 if (normal - axis) % 3 == 1 else -1
            value = (
                coeffs[:, 0] * aux[aperture, selected] + sign * coeffs[:, normal + 1] * difference
            )
            if axis == guide.reduced.invariant_axis:
                h_normal = _plane(guide, getattr(aux_grid, "H" + "xyz"[normal]))
                difference = h_normal[aperture, 1:count] - h_normal[aperture, : count - 1]
                value -= sign * coeffs[:, transverse + 1] * difference
            aux[aperture, selected] = value
            main[plane, lower : lower + stop] = aux[aperture, :stop]
        else:
            main_inside = plane if guide.direction_sign < 0 else plane - 1
            main[main_inside, lower : lower + stop] = aux[inside, :stop]
        normal_stop = main.shape[0] - int(axis == normal)
        rear = (
            slice(0, plane - int(axis == normal))
            if guide.direction_sign > 0
            else slice(plane + 1, normal_stop)
        )
        main[_rear(guide, rear), lower : lower + stop] = 0

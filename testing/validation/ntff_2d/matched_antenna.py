"""Finite parallel-plate antennas fed by an enclosed matched modal aperture."""

import numpy as np
import gprMax


def antenna_scene(family="TM", axis=2, spacing=0.0005, direction="+", steps=None):
    normal, transverse = (axis + 1) % 3, (axis + 2) % 3

    def point(n, t, a=np.inf):
        p = np.zeros(3)
        p[normal], p[transverse], p[axis] = n, t, a
        return tuple(p)

    scene = gprMax.Scene()
    pml_cells = round(0.006 / spacing)
    pml = [pml_cells] * 6
    pml[axis] = pml[axis + 3] = 0
    for obj in (
        gprMax.DomainMode(mode=family),
        gprMax.Discretisation(p1=(spacing,) * 3),
        gprMax.Domain(p1=point(0.08, 0.08)),
        gprMax.PMLThickness(thickness=tuple(pml)),
        gprMax.TimeWindow(time=2e-9) if steps is None else gprMax.TimeWindow(iterations=steps),
        gprMax.OMPThreads(1),
        gprMax.TimeStepStabilityFactor(f=0.99),
    ):
        scene.add(obj)

    def x(n):
        return n if direction == "+" else 0.08 - n

    def box(n1, n2, t1, t2):
        scene.add(
            gprMax.Box(
                p1=point(min(x(n1), x(n2)), t1, 0), p2=point(max(x(n1), x(n2)), t2), material_id="pec", averaging="n"
            )
        )

    for lower, upper in ((0.035, 0.036), (0.044, 0.045)):
        box(0.024, 0.046, lower, upper)
    box(0.024, 0.025, 0.035, 0.045)
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=22e9, id="pulse"))
    scene.add(gprMax.EigenmodeBand(id="band", fmin=22e9, fmax=22e9, points=1))
    scene.add(
        gprMax.EigenmodePort(
            port=1,
            p1=point(x(0.028), 0.036, 0),
            p2=point(x(0.028), 0.044),
            direction=direction,
            modes=(1,),
            anchors=(22e9,),
            plot_fields=False,
        )
    )
    scene.add(
        gprMax.VirtualWaveguide(
            port=1,
            length_cells=round(0.024 / spacing),
            pml_cells=pml_cells,
            source_clearance_cells=round(0.003 / spacing),
        )
    )
    scene.add(gprMax.EigenmodeExcitation(port=1, mode=1, waveform="pulse", plot_waveform=False))
    contours = []
    margin = max(0.012, (pml_cells + 2) * spacing)
    for method in ("ksir", "equivalent_current"):
        contour = gprMax.NTFFContour(
            p1=point(margin, margin, 0),
            p2=point(0.08 - margin, 0.08 - margin),
            id=method,
            frequencies=(22e9,),
            angles=tuple(range(0, 360, 2)),
            formulation=method,
        )
        scene.add(contour)
        contours.append(contour)
    return scene, contours

"""Editable reference projects in SI units."""

from .project import new_project, node, default_port, default_ntff


def horn():
    project = new_project()
    project["build"]["enabled"] = False
    project["name"] = "Waveguide-fed stepped horn"

    def box(a, b):
        a, b = [v * 1e-3 for v in a], [v * 1e-3 for v in b]
        project["features"].append(node("box", "Horn wall", origin=a, size=[y - x for x, y in zip(a, b)]))

    def plate(a, b):
        project["features"].append(
            node("sheet", "Expansion closure", p1=[v * 1e-3 for v in a], p2=[v * 1e-3 for v in b])
        )

    def section(x0, x1, y0, y1, z0, z1):
        box((x0, y0 - 1, z0 - 1), (x1, y0, z1 + 1))
        box((x0, y1, z0 - 1), (x1, y1 + 1, z1 + 1))
        box((x0, y0, z0 - 1), (x1, y1, z0))
        box((x0, y0, z1), (x1, y1, z1 + 1))

    section(12, 35, 33, 57, 29, 41)
    for i in range(9):
        x, y0, y1, z0, z1 = 35 + 5 * i, 33 - 2 * i, 57 + 2 * i, 29 - 2 * i, 41 + 2 * i
        section(x, x + 5, y0, y1, z0, z1)
        if i < 8:
            plate((x + 5, y0 - 2, z0 - 2), (x + 5, y0, z1 + 2))
            plate((x + 5, y1, z0 - 2), (x + 5, y1 + 2, z1 + 2))
            plate((x + 5, y0, z0 - 2), (x + 5, y1, z0))
            plate((x + 5, y0, z1), (x + 5, y1, z1 + 2))
    project["ports"] = [default_port()]
    project["ntff"] = [default_ntff()]
    return project


def patch():
    project = new_project()
    project["build"]["enabled"] = False
    project["name"] = "Probe-fed patch — illustrative materials"
    project["simulation"].update(
        domain=[0.060, 0.060, 0.040], spacing=[0.0005] * 3, time=8e-9, pml=6, fmin=3e9, fmax=8e9, points=101
    )
    features = project["features"]
    features.append(node("box", "Substrate", "substrate", origin=[0.012, 0.012, 0.019], size=[0.036, 0.036, 0.002]))
    features.append(node("sheet", "Patch", p1=[0.020, 0.021, 0.021], p2=[0.040, 0.039, 0.021]))
    ground = node("box", "Ground blank", origin=[0.012, 0.012, 0.0185], size=[0.036, 0.036, 0.0005])
    hole = node("cylinder", "Ground aperture", origin=[0.030, 0.030, 0.018], axis=[0, 0, 1], radius=0.002, height=0.002)
    features.extend([ground, hole, node("subtract", "Ground", target=ground["id"], tool=hole["id"])])
    # Continue both sides of the modal reference plane for uniform Yee samples.
    outer = node(
        "cylinder", "Coax outer solid", origin=[0.030, 0.030, 0.007], axis=[0, 0, 1], radius=0.003, height=0.012
    )
    bore = node("cylinder", "Coax bore", origin=[0.030, 0.030, 0.007], axis=[0, 0, 1], radius=0.002, height=0.012)
    tube = node("subtract", "Coax outer conductor", target=outer["id"], tool=bore["id"])
    features.extend([outer, bore, tube])
    features.append(
        node(
            "cylinder",
            "Coax dielectric",
            "substrate",
            origin=[0.030, 0.030, 0.007],
            axis=[0, 0, 1],
            radius=0.002,
            height=0.014,
        )
    )
    features.append(
        node("cylinder", "Probe", origin=[0.030, 0.030, 0.007], axis=[0, 0, 1], radius=0.00075, height=0.014)
    )
    port = default_port()
    port.update(p1=[0.027, 0.027, 0.008], p2=[0.033, 0.033, 0.008])
    project["ports"] = [port]
    monitor = default_ntff()
    monitor.update(p1=[0.005, 0.005, 0.005], p2=[0.055, 0.055, 0.035], frequencies=[3e9, 4e9, 5e9, 6e9, 7e9, 8e9])
    project["ntff"] = [monitor]
    return project

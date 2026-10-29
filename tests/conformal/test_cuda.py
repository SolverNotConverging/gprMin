import numpy as np
import pytest

import gprMax
from gprMax.conformal.cuda import CUDAConformalStepper
from gprMax.impedance_surfaces import SurfaceImpedanceModel
from tests.conformal.test_topology import compile_scene, material_grid


@pytest.mark.gpu
@pytest.mark.parametrize("precision", [np.float32, np.float64])
@pytest.mark.parametrize("model_order", [None, 0, 1, 2])
def test_device_fields_and_ade_match_cython(gpu_device, precision, model_order):
    import pycuda.driver as cuda

    grid = material_grid()
    grid.materials[0].er = 4.4
    grid.materials[0].se = 0.005
    grid.materials[0].sm = 0.01
    boundary = "pec" if model_order is None else "copper"
    topology = compile_scene(grid, gprMax.Triangle(
        p1=[-10, 14.4, -10], p2=[14.4, -10, -10], p3=[2.2, 2.2, 20],
        thickness=0, material_id=boundary))
    packing = topology.packing
    dt = 0.6 * packing.operator.cfl_limit
    if model_order == 2:
        model = SurfaceImpedanceModel("copper", A=[[-2, 0.5], [0.5, -3]],
                                      B=[1, 0.4], C=[0.1, 0.04], D=0.2)
    elif model_order == 1:
        model = SurfaceImpedanceModel("copper", A=[[-2]], B=[1], C=[0.1], D=0.2)
    else:
        model = SurfaceImpedanceModel("copper", D=0.2)
    models = {"copper": model}
    cpu_ports, device_ports = [topology.impedance_ports(models, dt) for _ in range(2)]
    rng = np.random.default_rng(6229)
    if cpu_ports is not None:
        for cpu_state, device_state in zip(cpu_ports.state, device_ports.state):
            cpu_state[:] = device_state[:] = rng.normal(size=len(cpu_state))*0.03
    e = rng.normal(size=len(packing.electric_indices)).astype(precision)
    h = rng.normal(size=len(packing.magnetic_indices)).astype(precision)
    ctx = cuda.Device(gpu_device).make_context()
    try:
        stepper = CUDAConformalStepper(packing, e, h, dt, magnetic_impedance=device_ports)
        for _ in range(100):
            packing.step(e, h, dt, backend="cython", magnetic_impedance=cpu_ports)
            stepper.step()
        actual_e, actual_h = stepper.fields()
        tolerance = 2e-5 if precision == np.float32 else 3e-13
        np.testing.assert_allclose(actual_e, e, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(actual_h, h, rtol=tolerance, atol=tolerance)
        states, current, voltage = stepper.boundary_state()
        if cpu_ports is not None:
            for actual, expected in zip(states, cpu_ports.state):
                np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
            np.testing.assert_allclose(current, cpu_ports.last_current, rtol=tolerance, atol=tolerance)
            np.testing.assert_allclose(voltage, cpu_ports.last_voltage, rtol=tolerance, atol=tolerance)
        else:
            assert states == ()
            assert current.size == voltage.size == 0
        del stepper
    finally:
        ctx.pop()
        ctx.detach()


@pytest.mark.gpu
@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_enlarged_fields_and_separate_wall_states_match_cython(gpu_device, precision):
    import pycuda.driver as cuda
    from gprMax.conformal.enlargement import plan_enlargement
    from gprMax.conformal.topology import MagneticBoundaryPort
    from tests.conformal.test_enlargement import topology

    cut = topology([[1, -1, 0], [0, 1, -1]], [0.001, 0.999])
    cut.packing.magnetic_loss[:] = [0.0002, 0.1998]
    cut.boundary_ports = (MagneticBoundaryPort(0, 0.2, "wall", 0),
                          MagneticBoundaryPort(1, 0.3, "wall", 0))
    plan = plan_enlargement(cut, 0.5)
    packing = plan.packing(cut)
    models = {"wall": SurfaceImpedanceModel("wall", A=[[-2]], B=[1], C=[0.1], D=0.2)}
    cpu_ports, device_ports = [plan.impedance_ports(cut, models, 0.5) for _ in range(2)]
    for ports in (cpu_ports, device_ports):
        ports.state[0][:] = 0.1
        ports.state[1][:] = -0.2
    e = np.array([1., 0., -1.], dtype=precision)
    h = plan.restrict_magnetic(np.array([2., 3.])).astype(precision)
    ctx = cuda.Device(gpu_device).make_context()
    try:
        stepper = CUDAConformalStepper(packing, e, h, 0.5, magnetic_impedance=device_ports)
        for _ in range(100):
            packing.step(e, h, 0.5, magnetic_impedance=cpu_ports)
            stepper.step()
        tolerance = 2e-6 if precision == np.float32 else 1e-14
        for actual, expected in zip(stepper.fields(), (e, h)):
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
        states, _, _ = stepper.boundary_state()
        for actual, expected in zip(states, cpu_ports.state):
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
        del stepper
    finally:
        ctx.pop()
        ctx.detach()

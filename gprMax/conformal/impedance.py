"""Independent opaque SIBC ports for the sparse reference discretisation.

The geometry compiler supplies tangential trace rows, quadrature weights,
and a distinct port for each side/polarisation. No port connects through an
opaque sheet. The existing passive impedance realization is reused without
resolving metal skin depth.
"""

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import splu


class ImpedancePorts:
    def __init__(self, operator, trace, weights, models, dt):
        operator.validate_timestep(dt)
        self.operator = operator
        self.dt = float(dt)
        self.trace = csr_matrix(trace, dtype=np.float64, copy=True)
        self.weights = np.array(weights, dtype=np.float64, copy=True)
        self.models = tuple(models)
        ports, electric_count = self.trace.shape
        if electric_count != len(operator.electric_mass) or not ports:
            raise ValueError("trace must map electric DOFs to at least one surface port")
        if len(self.models) != ports or self.weights.shape != (ports,):
            raise ValueError("one model and quadrature weight is required per port")
        if (not np.all(np.isfinite(self.trace.data))
                or not np.all(np.isfinite(self.weights)) or np.any(self.weights <= 0)):
            raise ValueError("trace must be finite and surface weights strictly positive")
        if any(model.allow_active for model in self.models):
            raise ValueError("conformal ports require passive impedance models")
        self.discrete = tuple(model.discretise(dt) for model in self.models)
        self.state = [np.zeros(model.order) for model in self.models]
        self.inv_z = np.array([1 / item.Z0 for item in self.discrete])
        self.conductance = self.trace.T @ diags(self.weights * self.inv_z) @ self.trace
        self.solve = splu((diags(operator.electric_mass)
                          + 0.5 * dt * self.conductance).tocsc())
        self.last_current = np.zeros(ports)
        self.last_midpoint_voltage = np.zeros(ports)

    def reset(self):
        for state in self.state:
            state.fill(0)
        self.last_current.fill(0)
        self.last_midpoint_voltage.fill(0)

    def correct(self, old_electric, free_electric):
        """Replace a free E update with the implicit midpoint boundary solve."""
        history = np.array([item.L @ state for item, state in zip(self.discrete, self.state)])
        rhs = (self.operator.electric_mass * free_electric
               - 0.5 * self.dt * (self.conductance @ old_electric)
               + self.dt * (self.trace.T @ (self.weights * self.inv_z * history)))
        updated = self.solve.solve(rhs)
        voltage = self.trace @ (0.5 * (updated + old_electric))
        current = (voltage - history) * self.inv_z
        for index, item in enumerate(self.discrete):
            self.state[index][:] = item.F @ self.state[index] + item.G * current[index]
        self.last_current[:] = current
        self.last_midpoint_voltage[:] = voltage
        free_electric[:] = updated


class MagneticImpedancePorts:
    """Implicit SIBC electric circulation driven by tangential magnetic traces.

    This is the impedance form of the boundary operator. Each sheet side
    owns its own trace and ADE state; positive quadrature weights give a
    nonnegative resistive contribution to the discrete Poynting balance.
    """

    def __init__(self, operator, trace, weights, models, dt, *, bulk_loss=None):
        operator.validate_timestep(dt)
        self.operator, self.dt = operator, float(dt)
        self.trace = csr_matrix(trace, dtype=np.float64, copy=True)
        self.weights = np.asarray(weights, dtype=np.float64).copy()
        self.models = tuple(models)
        count, magnetic_count = self.trace.shape
        if magnetic_count != len(operator.magnetic_mass) or count == 0:
            raise ValueError("trace must map magnetic DOFs to surface ports")
        if self.weights.shape != (count,) or len(self.models) != count:
            raise ValueError("one positive weight and model is required per magnetic port")
        if (not np.all(np.isfinite(self.weights)) or np.any(self.weights <= 0)
                or not np.all(np.isfinite(self.trace.data))):
            raise ValueError("magnetic traces and positive quadrature weights must be finite")
        if any(model.allow_active or model.is_pmc for model in self.models):
            raise ValueError("magnetic impedance ports require finite passive surface impedances")
        self.discrete = tuple(model.discretise(dt) for model in self.models)
        self.state = [np.zeros(model.order) for model in self.models]
        self.z = np.array([item.Z0 for item in self.discrete])
        self.resistance = self.trace.T @ diags(self.weights*self.z) @ self.trace
        self.bulk_loss = (np.zeros(magnetic_count) if bulk_loss is None
                          else np.array(bulk_loss, dtype=np.float64, copy=True))
        if (self.bulk_loss.shape != (magnetic_count,)
                or not np.all(np.isfinite(self.bulk_loss)) or np.any(self.bulk_loss < 0)):
            raise ValueError("bulk magnetic loss must be finite and nonnegative for every DOF")
        self.total_resistance = self.resistance + diags(self.bulk_loss)
        self.solve = splu((diags(operator.magnetic_mass) + 0.5*dt*self.total_resistance).tocsc())
        self.last_current = np.zeros(count)
        self.last_voltage = np.zeros(count)

    def reset(self):
        for state in self.state:
            state.fill(0)
        self.last_current.fill(0)
        self.last_voltage.fill(0)

    def correct(self, old_magnetic, free_magnetic):
        history = np.array([item.L @ state for item, state in zip(self.discrete, self.state)])
        rhs = (self.operator.magnetic_mass*free_magnetic
               - 0.5*self.dt*(self.total_resistance @ old_magnetic)
               - self.dt*(self.trace.T @ (self.weights*history)))
        updated = self.solve.solve(rhs)
        current = self.trace @ (0.5*(updated+old_magnetic))
        self.last_current[:] = current
        self.last_voltage[:] = self.z*current + history
        for index, item in enumerate(self.discrete):
            self.state[index][:] = item.F @ self.state[index] + item.G*current[index]
        free_magnetic[:] = updated

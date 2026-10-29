"""Device-resident sparse conformal fields and local magnetic SIBC histories.

The caller owns the CUDA context. This operator backend does not yet connect
the production Scene's source, PML, output, or eigenmode-port updates.
"""

import numpy as np


_SOURCE = r"""
typedef FIELD field_t;
extern "C" __global__ void magnetic(
    long long n, const long long *offset, const long long *index, const double *c,
    const double *mass, const double *loss, const field_t *e, field_t *h, field_t *old,
    const long long *poffset, const long long *pindex, const double *weight,
    const double *q, const double *z, const long long *soffset, const double *l,
    const double *state, double dt) {
    long long r = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (r >= n) return;
    double circulation = 0, history = 0, resistance = loss[r];
    for (long long k=offset[r]; k<offset[r+1]; ++k) circulation += c[k]*e[index[k]];
    for (long long k=poffset[r]; k<poffset[r+1]; ++k) {
        long long p = pindex[k];
        double memory = 0;
        for (long long s=soffset[p]; s<soffset[p+1]; ++s) memory += l[s]*state[s];
        resistance += weight[p]*z[p]*q[p]*q[p];
        history += weight[p]*q[p]*memory;
    }
    old[r] = h[r];
    h[r] = ((mass[r]-0.5*dt*resistance)*h[r]-dt*(circulation+history))
           / (mass[r]+0.5*dt*resistance);
}
extern "C" __global__ void electric(
    long long n, const long long *offset, const long long *index, const double *c,
    const double *mass, const double *loss, const field_t *h, field_t *e, double dt) {
    long long r = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (r >= n) return;
    double circulation = 0;
    for (long long k=offset[r]; k<offset[r+1]; ++k) circulation += c[k]*h[index[k]];
    e[r] = ((mass[r]-0.5*dt*loss[r])*e[r]+dt*circulation)/(mass[r]+0.5*dt*loss[r]);
}
extern "C" __global__ void boundary(
    long long n, const long long *row, const double *q, const double *z,
    const long long *soffset, const long long *foffset,
    const double *f, const double *g, const double *l,
    const double *state, double *next, const field_t *h, const field_t *old,
    double *current, double *voltage) {
    long long p = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= n) return;
    double j = q[p]*0.5*((double)h[row[p]]+(double)old[row[p]]), memory = 0;
    long long begin = soffset[p], end = soffset[p+1], order = end-begin;
    for (long long s=begin; s<end; ++s) memory += l[s]*state[s];
    current[p] = j;
    voltage[p] = z[p]*j+memory;
    for (long long s=0; s<order; ++s) {
        double value = g[begin+s]*j;
        for (long long t=0; t<order; ++t) value += f[foffset[p]+s*order+t]*state[begin+t];
        next[begin+s] = value;
    }
}
"""


class CUDAConformalStepper:
    """Advance the validated paired CPU operator without per-step transfers.

    Compiled cut-face SIBC traces have one magnetic DOF per port. General
    coupled trace matrices need a device sparse implicit solver and are
    rejected here, rather than replacing their solve by a diagonal one.
    """

    def __init__(self, packing, electric, magnetic, dt, *, magnetic_impedance=None):
        from pycuda import gpuarray
        from pycuda.compiler import SourceModule

        operator = packing.operator
        operator.validate_timestep(dt)
        operator._validate_fields(electric, magnetic)
        self.dt = np.float64(dt)
        self.ne, self.nh = len(electric), len(magnetic)
        self.port_count = 0
        self._state_offsets = np.zeros(1, dtype=np.int64)
        self._ports = magnetic_impedance
        rows, q, weights, z, state, f, g, l = [], [], [], [], [], [], [], []
        foffset = [0]
        if magnetic_impedance is not None:
            ports = magnetic_impedance
            if ports.operator is not operator or ports.dt != dt:
                raise ValueError("magnetic ports must match this operator and timestep")
            if not np.array_equal(ports.bulk_loss, packing.magnetic_loss):
                raise ValueError("magnetic ports must include this packing's bulk magnetic loss")
            trace = ports.trace.copy()
            trace.sum_duplicates()
            trace.eliminate_zeros()
            if np.any(np.diff(trace.indptr) != 1):
                raise ValueError("CUDA local SIBC requires one magnetic DOF per trace")
            rows, q = trace.indices, trace.data
            weights, z = ports.weights, ports.z
            self.port_count = len(weights)
            self._state_offsets = np.concatenate(([0], np.cumsum([len(s) for s in ports.state])))
            state = np.concatenate(ports.state)
            for item in ports.discrete:
                f.extend(item.F.ravel())
                g.extend(item.G.ravel())
                l.extend(item.L.ravel())
                foffset.append(len(f))
        # Each magnetic row owns its ports, and each port owns its ADE state.
        order = np.argsort(rows, kind="stable")
        counts = np.bincount(np.asarray(rows, dtype=np.int64), minlength=self.nh)
        poffset = np.concatenate(([0], np.cumsum(counts)))

        def upload(values, dtype=np.float64):
            # Allocate a sentinel for empty port/ADE arrays; kernels never
            # address it because the corresponding offsets are equal.
            array = np.ascontiguousarray(values, dtype=dtype)
            return gpuarray.to_gpu(array if array.size else np.zeros(1, dtype=dtype))

        self.electric, self.magnetic = gpuarray.to_gpu(electric), gpuarray.to_gpu(magnetic)
        self.old_magnetic = gpuarray.empty_like(self.magnetic)
        self.curl = tuple(upload(v, np.int64 if i < 2 else np.float64)
                          for i, v in enumerate(operator._packed[0]))
        self.transpose = tuple(upload(v, np.int64 if i < 2 else np.float64)
                               for i, v in enumerate(operator._packed[1]))
        self.me, self.mh = upload(operator.electric_mass), upload(operator.magnetic_mass)
        self.se, self.sm = upload(packing.electric_loss), upload(packing.magnetic_loss)
        self.poffset, self.pindex = upload(poffset, np.int64), upload(order, np.int64)
        self.rows, self.q = upload(rows, np.int64), upload(q)
        self.weights, self.z = upload(weights), upload(z)
        self.soffset = upload(self._state_offsets, np.int64)
        self.foffset = upload(foffset, np.int64)
        self.f, self.g, self.l = upload(f), upload(g), upload(l)
        self.state = upload(state)
        self.next_state = gpuarray.empty_like(self.state)
        self.current, self.voltage = upload(np.zeros(self.port_count)), upload(np.zeros(self.port_count))
        scalar = "double" if electric.dtype == np.float64 else "float"
        self.module = SourceModule(_SOURCE.replace("FIELD", scalar), no_extern_c=True)
        self._h = self.module.get_function("magnetic")
        self._e = self.module.get_function("electric")
        self._boundary = self.module.get_function("boundary")

    @staticmethod
    def _launch(kernel, count, *args):
        kernel(np.int64(count), *args, block=(128, 1, 1), grid=((count+127)//128, 1, 1))

    def step(self):
        self._launch(self._h, self.nh, *self.curl, self.mh, self.sm, self.electric,
                     self.magnetic, self.old_magnetic, self.poffset, self.pindex,
                     self.weights, self.q, self.z, self.soffset, self.l, self.state, self.dt)
        if self.port_count:
            self._launch(self._boundary, self.port_count, self.rows, self.q, self.z,
                         self.soffset, self.foffset, self.f, self.g, self.l, self.state,
                         self.next_state, self.magnetic, self.old_magnetic, self.current, self.voltage)
            self.state, self.next_state = self.next_state, self.state
        self._launch(self._e, self.ne, *self.transpose, self.me, self.se,
                     self.magnetic, self.electric, self.dt)

    def fields(self):
        """Copy fields to the host explicitly, for output or verification."""
        return self.electric.get(), self.magnetic.get()

    def boundary_state(self):
        """Copy independent ADE histories and the latest boundary power traces."""
        flat = self.state.get()
        states = tuple(flat[a:b].copy() for a, b in zip(self._state_offsets[:-1], self._state_offsets[1:]))
        return states, self.current.get()[:self.port_count], self.voltage.get()[:self.port_count]

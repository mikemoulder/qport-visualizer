"""QAOA for portfolio selection: circuits, simulation, and the hybrid loop.

Two variants are built and compared:

penalty  The textbook recipe. Start in |+>^n (every bitstring equally likely),
         alternate the cost layer exp(-i gamma H_pen) with the X mixer
         exp(-i beta sum X_i). The budget constraint is a penalty inside H_pen,
         so probability can leak onto portfolios that break the budget.

xy       Constraint-preserving QAOA (Hadfield et al., "Quantum Alternating
         Operator Ansatz", 2019). Start in the Dicke state, an equal superposition
         of exactly the valid picks (prepared with whichever compiles to fewer
         CNOTs: generic synthesis or the deterministic circuit of Baertschi &
         Eidenbenz, 2019). The XY ring mixer exp(-i beta (XX+YY)/2)
         only swaps an asset in for an asset out, so the state never leaves the
         feasible subspace and the cost needs no penalty.

The classical half of the loop (COBYLA) proposes angles; the quantum half
returns the expected cost. Inside the loop we simulate with a fast exact
kernel built from Qiskit's own gate matrices; `verify_kernel` checks it
against qiskit.quantum_info.Statevector on the real circuits every run.
"""
import time
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RXGate, StatePreparation, XXPlusYYGate
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

VARIANTS = ("penalty", "xy")
GRID = 41  # p=1 landscape resolution per axis


# --- circuit pieces --------------------------------------------------------------
def ring_pairs(n):
    """Even bonds, then odd bonds, then the wrap-around bond (parity ring mixer)."""
    even = [(i, i + 1) for i in range(0, n - 1, 2)]
    odd = [(i, i + 1) for i in range(1, n - 1, 2)]
    wrap = [(n - 1, 0)] if n > 2 else []
    return even + odd + wrap


def dicke_state(n, k):
    """Equal superposition of all n-bit strings with Hamming weight k."""
    amps = np.array([1.0 if bin(i).count("1") == k else 0.0 for i in range(2 ** n)])
    return amps / np.linalg.norm(amps)


def _cry(qc, theta, c, t):
    qc.ry(theta / 2, t)
    qc.cx(c, t)
    qc.ry(-theta / 2, t)
    qc.cx(c, t)


def _ccry(qc, theta, c1, c2, t):
    # The Ry angles add up across CX parity frames to theta * c1 * c2 (4 CNOTs).
    for sign, c in ((1, c1), (-1, c2), (1, c1), (-1, c2)):
        qc.ry(sign * theta / 4, t)
        qc.cx(c, t)


def _scs(qc, l, k):
    """Split & Cyclic Shift block on 1-indexed qubits l-k .. l."""
    q = lambda j: j - 1  # noqa: E731
    qc.cx(q(l - 1), q(l))
    _cry(qc, 2 * np.arccos(np.sqrt(1 / l)), q(l), q(l - 1))
    qc.cx(q(l - 1), q(l))
    for m in range(2, k + 1):
        qc.cx(q(l - m), q(l))
        _ccry(qc, 2 * np.arccos(np.sqrt(m / l)), q(l), q(l - m + 1), q(l - m))
        qc.cx(q(l - m), q(l))


def dicke_circuit(n, k):
    """Deterministic Dicke-state preparation, O(n k) CNOTs (Baertschi & Eidenbenz 2019)."""
    qc = QuantumCircuit(n, name=f"Dicke({n},{k})")
    qc.x(range(n - k, n))
    for l in range(n, k, -1):
        _scs(qc, l, k)
    for l in range(k, 1, -1):
        _scs(qc, l, l - 1)
    return qc


def cx_count(qc):
    return int(transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1)
               .count_ops().get("cx", 0))


def cheapest_dicke(n, k):
    """Compile both Dicke preparations and keep the one with fewer CNOTs.

    Generic state synthesis wins on tiny registers (57 vs 62 CNOTs for 6 choose 3), the
    O(n k) Baertschi-Eidenbenz circuit wins from 8 qubits on (118 vs 247 for 8 choose 4).
    """
    be = dicke_circuit(n, k)
    generic = QuantumCircuit(n, name=f"Dicke({n},{k})")
    generic.append(StatePreparation(dicke_state(n, k), label=f"Dicke({n},{k})"), range(n))
    options = [("Baertschi-Eidenbenz", be, cx_count(be)),
               ("generic state synthesis", generic, cx_count(generic))]
    return min(options, key=lambda o: o[2])


def cost_terms(H):
    """Split a diagonal Ising Hamiltonian into ([(i, h_i)], [(i, j, J_ij)])."""
    fields, couplings = [], []
    for paulis, qubits, coeff in H.to_sparse_list():
        c = float(np.real(coeff))
        if paulis == "Z":
            fields.append((qubits[0], c))
        elif paulis == "ZZ":
            couplings.append((qubits[0], qubits[1], c))
        elif paulis.strip("I"):
            raise ValueError(f"unexpected term {paulis}")
    return fields, couplings


def build_layers(H, p, mixer):
    """Parameterized cost/mixer layers (no initial-state preparation).

    exp(-i gamma h Z) = RZ(2 gamma h) and exp(-i gamma J ZZ) = RZZ(2 gamma J);
    all terms commute, so this is the exact cost unitary, not an approximation.
    """
    n = H.num_qubits
    fields, couplings = cost_terms(H)
    gamma, beta = ParameterVector("γ", p), ParameterVector("β", p)
    qc = QuantumCircuit(n, name=f"QAOA-{mixer} p={p}")
    for layer in range(p):
        for i, j, J in couplings:
            qc.rzz(2 * J * gamma[layer], i, j)
        for i, h in fields:
            qc.rz(2 * h * gamma[layer], i)
        if mixer == "x":
            for q in range(n):
                qc.rx(2 * beta[layer], q)
        else:
            for a, b in ring_pairs(n):
                qc.append(XXPlusYYGate(2 * beta[layer]), [a, b])
        if layer < p - 1:
            qc.barrier()
    return qc, gamma, beta


def cold_start(p, size=0.05):
    """Tiny angles in the adiabatic direction (gamma > 0, beta < 0 with these gate conventions:
    a Trotterized sweep from the mixer's ground state to the cost's ground state)."""
    return np.full(p, size), np.full(p, -size)


def grid_minima(E):
    """Indices (row, col) of local minima of a 2-D grid, lowest first."""
    padded = np.pad(E, 1, constant_values=np.inf)
    rows, cols = E.shape
    mins = []
    for r in range(rows):
        for c in range(cols):
            window = padded[r:r + 3, c:c + 3]
            if E[r, c] <= window.min():
                mins.append((E[r, c], r, c))
    return [(r, c) for _, r, c in sorted(mins)]


def interp_init(gammas, betas):
    """INTERP warm start for depth p+1 from optimized depth-p angles (Zhou et al. 2020)."""
    def grow(v):
        p = len(v)
        v = np.concatenate([[0.0], v, [0.0]])
        return np.array([(i / p) * v[i] + ((p - i) / p) * v[i + 1] for i in range(p + 1)])
    return grow(np.asarray(gammas)), grow(np.asarray(betas))


# --- fast exact kernel ----------------------------------------------------------------
def _apply_1q(psi, U, q, n):
    ax = n - 1 - q  # little-endian: qubit 0 is the last tensor axis
    psi = np.tensordot(U, psi, axes=([1], [ax]))
    return np.moveaxis(psi, 0, ax)


def _apply_2q(psi, U, a, b, n):
    # Qiskit's 4x4 matrix is indexed (x_b x_a) with qarg a as the low bit.
    ax_a, ax_b = n - 1 - a, n - 1 - b
    psi = np.tensordot(U.reshape(2, 2, 2, 2), psi, axes=([2, 3], [ax_b, ax_a]))
    return np.moveaxis(psi, [0, 1], [ax_b, ax_a])


def ising_diagonal(H, n):
    bits = (np.arange(2 ** n)[:, None] >> np.arange(n)) & 1
    z = 1 - 2 * bits
    fields, couplings = cost_terms(H)
    d = np.zeros(2 ** n)
    for i, h in fields:
        d += h * z[:, i]
    for i, j, J in couplings:
        d += J * z[:, i] * z[:, j]
    return d


# --- results ---------------------------------------------------------------------------
@dataclass
class DepthResult:
    p: int
    gammas: np.ndarray
    betas: np.ndarray
    energy: float              # expected cost in true units (penalized for 'penalty')
    probs: np.ndarray          # final probability of every bitstring
    approx_ratio: float
    p_opt: float
    p_feasible: float
    evaluations: int
    seconds: float
    trace: list = field(default_factory=list)  # (params, energy, probs) per evaluation


class QAOASolver:
    def __init__(self, problem, variant):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.problem, self.variant = problem, variant
        n = self.n = problem.n
        self.H, self.scale = problem.ising(variant)
        self.diag = ising_diagonal(self.H, n)
        if variant == "penalty":
            self.target = problem.penalized_costs()   # what the optimizer minimizes
            self.init_vec = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
            self.mixer = "x"
        else:
            self.target = problem.costs
            self.init_vec = dicke_state(n, problem.budget).astype(complex)
            self.mixer = "xy"
        # beta has period pi for both mixers (for the XY ring, beta + pi multiplies in a ZZ per
        # bond, and each qubit sits on two bonds). Centering the window on 0 lets a cold start
        # walk downhill in either direction. gamma has no period in general; we search [0, pi].
        self.beta_min, self.beta_max = -np.pi / 2, np.pi / 2
        self.gamma_max = np.pi
        self._circuits = {}
        self._init = None

    # --- circuits ----------------------------------------------------------------
    def layers(self, p):
        if p not in self._circuits:
            self._circuits[p] = build_layers(self.H, p, self.mixer)
        return self._circuits[p]

    def bind(self, p, gammas, betas):
        qc, g, b = self.layers(p)
        values = dict(zip(g, gammas))
        values.update(zip(b, betas))
        return qc.assign_parameters(values)

    def init_circuit(self):
        """(method name, circuit) preparing the starting state."""
        if self._init is None:
            n = self.n
            if self.variant == "penalty":
                qc = QuantumCircuit(n, name="uniform")
                qc.h(range(n))
                self._init = ("Hadamard on every qubit", qc)
            else:
                name, qc, _ = cheapest_dicke(n, self.problem.budget)
                self._init = (name, qc)
        return self._init

    def full_circuit(self, p, gammas, betas, measure=True):
        """Self-contained circuit (state preparation + layers) for a real backend."""
        qc = QuantumCircuit(self.n, name=f"QPort {self.variant}-QAOA p={p}")
        qc.compose(self.init_circuit()[1], inplace=True)
        qc.barrier()
        qc.compose(self.bind(p, gammas, betas), inplace=True)
        if measure:
            qc.measure_all()
        return qc

    def gate_counts(self, p):
        qc = transpile(self.full_circuit(p, np.full(p, 0.3), np.full(p, 0.3), measure=False),
                       basis_gates=["cx", "rz", "sx", "x"], optimization_level=1)
        ops = qc.count_ops()
        return {"two_qubit": int(ops.get("cx", 0)),
                "one_qubit": int(sum(v for k, v in ops.items() if k in ("rz", "sx", "x"))),
                "depth": int(qc.depth())}

    # --- simulation ----------------------------------------------------------------
    def statevector(self, p, gammas, betas):
        n = self.n
        psi = self.init_vec.reshape((2,) * n)
        flat_diag = self.diag.reshape((2,) * n)
        pairs = ring_pairs(n)
        for g, b in zip(gammas, betas):
            psi = psi * np.exp(-1j * g * flat_diag)
            if self.mixer == "x":
                U = RXGate(2 * b).to_matrix()
                for q in range(n):
                    psi = _apply_1q(psi, U, q, n)
            else:
                U = XXPlusYYGate(2 * b).to_matrix()
                for a, c in pairs:
                    psi = _apply_2q(psi, U, a, c, n)
        return psi.reshape(-1)

    def probs(self, p, gammas, betas):
        return np.abs(self.statevector(p, gammas, betas)) ** 2

    def energy(self, probs):
        return float(probs @ self.target)

    def verify_kernel(self, p=2, seed=0):
        """Max |amplitude difference| between the kernel and Qiskit Statevector."""
        rng = np.random.default_rng(seed)
        g, b = rng.uniform(0, 1, p), rng.uniform(0, 1, p)
        ref = Statevector(self.full_circuit(p, g, b, measure=False)).data
        ours = self.statevector(p, g, b)
        phase = np.vdot(ours, ref)  # the circuit may differ by a global phase
        phase = phase / abs(phase) if abs(phase) > 1e-12 else 1.0
        return float(np.max(np.abs(ours * phase - ref)))

    def landscape(self, grid=GRID):
        gs = np.linspace(0, self.gamma_max, grid)
        bs = np.linspace(self.beta_min, self.beta_max, grid)
        E = np.array([[self.energy(self.probs(1, [g], [b])) for g in gs] for b in bs])
        return gs, bs, E

    def _summarize(self, p, gammas, betas, evaluations, seconds, trace):
        pr = self.probs(p, gammas, betas)
        prob = self.problem
        return DepthResult(p, np.asarray(gammas), np.asarray(betas), self.energy(pr), pr,
                           prob.approx_ratio(pr), float(pr[prob.optimum]),
                           float(pr[prob.feasible].sum()), evaluations, seconds, trace)

    def optimize(self, p, x0_gammas, x0_betas, maxiter=600):
        trace = []

        def objective(x):
            pr = self.probs(p, x[:p], x[p:])
            e = self.energy(pr)
            trace.append((x.copy(), e, pr))
            return e

        t0 = time.perf_counter()
        # Bounded to one period-window so every angle can be drawn on the landscape;
        # points outside it are mirror images under the circuit's symmetries.
        bounds = [(0.0, self.gamma_max)] * p + [(self.beta_min, self.beta_max)] * p
        x0 = np.clip(np.concatenate([x0_gammas, x0_betas]), [b[0] for b in bounds],
                     [b[1] for b in bounds])
        res = minimize(objective, x0, method="COBYLA", bounds=bounds,
                       options={"maxiter": maxiter, "rhobeg": 0.1, "tol": 1e-6})
        return self._summarize(p, res.x[:p], res.x[p:], res.nfev,
                               time.perf_counter() - t0, trace)

    def solve(self, max_depth, landscape=None, starts=3):
        """p=1: refine from the lowest grid basins and a cold start, keep the best;
        then INTERP warm starts up to max_depth."""
        if landscape is None:
            landscape = self.landscape()
        gs, bs, E = landscape
        candidates = [([gs[gi]], [bs[bi]]) for bi, gi in grid_minima(E)[:starts]]
        candidates.append(cold_start(1))
        runs = [self.optimize(1, g, b) for g, b in candidates]
        results = [min(runs, key=lambda r: r.energy)]
        for p in range(2, max_depth + 1):
            g0, b0 = interp_init(results[-1].gammas, results[-1].betas)
            results.append(self.optimize(p, g0, b0))
        return results

    # --- cross-check with a Qiskit primitive -----------------------------------------
    def estimator_check(self, p, gammas, betas):
        """<H> from StatevectorEstimator on the real circuit vs our kernel's probabilities."""
        from qiskit.primitives import StatevectorEstimator
        qc = self.full_circuit(p, gammas, betas, measure=False)
        ev = float(StatevectorEstimator().run([(qc, self.H)]).result()[0].data.evs)
        return ev, float(self.probs(p, gammas, betas) @ self.diag)

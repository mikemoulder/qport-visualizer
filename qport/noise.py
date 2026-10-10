"""How much of QAOA's advantage survives hardware noise?

Each optimized circuit is transpiled to CX plus general single-qubit gates (U)
and simulated as a density matrix. After every CX we apply a two-qubit
depolarizing channel with error e2; after every single-qubit gate a one-qubit
channel with e1 = e2 / 10. The angles are the noiseless optima, as when
parameters are trained in simulation and then sent to a device.

The simulation runs on a small numpy density-matrix kernel (gates and channels
applied to tensor axes); `verify` checks it against qiskit.quantum_info's
DensityMatrix with Kraus channels, which is ~100x slower at 8 qubits.

This is a deliberately simple, transparent model, not a device emulator.
Published two-qubit error rates on current trapped-ion and superconducting
hardware sit roughly in the 0.1%-1% range, which is the sweep used here.
"""
import itertools

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import DensityMatrix, Kraus, Operator, Pauli, SuperOp

ERROR_RATES = (0.0, 0.001, 0.003, 0.005, 0.01, 0.02)


def depolarizing(num_qubits, e):
    """Depolarizing channel: with probability e, replace the state by I/d."""
    d2 = 4 ** num_qubits
    labels = ["".join(s) for s in itertools.product("IXYZ", repeat=num_qubits)]
    ops = [np.sqrt(1 - e * (d2 - 1) / d2) * Pauli(labels[0]).to_matrix()]
    ops += [np.sqrt(e / d2) * Pauli(lab).to_matrix() for lab in labels[1:]]
    return SuperOp(Kraus(ops))


def _compile(circuit):
    return transpile(circuit, basis_gates=["cx", "u"], optimization_level=1)


def noisy_probs_qiskit(circuit, e2):
    """Reference implementation with qiskit.quantum_info (slow; used for verification)."""
    qc = _compile(circuit)
    n = qc.num_qubits
    rho = DensityMatrix.from_label("0" * n)
    ch1 = depolarizing(1, e2 / 10) if e2 else None
    ch2 = depolarizing(2, e2) if e2 else None
    for inst in qc.data:
        op = inst.operation
        if op.name == "barrier":
            continue
        qargs = [qc.find_bit(q).index for q in inst.qubits]
        rho = rho.evolve(Operator(op), qargs)
        if e2 and op.name == "cx":
            rho = rho.evolve(ch2, qargs)
        elif e2 and op.name == "u":
            rho = rho.evolve(ch1, qargs)
    return np.clip(np.real(np.diag(rho.data)), 0, None)


def _apply_gate(rho, U, qargs, n):
    """rho -> U rho U^dagger on the given qubits; rho has 2n tensor axes (rows, then cols)."""
    k = len(qargs)
    Ut = U.reshape((2,) * (2 * k))
    # Qiskit matrices are little-endian in qargs: the first qarg is the lowest bit.
    row_axes = [n - 1 - q for q in reversed(qargs)]
    col_axes = [2 * n - 1 - q for q in reversed(qargs)]
    rho = np.tensordot(Ut, rho, axes=(list(range(k, 2 * k)), row_axes))
    rho = np.moveaxis(rho, list(range(k)), row_axes)
    rho = np.tensordot(Ut.conj(), rho, axes=(list(range(k, 2 * k)), col_axes))
    return np.moveaxis(rho, list(range(k)), col_axes)


def _depolarize(rho, e, qargs, n):
    """rho -> (1 - e) rho + e (Tr_q rho) (x) I/d on the given qubits."""
    if not e:
        return rho
    mixed = rho
    for q in qargs:
        ra, ca = n - 1 - q, 2 * n - 1 - q
        traced = np.trace(mixed, axis1=ra, axis2=ca)
        mixed = np.multiply.outer(traced, np.eye(2) / 2)
        mixed = np.moveaxis(mixed, [-2, -1], [ra, ca])
    return (1 - e) * rho + e * mixed


def noisy_probs(circuit, e2):
    """Measurement distribution of `circuit` (no measurements) under depolarizing noise."""
    qc = _compile(circuit)
    n = qc.num_qubits
    rho = np.zeros((2,) * (2 * n), dtype=complex)
    rho[(0,) * (2 * n)] = 1.0
    cache = {}
    for inst in qc.data:
        op = inst.operation
        if op.name == "barrier":
            continue
        qargs = [qc.find_bit(q).index for q in inst.qubits]
        key = (op.name, tuple(float(x) for x in op.params))
        if key not in cache:
            cache[key] = op.to_matrix()
        rho = _apply_gate(rho, cache[key], qargs, n)
        if e2 and op.name == "cx":
            rho = _depolarize(rho, e2, qargs, n)
        elif e2 and op.name == "u":
            rho = _depolarize(rho, e2 / 10, qargs, n)
    diag = np.real(np.diagonal(rho.reshape(2 ** n, 2 ** n)))
    return np.clip(diag, 0, None)


def verify(circuit, e2=0.01):
    """Max |difference| between the fast kernel and the Qiskit reference."""
    return float(np.max(np.abs(noisy_probs(circuit, e2) - noisy_probs_qiskit(circuit, e2))))


def study(solvers_and_results, problem, error_rates=ERROR_RATES):
    """Rows of {variant, p, e2, approx_ratio, p_opt, p_feasible, two_qubit}."""
    rows = []
    for solver, results in solvers_and_results:
        for r in results:
            circ = solver.full_circuit(r.p, r.gammas, r.betas, measure=False)
            cx = solver.gate_counts(r.p)["two_qubit"]
            for e2 in error_rates:
                pr = noisy_probs(circ, e2)
                pr = pr / pr.sum()
                kept = postselect(pr, problem.feasible)
                rows.append({"variant": solver.variant, "p": r.p, "e2": e2,
                             "approx_ratio": problem.approx_ratio(pr),
                             "p_opt": float(pr[problem.optimum]),
                             "p_feasible": float(pr[problem.feasible].sum()),
                             "approx_ratio_post": problem.approx_ratio(kept),
                             "p_opt_post": float(kept[problem.optimum]),
                             "two_qubit": cx})
    return rows


def postselect(probs, feasible):
    """Discard outcomes that break the budget and renormalize.

    For XY-QAOA an invalid outcome can only come from an error, so this is a
    free, physically motivated error-mitigation step (it costs extra shots).
    """
    kept = np.where(feasible, probs, 0.0)
    total = kept.sum()
    return kept / total if total > 0 else kept

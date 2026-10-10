"""Qollab smoke test #2: single-circuit IonQ job, packed circuits, settings form.

Paste into main.py and add `pylatexenc` to requirements.txt (or the "Qiskit
Visualizations" preset in the Libraries list). Run once on the Built-in
simulator and once on an IonQ remote simulator. Qollab calls main() itself,
so don't call it here: that would submit a second IonQ job.
"""
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp

N = 6
rng = np.random.default_rng(0)
Q = rng.normal(size=(N, N))
Q = (Q + Q.T) / 2


def qubo_to_ising(Q):
    n = len(Q)
    terms = []
    for i in range(n):
        terms.append(("Z", [i], -Q[i, i] / 2))
        for j in range(i + 1, n):
            w = Q[i, j] / 2
            terms += [("ZZ", [i, j], w), ("Z", [i], -w), ("Z", [j], -w)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify()


def main(risk: float = 0.5, budget: int = 3, reps: int = 1, show_plots: bool = True):
    print(f"=== SETTINGS FORM === risk={risk} budget={budget} reps={reps} show_plots={show_plots}")
    print(f"backend: {type(backend).__module__}.{type(backend).__name__} "  # noqa: F821 (injected by Qollab)
          f"name={getattr(backend, 'name', '?')}")  # noqa: F821

    H = qubo_to_ising(Q)
    ansatz = qaoa_ansatz(H, reps=1)

    # Two independent 6-qubit experiments packed into ONE 12-qubit circuit = one job.
    packed = QuantumCircuit(2 * N)
    packed.compose(ansatz.assign_parameters([0.3, 0.7]), qubits=range(N), inplace=True)
    packed.compose(ansatz.assign_parameters([0.6, 0.2]), qubits=range(N, 2 * N), inplace=True)
    packed.measure_all()

    try:
        job_circ = transpile(packed, backend)  # noqa: F821
        print("transpile: OK, ops =", dict(job_circ.count_ops()))
    except Exception as e:  # noqa: BLE001
        job_circ = packed
        print(f"transpile: FAILED ({type(e).__name__}: {e}); submitting untranspiled")

    t0 = time.perf_counter()
    try:
        counts = backend.run(job_circ).result().get_counts()  # noqa: F821
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] single 12-qubit job: {type(e).__name__}: {e}")
        return
    print(f"[OK  ] single 12-qubit job in {time.perf_counter() - t0:.1f}s, "
          f"{sum(counts.values())} shots, {len(counts)} distinct outcomes")

    # Bitstrings are little-endian: rightmost N chars = qubits 0..N-1 (experiment A).
    exp_a, exp_b = {}, {}
    for bits, c in counts.items():
        bits = bits.replace(" ", "")
        exp_a[bits[-N:]] = exp_a.get(bits[-N:], 0) + c
        exp_b[bits[:-N]] = exp_b.get(bits[:-N], 0) + c
    top = lambda d: sorted(d.items(), key=lambda kv: -kv[1])[:3]  # noqa: E731
    print("experiment A top3:", top(exp_a))
    print("experiment B top3:", top(exp_b))

    if show_plots:
        import matplotlib.pyplot as plt
        # Qollab's editor is dark; default transparent figures render near-invisible.
        plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white",
                             "axes.facecolor": "white"})
        try:
            ansatz.draw(output="mpl", fold=30)
            plt.show()
            print("[OK  ] circuit mpl drawing (check it rendered)")
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] circuit mpl drawing: {type(e).__name__}: {e}")
        fig, ax = plt.subplots(figsize=(6, 3))
        keys = sorted(exp_a, key=lambda k: -exp_a[k])[:10]
        ax.bar(keys, [exp_a[k] for k in keys])
        ax.set_title("Experiment A: top-10 portfolios (check this rendered)")
        ax.tick_params(axis="x", rotation=60)
        plt.tight_layout()
        plt.show()

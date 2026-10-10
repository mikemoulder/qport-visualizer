"""Qollab runtime smoke test.

Paste into main.py of a Qollab Python project and press Run (Built-in simulator
first, then an IonQ remote simulator via Select QPU).
Every check is isolated, so one failure won't hide the rest.
Copy the full output back.
"""
import sys
import time
import platform

RESULTS = []


def check(name, fn):
    t0 = time.perf_counter()
    try:
        out = fn()
        status = "OK  "
    except Exception as e:  # noqa: BLE001 - we want to report everything
        out = f"{type(e).__name__}: {e}"
        status = "FAIL"
    dt = time.perf_counter() - t0
    line = f"[{status}] {name} ({dt:.2f}s) -> {out}"
    RESULTS.append(line)
    print(line, flush=True)


# 1. Environment ------------------------------------------------------------
print("=== ENVIRONMENT ===")
print("python:", sys.version.replace("\n", " "))
print("sys.platform:", sys.platform, "| machine:", platform.machine())
print("in-browser (pyodide/emscripten):", sys.platform == "emscripten" or "pyodide" in sys.modules)
user_globals = sorted(k for k in globals() if not k.startswith("_"))
print("globals at start (Qollab may inject a backend here):", user_globals)


def _cwd():
    import os
    return f"cwd={os.getcwd()} files={sorted(os.listdir('.'))[:30]}"


check("filesystem / bundled files visible", _cwd)

# 2. Package availability ---------------------------------------------------
print("\n=== PACKAGES ===")


def _ver(mod):
    def f():
        m = __import__(mod)
        return getattr(m, "__version__", "imported (no __version__)")
    return f


for mod in [
    "qiskit", "qiskit_aer", "qiskit_ionq", "qiskit_ibm_runtime",
    "qiskit_algorithms", "qiskit_optimization", "qiskit_finance",
    "numpy", "scipy", "pandas", "matplotlib", "pylatexenc",
    "plotly", "ipywidgets", "IPython", "networkx",
]:
    check(f"import {mod}", _ver(mod))

# 3. Hand-rolled QAOA on a 6-asset toy QUBO ---------------------------------
print("\n=== QAOA CORE ===")
try:
    import numpy as np
except ImportError:
    print("[FAIL] numpy missing - cannot run QAOA checks. Copy the output above back.")
    raise SystemExit(0)

N = 6
rng = np.random.default_rng(0)
Q = rng.normal(size=(N, N))
Q = (Q + Q.T) / 2


def qubo_to_ising():
    from qiskit.quantum_info import SparsePauliOp
    # x_i = (1 - z_i)/2  ->  x^T Q x expressed in Z terms (constant dropped)
    terms = []
    for i in range(N):
        for j in range(N):
            if i == j:
                terms.append(("Z", [i], -Q[i, i] / 2))
            elif i < j:
                w = Q[i, j] / 2  # Q_ij + Q_ji = 2 Q_ij, times 1/4
                terms.append(("ZZ", [i, j], w))
                terms.append(("Z", [i], -w))
                terms.append(("Z", [j], -w))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=N).simplify()


H = None


def _ham():
    global H
    H = qubo_to_ising()
    return f"{len(H)} Pauli terms"


check("QUBO -> SparsePauliOp", _ham)

ansatz = None


def _ansatz():
    global ansatz
    try:
        from qiskit.circuit.library import qaoa_ansatz  # Qiskit >= 2.1
        ansatz = qaoa_ansatz(H, reps=1)
        src = "qaoa_ansatz()"
    except ImportError:
        from qiskit.circuit.library import QAOAAnsatz
        ansatz = QAOAAnsatz(H, reps=1)
        src = "QAOAAnsatz"
    return f"{src}: {ansatz.num_qubits} qubits, {ansatz.num_parameters} params"


check("build QAOA ansatz (p=1)", _ansatz)


def _draw_text():
    return "\n" + str(ansatz.draw(output="text", fold=100))


check("circuit text drawing", _draw_text)


def energy(params):
    from qiskit.quantum_info import Statevector
    sv = Statevector(ansatz.assign_parameters(params))
    return float(np.real(sv.expectation_value(H)))


def _estimator():
    from qiskit.primitives import StatevectorEstimator
    res = StatevectorEstimator().run([(ansatz, H, [0.3, 0.7])]).result()
    return f"<H> = {float(res[0].data.evs):.4f}"


check("StatevectorEstimator", _estimator)


def _landscape():
    grid = np.linspace(0, np.pi, 15)
    t0 = time.perf_counter()
    E = [[energy([b, g]) for g in grid] for b in grid]
    per = (time.perf_counter() - t0) / grid.size ** 2
    return f"15x15 grid, min={np.min(E):.3f}, {per * 1000:.1f} ms/point"


check("p=1 energy landscape (speed test)", _landscape)


def _optimize():
    from scipy.optimize import minimize
    trace = []
    res = minimize(lambda p: trace.append(energy(p)) or trace[-1],
                   x0=[0.5, 0.5], method="COBYLA", options={"maxiter": 60})
    return f"E*={res.fun:.4f} after {len(trace)} evals"


check("scipy COBYLA hybrid loop", _optimize)


def _exact():
    best = min(range(2 ** N),
               key=lambda k: (x := np.array([(k >> i) & 1 for i in range(N)])) @ Q @ x)
    return f"brute-force optimum bitstring={best:0{N}b}"


check("classical brute force (2^6)", _exact)


def _sampler():
    from qiskit.primitives import StatevectorSampler
    qc = ansatz.assign_parameters([0.3, 0.7])
    qc.measure_all()
    counts = StatevectorSampler().run([qc], shots=1024).result()[0].data.meas.get_counts()
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
    return f"top3={top}"


check("StatevectorSampler (shots)", _sampler)


def _aer():
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    qc = ansatz.assign_parameters([0.3, 0.7])
    qc.measure_all()
    sim = AerSimulator()
    counts = sim.run(transpile(qc, sim), shots=1024).result().get_counts()
    return f"{len(counts)} distinct outcomes"


check("qiskit_aer run", _aer)

# 4. Output / UI rendering --------------------------------------------------
print("\n=== OUTPUT / UI ===")


def _mpl():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["a", "b", "c"], [1, 3, 2])
    ax.set_title("SMOKE TEST: do you see this bar chart?")
    plt.show()
    return f"backend={plt.get_backend()} (check visually that the chart rendered)"


check("matplotlib plt.show()", _mpl)


def _mpl_circuit():
    import matplotlib.pyplot as plt
    ansatz.draw(output="mpl", fold=40)
    plt.show()
    return "drawn (check visually)"


check("circuit draw output='mpl'", _mpl_circuit)


# 5. Injected backend (from the Select QPU dialog) -------------------------
print("\n=== BACKEND ===")


def _find_backend():
    import builtins
    b = globals().get("backend", getattr(builtins, "backend", None))
    if b is None:
        raise LookupError("no `backend` in globals or builtins")
    return b


def _backend_info():
    b = _find_backend()
    return (f"type={type(b).__module__}.{type(b).__name__} "
            f"name={getattr(b, 'name', '?')} num_qubits={getattr(b, 'num_qubits', '?')}")


check("injected `backend` object", _backend_info)


def _backend_run():
    # Two circuits in ONE run() call = one job; this is the IonQ-safe pattern.
    from qiskit import transpile
    b = _find_backend()
    circs = []
    for params in ([0.3, 0.7], [0.6, 0.2]):
        qc = ansatz.assign_parameters(params)
        qc.measure_all()
        circs.append(qc)
    try:
        circs = transpile(circs, b)
        how = "transpiled"
    except Exception as e:  # noqa: BLE001
        how = f"untranspiled ({type(e).__name__})"
    result = b.run(circs).result()
    counts = [result.get_counts(i) for i in range(len(circs))]
    return f"{how}; 2 circuits in one job -> {[sum(c.values()) for c in counts]} shots each"


check("backend.run([2 circuits]) single job", _backend_run)

# Summary --------------------------------------------------------------------
print("\n=== SUMMARY ===")
for line in RESULTS:
    print(line.split("\n")[0])

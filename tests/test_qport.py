"""Correctness checks for every claim the project makes.

Run with:  python -m pytest tests/   (or simply: python tests/test_qport.py)
"""
import os
import sys
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # data/ path is relative

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit.library import qaoa_ansatz  # noqa: E402
from qiskit.providers.basic_provider import BasicSimulator  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402

from qport import classical, noise  # noqa: E402
from qport.data import estimate, load_prices, split_returns  # noqa: E402
from qport.problem import PortfolioProblem, all_bitstrings, qubo_to_ising, qubo_value  # noqa: E402
from qport.qaoa import (QAOASolver, cheapest_dicke, dicke_circuit, dicke_state,  # noqa: E402
                        interp_init, ising_diagonal)


def random_problem(n=6, budget=3, q=2.0, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n)) * 0.2
    sigma = A @ A.T + np.eye(n) * 0.02
    mu = rng.uniform(0.0, 0.4, n)
    return PortfolioProblem([f"A{i}" for i in range(n)], mu, sigma, budget, q)


def real_problem():
    m = load_prices().subset(["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ"])
    train, _, _ = split_returns(m, "2023-12-31")
    mu, sigma = estimate(train)
    return PortfolioProblem(m.tickers, mu, sigma, 3, 3.0)


# --- formulation ----------------------------------------------------------------------
def test_ising_matches_qubo_on_every_bitstring():
    rng = np.random.default_rng(1)
    for n in (3, 5, 7):
        Q = rng.normal(size=(n, n))
        Q = (Q + Q.T) / 2
        H, offset = qubo_to_ising(Q, const=0.7)
        diag = np.real(np.diag(H.to_matrix()))
        assert np.allclose(diag + offset, qubo_value(Q, all_bitstrings(n), 0.7))


def test_kernel_diagonal_matches_qiskit_matrix():
    p = random_problem(5, 2)
    H, _ = p.ising("penalty")
    assert np.allclose(ising_diagonal(H, 5), np.real(np.diag(H.to_matrix())))


def test_penalized_qubo_minimum_is_the_constrained_optimum():
    for seed in range(5):
        p = random_problem(6, 3, seed=seed)
        assert int(np.argmin(p.penalized_costs())) == p.optimum
        assert p.feasible[p.optimum]
        assert 0 < p.min_penalty <= p.safe_penalty


def test_too_small_penalty_is_rejected():
    p = random_problem()
    try:
        PortfolioProblem(p.tickers, p.mu, p.sigma, 3, 2.0, penalty=0.5 * p.min_penalty)
    except ValueError:
        return
    raise AssertionError("a penalty below the minimum should be rejected")


def test_exact_enumeration_agrees_and_counts_combinations():
    p = real_problem()
    ex = classical.exact(p)
    assert ex["index"] == p.optimum
    assert ex["evaluated"] == comb(6, 3)
    assert abs(ex["cost"] - p.best_cost) < 1e-12


def test_approx_ratio_endpoints():
    p = random_problem()
    onehot = np.zeros(2 ** p.n)
    onehot[p.optimum] = 1
    assert abs(p.approx_ratio(onehot) - 1) < 1e-12
    invalid = np.zeros(2 ** p.n)
    invalid[0] = 1  # picks nothing: invalid, scored as worst
    assert abs(p.approx_ratio(invalid)) < 1e-12


# --- circuits -----------------------------------------------------------------------------
def test_dicke_circuit_is_exact():
    for n, k in [(3, 1), (4, 2), (5, 2), (6, 3), (7, 3), (6, 5)]:
        sv = Statevector(dicke_circuit(n, k)).data
        assert abs(abs(np.vdot(dicke_state(n, k), sv)) ** 2 - 1) < 1e-10


def test_cheapest_dicke_is_exact_and_no_worse_than_either_option():
    for n, k in [(4, 2), (6, 3), (8, 4)]:
        name, qc, cx = cheapest_dicke(n, k)
        sv = Statevector(qc).data
        assert abs(abs(np.vdot(dicke_state(n, k), sv)) ** 2 - 1) < 1e-10
        from qport.qaoa import cx_count
        assert cx <= cx_count(dicke_circuit(n, k))


def test_kernel_matches_qiskit_statevector():
    for n, budget in [(4, 2), (6, 3), (7, 2)]:
        prob = random_problem(n, budget, seed=n)
        for variant in ("penalty", "xy"):
            s = QAOASolver(prob, variant)
            for p in (1, 2, 3):
                assert s.verify_kernel(p=p, seed=p) < 1e-10


def test_penalty_ansatz_matches_qiskit_library_qaoa():
    prob = random_problem(5, 2, seed=3)
    s = QAOASolver(prob, "penalty")
    lib = qaoa_ansatz(s.H, reps=2)  # parameters sorted: β[0], β[1], γ[0], γ[1]
    g, b = np.array([0.4, 0.9]), np.array([-0.3, 0.2])
    ref = Statevector(lib.assign_parameters(np.concatenate([b, g]))).probabilities()
    assert np.allclose(ref, s.probs(2, g, b), atol=1e-10)


def test_xy_mixer_never_leaves_the_valid_subspace():
    prob = random_problem(6, 3)
    s = QAOASolver(prob, "xy")
    rng = np.random.default_rng(7)
    for p in (1, 2, 3):
        pr = s.probs(p, rng.uniform(-3, 3, p), rng.uniform(-3, 3, p))
        assert abs(pr[prob.feasible].sum() - 1) < 1e-12


def test_estimator_primitive_agrees_with_kernel():
    prob = random_problem(5, 2, seed=11)
    for variant in ("penalty", "xy"):
        ev, ours = QAOASolver(prob, variant).estimator_check(2, [0.3, 0.8], [-0.4, 0.2])
        assert abs(ev - ours) < 1e-10


def test_interp_init_grows_by_one_layer():
    g, b = interp_init([0.5], [-0.3])
    assert g.shape == (2,) and b.shape == (2,)
    assert np.allclose(g, [0.5, 0.5]) and np.allclose(b, [-0.3, -0.3])


def test_qaoa_beats_its_starting_state_and_xy_beats_random():
    prob = real_problem()
    for variant in ("penalty", "xy"):
        s = QAOASolver(prob, variant)
        r = s.solve(1)[0]
        start = s.energy(s.probs(1, [0.0], [0.0]))
        assert r.energy < start
    xy = QAOASolver(prob, "xy").solve(1)[0]
    assert xy.p_opt > 1 / prob.feasible.sum()


# --- noise -------------------------------------------------------------------------------
def test_noiseless_density_matrix_equals_kernel():
    prob = random_problem(5, 2, seed=2)
    s = QAOASolver(prob, "xy")
    g, b = [0.7], [-0.4]
    dm = noise.noisy_probs(s.full_circuit(1, g, b, measure=False), 0.0)
    assert np.allclose(dm, s.probs(1, g, b), atol=1e-10)


def test_fast_density_matrix_matches_qiskit_reference():
    for n, budget in [(4, 2), (5, 2)]:
        prob = random_problem(n, budget, seed=n)
        for variant in ("penalty", "xy"):
            circ = QAOASolver(prob, variant).full_circuit(2, [0.4, 0.7], [-0.3, 0.2], measure=False)
            assert noise.verify(circ, e2=0.02) < 1e-10


def test_noise_breaks_the_constraint_and_postselection_restores_it():
    prob = random_problem(5, 2, seed=2)
    s = QAOASolver(prob, "xy")
    pr = noise.noisy_probs(s.full_circuit(1, [0.7], [-0.4], measure=False), 0.02)
    assert abs(pr.sum() - 1) < 1e-9
    assert pr[prob.feasible].sum() < 0.99
    kept = noise.postselect(pr, prob.feasible)
    assert abs(kept[prob.feasible].sum() - 1) < 1e-12


def test_noise_fit_recovers_a_known_error_rate():
    prob = random_problem(5, 2, seed=4)
    circ = QAOASolver(prob, "xy").full_circuit(1, [0.7], [-0.4], measure=False)
    target = noise.noisy_probs(circ, 0.0125)
    e2, _, tvd = noise.fit_error_rate(circ, target / target.sum())
    assert abs(e2 - 0.0125) < 0.001 and tvd < 0.01


# --- conventions and evaluation ---------------------------------------------------------
def test_counts_bitstring_maps_to_asset_index():
    # Prepare x = assets {0, 2} (k = 0b000101 = 5) and check the counts key decodes to k.
    qc = QuantumCircuit(6)
    qc.x([0, 2])
    qc.measure_all()
    counts = BasicSimulator().run(qc, shots=10).result().get_counts()
    (key,) = counts
    assert int(key.replace(" ", ""), 2) == 5


def test_backtest_constant_returns():
    daily = np.full((252, 2), 0.001)
    bt = classical.backtest(np.array([0.5, 0.5]), daily)
    assert abs(bt["cagr"] - (1.001 ** 252 - 1)) < 1e-9
    assert bt["max_drawdown"] == 0.0


def test_markowitz_weights_are_a_valid_allocation():
    p = real_problem()
    w, f = classical.markowitz(p.mu, p.sigma, 3.0)
    assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all()
    assert f <= p.best_cost + 1e-9  # continuous relaxation can only do better


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")

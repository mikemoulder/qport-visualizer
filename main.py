"""QPort Visualizer: hybrid quantum-classical portfolio selection, made visible.

Qollab calls main() automatically and turns its parameters into the Run
settings form. Choose a backend with Select QPU; the run sends exactly one
job to it (IonQ backends accept one job per run), and every other quantum
evaluation runs on the in-browser simulator.
"""
import textwrap
import time

import numpy as np

from qport import classical, noise, report, viz
from qport.data import load_prices, split_returns, estimate
from qport.problem import PortfolioProblem, label
from qport.qaoa import QAOASolver, VARIANTS, cold_start

DEFAULT_TICKERS = "AAPL,MSFT,NVDA,JPM,XOM,JNJ"


def main(tickers: str = DEFAULT_TICKERS, budget: int = 3, risk_aversion: float = 3.0,
         max_depth: int = 3, train_end: str = "2023-12-31", final_variant: str = "xy",
         final_depth: int = 1, noise_study: bool = True, penalty_study: bool = False,
         show_figures: bool = True, run_on_backend: bool = True):
    t_start = time.perf_counter()
    viz.style()
    figs = show_figures

    # ------------------------------------------------------------------ settings
    market = load_prices()
    chosen = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    unknown = [t for t in chosen if t not in market.tickers]
    if unknown or not 3 <= len(chosen) <= 10:
        print(f"Tickers must be 3-10 of {', '.join(market.tickers)}; got {chosen} "
              f"(unknown: {unknown}).")
        return
    if final_variant not in VARIANTS:
        print(f"final_variant must be one of {VARIANTS}; got {final_variant!r}.")
        return
    max_depth = max(1, min(int(max_depth), 5))
    final_depth = max(1, min(int(final_depth), max_depth))

    print("QPort Visualizer · hybrid quantum-classical portfolio selection")
    print(f"settings: tickers={','.join(chosen)} budget={budget} risk_aversion={risk_aversion} "
          f"max_depth={max_depth} train_end={train_end} final={final_variant} p={final_depth}")

    # ------------------------------------------------------------------ 1. data
    market = market.subset(chosen)
    try:
        train, test, test_dates = split_returns(market, train_end)
    except ValueError as e:
        print(e)
        return
    mu, sigma = estimate(train)
    window = f"{market.dates[0]} to {train_end}"
    report.header(1, "The data")
    report.explain(f"""
        Daily adjusted closing prices for {len(chosen)} US large caps. We estimate each
        asset's expected return and the covariance between assets from {window}
        ({len(train)} trading days). Everything after {train_end} ({len(test)} days) is held
        back to test the chosen portfolios on data no optimizer has seen.""")
    vol = np.sqrt(np.diag(sigma))
    report.table([{"t": t, "mu": m, "vol": v, "sr": m / v} for t, m, v in zip(chosen, mu, vol)],
                 [("t", "asset", "{}"), ("mu", "return/yr", "{:+.1%}"),
                  ("vol", "volatility/yr", "{:.1%}"), ("sr", "return/risk", "{:.2f}")])
    report.show(*viz.fig_data(chosen, mu, sigma, window), figs)

    # ------------------------------------------------------------------ 2. problem
    try:
        problem = PortfolioProblem(chosen, mu, sigma, budget, risk_aversion)
    except ValueError as e:
        print(e)
        return
    n = problem.n
    H, _ = problem.ising("penalty")
    report.header(2, "The problem, rewritten for a quantum computer")
    report.explain(f"""
        Choose exactly {budget} of the {n} assets and hold them in equal weights. Score a
        choice x (one bit per asset) by  f(x) = risk_aversion x variance - expected return.
        With risk_aversion = {risk_aversion}, a portfolio must earn its risk.

        A quantum optimizer needs this as an energy. Squaring out the budget rule gives a
        QUBO, x^T Q x, with a penalty weight of {problem.penalty:.3f} for every asset too many or
        too few. That weight is provably large enough: breaking the budget can never pay off.
        Substituting x = (1 - Z)/2 turns the QUBO into a Hamiltonian of {len(H)} Pauli Z and
        ZZ terms on {n} qubits, one qubit per asset. Its lowest-energy state is the best
        portfolio.""")
    report.show(*viz.fig_problem(problem), figs)

    # ------------------------------------------------------------------ 3. classical
    report.header(3, "Classical baselines")
    ex = classical.exact(problem)
    naive_k = classical.naive_top_sharpe(problem)
    w_mk, f_mk = classical.markowitz(mu, sigma, risk_aversion)
    rnd = classical.uniform_guess(problem)
    uniform_valid = np.where(problem.feasible, 1.0, 0.0) / problem.feasible.sum()
    random_ar = problem.approx_ratio(uniform_valid)
    report.explain(f"""
        With {n} assets there are only {ex['evaluated']} valid portfolios, so a laptop checks
        them all in {ex['seconds'] * 1e3:.2f} ms. That exact answer is the ground truth we
        score QAOA against. At this size QAOA cannot beat it, and we don't claim it does.
        The questions here are how well QAOA concentrates probability on good answers,
        and how the circuit design and hardware noise change that.

        Markowitz optimizes continuous weights with no limit on the number of assets.
        That's what a practitioner would run, and a lower bound on f.""")
    rows = [{"m": "Exact (all valid picks)", "pick": viz.short(problem.optimum, chosen),
             "f": problem.best_cost, **problem.stats(problem.optimum)},
            {"m": "Naive top-Sharpe", "pick": viz.short(naive_k, chosen),
             "f": float(problem.costs[naive_k]), **problem.stats(naive_k)}]
    mk_ret, mk_vol = float(mu @ w_mk), float(np.sqrt(w_mk @ sigma @ w_mk))
    rows.append({"m": "Markowitz (continuous)",
                 "pick": ", ".join(f"{t} {w:.0%}" for t, w in zip(chosen, w_mk) if w > 0.005),
                 "f": f_mk, "return": mk_ret, "vol": mk_vol, "sharpe": mk_ret / mk_vol})
    report.table(rows, [("m", "method", "{}"), ("pick", "portfolio", "{}"), ("f", "f(x)", "{:.4f}"),
                        ("return", "return", "{:+.1%}"), ("vol", "vol", "{:.1%}"),
                        ("sharpe", "Sharpe", "{:.2f}")], "In-sample (estimation window):")

    # ------------------------------------------------------------------ 4. QAOA
    report.header(4, "QAOA: the hybrid quantum-classical loop")
    report.explain(f"""
        QAOA prepares a starting state, then alternates two layers p times: a cost layer
        that adds a phase to each portfolio in proportion to its energy (angle gamma), and
        a mixer layer that moves amplitude between portfolios (angle beta). A classical
        optimizer (COBYLA) tunes the angles to minimize the expected cost measured from
        the circuit. We compare two designs:

        Penalty + X mixer (textbook): start in an equal superposition of all {2 ** n}
        bitstrings; the budget rule lives only in the penalty, so probability can leak onto
        invalid picks.

        Dicke + XY mixer (constraint-preserving): start in an equal superposition of only
        the {ex['evaluated']} valid picks (a Dicke state), and mix with XY gates that swap
        one asset in for one out. The budget can never be broken, so no penalty is needed.""")
    solvers, results, scapes = {}, {}, {}
    for v in VARIANTS:
        s = QAOASolver(problem, v)
        err = s.verify_kernel(p=2)
        t0 = time.perf_counter()
        scapes[v] = s.landscape()
        results[v] = s.solve(max_depth, scapes[v])
        solvers[v] = s
        print(f"  {viz.METHOD[v]['name']}: optimized p=1..{max_depth} in "
              f"{time.perf_counter() - t0:.1f}s; simulator verified against "
              f"qiskit.quantum_info.Statevector (max amplitude error {err:.1e})")
    print()
    s = solvers[final_variant]
    r_final = results[final_variant][final_depth - 1]
    describe_circuit(s, final_depth, final_variant)
    rows = []
    for v in VARIANTS:
        for r in results[v]:
            gc = solvers[v].gate_counts(r.p)
            rows.append({"m": viz.METHOD[v]["name"], "p": r.p, "ar": r.approx_ratio, "popt": r.p_opt,
                         "valid": r.p_feasible, "x": r.p_opt / rnd["feasible_only"],
                         "sts": report.shots_to_solution(r.p_opt), "cx": gc["two_qubit"],
                         "ev": r.evaluations})
    rows.append({"m": "Random valid guess", "p": None, "ar": random_ar, "popt": rnd["feasible_only"],
                 "valid": 1.0, "x": 1.0, "sts": report.shots_to_solution(rnd["feasible_only"]),
                 "cx": 0, "ev": None})
    rows.append({"m": "Random bitstring", "p": None, "ar": problem.approx_ratio(
        np.full(2 ** n, 2.0 ** -n)), "popt": rnd["any_bitstring"], "valid": problem.feasible.mean(),
        "x": rnd["any_bitstring"] / rnd["feasible_only"],
        "sts": report.shots_to_solution(rnd["any_bitstring"]), "cx": 0, "ev": None})
    report.table(rows, [("m", "method", "{}"), ("p", "p", "{}"), ("ar", "approx ratio", "{:.3f}"),
                        ("popt", "P(best)", "{:.1%}"), ("valid", "P(valid)", "{:.1%}"),
                        ("x", "vs random", "{:.1f}x"), ("sts", "shots for 99%", "{}"),
                        ("cx", "CNOTs", "{}"), ("ev", "evaluations", "{}")],
                 "Noiseless results (approx ratio: 1 = always optimal, invalid picks count as worst):")
    # Cold starts (angles ~0, i.e. almost the bare initial state) show the optimizer's journey;
    # the reported results above start from the p=1 grid scan and INTERP warm starts.
    colds = {v: solvers[v].optimize(1, *cold_start(1)) for v in VARIANTS}
    report.show(*viz.fig_landscapes([(solvers[v], scapes[v], results[v][0], colds[v])
                                     for v in VARIANTS]), figs)
    flow = (colds[final_variant] if final_depth == 1 else
            s.optimize(final_depth, *cold_start(final_depth)))
    report.show(*viz.fig_probability_flow(problem, flow, final_variant), figs)
    report.show(*viz.fig_distribution(problem, {v: results[v][final_depth - 1] for v in VARIANTS}),
                figs)
    if max_depth > 1:
        report.show(*viz.fig_depth(problem, results, random_ar), figs)

    if penalty_study:
        report.explain("""
            Penalty study: the textbook variant needs a penalty weight. Too small and
            probability leaks onto invalid picks; too large and the penalty drowns out
            the differences between valid portfolios that we care about.""")
        prow = []
        for mult in (1.1, 2, 5, None):
            lam = mult * problem.min_penalty if mult else problem.safe_penalty
            pp = PortfolioProblem(chosen, mu, sigma, budget, risk_aversion, penalty=lam)
            r = QAOASolver(pp, "penalty").solve(1)[0]
            prow.append({"w": f"{mult}x minimum" if mult else "provably safe (default)", "lam": lam,
                         "ar": r.approx_ratio, "popt": r.p_opt, "valid": r.p_feasible})
        report.table(prow, [("w", "penalty weight", "{}"), ("lam", "lambda", "{:.4f}"),
                            ("ar", "approx ratio", "{:.3f}"), ("popt", "P(best)", "{:.1%}"),
                            ("valid", "P(valid)", "{:.1%}")], "Penalty QAOA at p=1:")

    # ------------------------------------------------------------------ 5. noise
    if noise_study:
        report.header(5, "Hardware noise: why the shallow circuit is sent to the backend")
        report.explain("""
            Real qubits make errors. We re-simulate each optimized circuit as a density
            matrix, adding a depolarizing error after every CNOT (and a tenth of that after
            every single-qubit gate). The XY design pays for its constraint guarantee with
            extra CNOTs (the Dicke-state preparation and the XY gates), so it has more to lose.
            With XY, any invalid sample must be an error, so we can also discard those
            samples: a free error-mitigation step.""")
        rates = noise.ERROR_RATES
        if n >= 8:
            rates = (0.0, 0.003, 0.01, 0.02)
            print(f"  ({n} qubits: density matrices have {4 ** n:,} entries, so we sweep only "
                  f"{len(rates)} error rates to keep the run short.)")
        t0 = time.perf_counter()
        noise_rows = noise.study([(solvers[v], results[v]) for v in VARIANTS], problem, rates)
        print(f"  {len(noise_rows)} noisy density-matrix simulations in {time.perf_counter() - t0:.1f}s\n")
        report.table(noise_rows, [("variant", "variant", "{}"), ("p", "p", "{}"),
                                  ("two_qubit", "CNOTs", "{}"), ("e2", "2q error", "{:.1%}"),
                                  ("approx_ratio", "approx ratio", "{:.3f}"),
                                  ("p_opt", "P(best)", "{:.1%}"), ("p_feasible", "P(valid)", "{:.1%}"),
                                  ("approx_ratio_post", "ratio, invalid discarded", "{:.3f}")])
        report.show(*viz.fig_noise(noise_rows, random_ar), figs)

    # ------------------------------------------------------------------ 6. backtest
    report.header(6, "Out of sample: would these picks have held up?")
    picks = {v: int(np.flatnonzero(problem.feasible)[np.argmax(results[v][final_depth - 1]
                                                             .probs[problem.feasible])])
             for v in VARIANTS}
    shots = 100
    hit = {v: 1 - (1 - results[v][final_depth - 1].p_opt) ** shots for v in VARIANTS}
    report.explain(f"""
        A QAOA run returns samples. Its most frequent valid pick at p={final_depth}:
        XY-QAOA {viz.short(picks['xy'], chosen)}, penalty-QAOA {viz.short(picks['penalty'], chosen)};
        the exact optimum is {viz.short(problem.optimum, chosen)}. In practice you score every
        sampled portfolio classically (cheap) and keep the best: {shots} shots contain the
        optimum with probability {hit['xy']:.1%} for XY-QAOA and {hit['penalty']:.1%} for
        penalty-QAOA.

        We now hold each portfolio, unchanged, from {test_dates[0]} to {test_dates[-1]}. This
        tests the estimates as much as the optimizer: past returns are a noisy guide to
        future ones.""")
    entries = [("exact", problem.optimum)]
    for v in VARIANTS:
        if picks[v] != problem.optimum:
            entries.append((v, picks[v]))
    portfolios = []
    for key, k in entries:
        same = [f"{v}-QAOA" for v in VARIANTS if picks[v] == k and key == "exact"]
        if key == "exact":
            lab = f"{viz.short(k, chosen)}: exact optimum" + (
                f" (= most frequent pick of {' and '.join(same)})" if same else "")
        else:
            lab = f"{viz.short(k, chosen)}: most frequent pick of {key}-QAOA"
        w = classical.weights_of(k, n)
        portfolios.append((key, lab, w, classical.backtest(w, test)))
    if naive_k not in [k for _, k in entries]:
        w = classical.weights_of(naive_k, n)
        portfolios.append(("naive", f"{viz.short(naive_k, chosen)}: naive top-Sharpe", w,
                           classical.backtest(w, test)))
    portfolios.append(("markowitz", "Markowitz (continuous)", w_mk, classical.backtest(w_mk, test)))
    w_eq = np.full(n, 1 / n)
    portfolios.append(("equal", f"Equal weight, all {n}", w_eq, classical.backtest(w_eq, test)))
    report.table([{"m": lab, **bt} for _, lab, _, bt in portfolios],
                 [("m", "portfolio", "{}"), ("cagr", "return/yr", "{:+.1%}"), ("vol", "vol/yr", "{:.1%}"),
                  ("sharpe", "Sharpe", "{:.2f}"), ("max_drawdown", "max drawdown", "{:.1%}")])
    report.show(*viz.fig_backtest(chosen, test_dates, portfolios), figs)

    # ------------------------------------------------------------------ 7. backend
    if run_on_backend:
        report.header(7, "One job on the selected backend")
        backend = globals().get("backend")
        if backend is None:
            import builtins
            backend = getattr(builtins, "backend", None)
        if backend is None:
            print("  No backend was provided (choose one with Select QPU); skipping.")
        else:
            run_backend(backend, problem, solvers[final_variant], r_final, final_variant, figs)

    # ------------------------------------------------------------------ summary
    report.header("✓", "What we learned")
    xy, pen = results["xy"][final_depth - 1], results["penalty"][final_depth - 1]
    lines = [
        f"Exact search found {viz.short(problem.optimum, chosen)} by checking all "
        f"{ex['evaluated']} valid portfolios in {ex['seconds'] * 1e3:.2f} ms: at this size, "
        f"classical wins outright.",
        f"Circuit design matters more than depth: at p={final_depth}, XY-QAOA puts {xy.p_opt:.0%} "
        f"of its probability on the best portfolio ({xy.p_opt / rnd['feasible_only']:.1f}x random "
        f"valid guessing, approximation ratio {xy.approx_ratio:.2f}).",
        f"The textbook penalty recipe {'scores below' if pen.approx_ratio < random_ar else 'barely beats'} "
        f"random valid guessing (ratio {pen.approx_ratio:.2f} vs {random_ar:.2f}): "
        f"{1 - pen.p_feasible:.0%} of its probability lands on portfolios that break the budget.",
    ]
    if noise_study:
        e_ref = min({r["e2"] for r in noise_rows if r["e2"] > 0}, key=lambda e: abs(e - 0.005))
        mid = {(r["variant"], r["p"]): r for r in noise_rows if r["e2"] == e_ref}
        if max_depth > 1:
            shallow, deep = mid[("xy", 1)], mid[("xy", max_depth)]
            verb = "flips" if shallow["approx_ratio"] > deep["approx_ratio"] else "narrows"
            lines.append(f"Noise {verb} the depth trade-off: at {e_ref:.1%} two-qubit error, XY p=1 "
                         f"keeps a ratio of {shallow['approx_ratio']:.2f} vs "
                         f"{deep['approx_ratio']:.2f} at p={max_depth}; discarding invalid samples "
                         f"lifts p=1 back to {shallow['approx_ratio_post']:.2f}.")
    for i, line in enumerate(lines, 1):
        print(textwrap_fill(f"{i}. {line}"))
    print(f"\n  Total run time {time.perf_counter() - t_start:.1f}s.")


def describe_circuit(solver, p, variant):
    """Compact, screen-reader-friendly summary of the circuit sent to the backend."""
    from qport.qaoa import cost_terms, ring_pairs
    n = solver.n
    fields, couplings = cost_terms(solver.H)
    gc = solver.gate_counts(p)
    print(f"  The circuit sent to the backend ({viz.METHOD[variant]['name']}, p={p}, {n} qubits; "
          "Qollab's Circuit panel draws it):")
    if variant == "penalty":
        print("    prepare  : H on every qubit (equal superposition of all "
              f"{2 ** n} bitstrings)")
    else:
        method = solver.init_circuit()[0]
        print(f"    prepare  : Dicke({n},{solver.problem.budget}) state via {method} "
              "(the cheaper of two compiled options)")
    for layer in range(1, p + 1):
        mix = (f"{n} RX(2β{layer})" if variant == "penalty"
               else f"{len(ring_pairs(n))} XX+YY(2β{layer}) on the ring {ring_pairs(n)}")
        print(f"    layer {layer}  : cost = {len(couplings)} RZZ + {len(fields)} RZ (angle γ{layer}); "
              f"mixer = {mix}")
    print(f"    measure  : all {n} qubits")
    print(f"    compiled to CX/RZ/SX/X: {gc['two_qubit']} CNOTs, {gc['one_qubit']} single-qubit "
          f"gates, depth {gc['depth']}")
    print()


def textwrap_fill(text):
    return textwrap.fill(text, report.WIDTH, initial_indent="  ", subsequent_indent="     ")


def run_backend(backend, problem, solver, res, variant, figs):
    from qiskit import transpile
    name = getattr(backend, "name", type(backend).__name__)
    qc = solver.full_circuit(res.p, res.gammas, res.betas, measure=True)
    report.explain(f"""
        Sending the optimized {viz.METHOD[variant]['name']} circuit (p={res.p}) to {name} as a
        single job. Qollab's Circuit and Probabilities panels show this exact job. Every
        evaluation above ran on the in-browser simulator; this run checks the prediction
        against an independent backend.""")
    try:
        tqc = transpile(qc, backend)
    except Exception as e:  # noqa: BLE001 - some backends transpile server-side
        print(f"  (local transpile skipped: {type(e).__name__}; submitting as built)")
        tqc = qc
    t0 = time.perf_counter()
    try:
        counts = backend.run(tqc).result().get_counts()
    except Exception as e:  # noqa: BLE001
        print(f"  Backend run failed: {type(e).__name__}: {e}")
        return
    shots = sum(counts.values())
    measured = np.zeros(2 ** problem.n)
    for bits, c in counts.items():
        measured[int(bits.replace(" ", ""), 2)] += c / shots
    print(f"  {shots} shots in {time.perf_counter() - t0:.1f}s on {name}.\n")
    ideal = res.probs
    tvd = 0.5 * np.abs(ideal - measured).sum()
    seen = [k for k in np.argsort(-measured) if problem.feasible[k] and measured[k] > 0]
    best_seen = min(seen, key=lambda k: problem.costs[k]) if seen else None
    report.table([
        {"q": "P(best portfolio)", "i": f"{ideal[problem.optimum]:.1%}", "m": f"{measured[problem.optimum]:.1%}"},
        {"q": "P(valid portfolio)", "i": f"{ideal[problem.feasible].sum():.1%}",
         "m": f"{measured[problem.feasible].sum():.1%}"},
        {"q": "most frequent valid pick", "i": viz.short(int(np.argmax(np.where(problem.feasible, ideal, 0))),
                                                         problem.tickers),
         "m": viz.short(seen[0], problem.tickers) if seen else "-"},
        {"q": "best valid pick in the shots", "i": viz.short(problem.optimum, problem.tickers),
         "m": viz.short(best_seen, problem.tickers) if best_seen is not None else "-"},
        {"q": "total variation distance", "i": "0", "m": f"{tvd:.3f}"},
    ], [("q", "", "{}"), ("i", "noiseless prediction", "{}"), ("m", f"measured ({name})", "{}")])
    report.show(*viz.fig_hardware(problem, ideal, measured, name, variant, res.p), figs)
    print(f"  Most frequent bitstring: {label(int(np.argmax(measured)), problem.n)} "
          "(qubit 0 = first ticker, rightmost)")

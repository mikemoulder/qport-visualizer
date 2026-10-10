"""Accessible figures for every step of the pipeline.

Design rules (applied everywhere):
- Light chart surface with dark ink, so figures stay readable on Qollab's dark editor.
- One fixed color per method, in a validated colorblind-safe order; identity is
  never carried by color alone (markers, line styles, hatching, direct labels).
- One y-scale per panel. Different measures get their own panel.
- Every figure function returns (fig, description). The caller prints the
  description as a text alternative, next to a numeric table.
"""
import datetime as dt

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .problem import picked

# --- tokens ---------------------------------------------------------------------
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

# Fixed identity per method (validated categorical order: blue, orange, aqua, yellow, magenta).
METHOD = {
    "exact":     {"color": "#2a78d6", "marker": "D", "ls": "-",  "hatch": "",    "name": "Exact (brute force)"},
    "penalty":   {"color": "#eb6834", "marker": "o", "ls": "-",  "hatch": "//",  "name": "QAOA, penalty + X mixer"},
    "xy":        {"color": "#1baf7a", "marker": "s", "ls": "--", "hatch": "",    "name": "QAOA, Dicke + XY mixer"},
    "markowitz": {"color": "#eda100", "marker": "^", "ls": "-.", "hatch": "..",  "name": "Markowitz (continuous)"},
    "naive":     {"color": "#e87ba4", "marker": "v", "ls": ":",  "hatch": "\\\\", "name": "Naive top-Sharpe"},
    "equal":     {"color": MUTED,     "marker": "x", "ls": (0, (1, 3)), "hatch": "xx", "name": "Equal weight (all assets)"},
    "random":    {"color": MUTED,     "marker": "",  "ls": "--", "hatch": "",    "name": "Random valid guess"},
}

BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8e8d", "#b52f2f"])


def style():
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.frameon": False, "legend.fontsize": 9.5, "font.size": 10.5,
        "lines.linewidth": 2, "lines.markersize": 7, "figure.dpi": 100,
        "svg.fonttype": "none",
    })


def short(k, tickers):
    return "+".join(picked(k, tickers)) or "(none)"


def _suptitle(fig, text):
    fig.suptitle(text, fontsize=13.5, fontweight="bold", color=INK, x=0.01, ha="left")


# --- Figure 1: the data -------------------------------------------------------------
def fig_data(tickers, mu, sigma, window):
    vol = np.sqrt(np.diag(sigma))
    corr = sigma / np.outer(vol, vol)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})
    _suptitle(fig, f"1 · The assets: risk, return and how they move together ({window})")
    a.scatter(vol * 100, mu * 100, s=70, color=METHOD["exact"]["color"], edgecolor=SURFACE, zorder=3)
    for t, x, y in zip(tickers, vol * 100, mu * 100):
        a.annotate(t, (x, y), xytext=(6, 4), textcoords="offset points", color=INK, fontsize=10)
    a.set_xlabel("Volatility, % per year (risk)")
    a.set_ylabel("Mean return, % per year")
    a.set_title("Each asset alone", loc="left")
    im = b.imshow(corr, cmap=DIVERGING, norm=TwoSlopeNorm(0, -1, 1))
    b.set_xticks(range(len(tickers)), tickers, rotation=45, ha="right")
    b.set_yticks(range(len(tickers)), tickers)
    b.grid(False)
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            b.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                   color=SURFACE if abs(corr[i, j]) > 0.6 else INK)
    b.set_title("Correlation (low = diversifies)", loc="left")
    fig.colorbar(im, ax=b, fraction=0.046, pad=0.04)
    fig.tight_layout()
    lo = np.unravel_index(np.argmin(corr + np.eye(len(tickers)) * 2), corr.shape)
    best = int(np.argmax(mu / vol))
    desc = (f"Left: scatter of {len(tickers)} assets, volatility vs mean annual return. "
            f"Best return/risk ratio: {tickers[best]} ({mu[best]:.0%} return, {vol[best]:.0%} volatility). "
            f"Right: correlation matrix; the least correlated pair is {tickers[lo[0]]}-{tickers[lo[1]]} "
            f"({corr[lo]:.2f}).")
    return fig, desc


# --- Figure 2: the problem ------------------------------------------------------------
def fig_problem(problem):
    tickers, n = problem.tickers, problem.n
    feas = np.flatnonzero(problem.feasible)
    order = feas[np.argsort(problem.costs[feas])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 5), gridspec_kw={"width_ratios": [1, 1.2]})
    _suptitle(fig, f"2 · The problem: pick {problem.budget} of {n} assets "
                   f"({len(feas)} valid portfolios out of 2^{n} = {2 ** n} bitstrings)")
    Q = problem.Q_obj
    lim = float(np.abs(Q).max())
    im = a.imshow(Q, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim))
    a.set_xticks(range(n), tickers, rotation=45, ha="right")
    a.set_yticks(range(n), tickers)
    a.grid(False)
    for i in range(n):
        for j in range(n):
            a.text(j, i, f"{Q[i, j]:.3f}", ha="center", va="center", fontsize=8,
                   color=SURFACE if abs(Q[i, j]) > 0.6 * lim else INK)
    a.set_title("QUBO matrix Q (objective part)", loc="left")
    a.set_xlabel(f"+ budget penalty: +{problem.penalty:.3f} off-diagonal, "
                 f"{problem.penalty * (1 - 2 * problem.budget):+.3f} on the diagonal", fontsize=9)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)

    vals = problem.costs[order]
    y = np.arange(len(order))[::-1]
    colors = [METHOD["exact"]["color"] if k == problem.optimum else AXIS for k in order]
    b.barh(y, vals, color=colors, edgecolor=SURFACE, linewidth=1, height=0.8)
    b.set_yticks(y, [("★ " if k == problem.optimum else "") + short(k, tickers) for k in order],
                 fontsize=8.5)
    b.set_xlabel("Objective f(x) = risk_aversion·variance − return  (lower is better)")
    b.set_title("Every valid portfolio, ranked", loc="left")
    b.axvline(0, color=AXIS, lw=1)
    b.grid(axis="y", visible=False)
    fig.tight_layout()
    desc = (f"Left: the {n}x{n} QUBO matrix of the objective; diagonal = single-asset terms "
            f"(risk minus return), off-diagonal = pair terms (covariance). The budget penalty "
            f"(weight {problem.penalty:.3f}) is added on top to enforce exactly {problem.budget} picks. Right: all {len(feas)} valid portfolios ranked by objective; "
            f"best is {short(problem.optimum, tickers)} (f = {problem.best_cost:.4f}), "
            f"worst is {short(order[-1], tickers)} (f = {problem.worst_feasible_cost:.4f}).")
    return fig, desc


# --- Figure 3: p=1 landscapes with the optimizer's path -----------------------------
def fig_landscapes(entries):
    """entries: list of (solver, (gs, bs, E), best p=1 DepthResult, cold-start p=1 DepthResult)."""
    fig, axes = plt.subplots(1, len(entries), figsize=(5.6 * len(entries), 5.3))
    axes = np.atleast_1d(axes)
    _suptitle(fig, "3 · The hybrid loop at depth p=1: the quantum circuit scores angles, "
                   "a classical optimizer walks downhill")
    parts = []
    for ax, (solver, (gs, bs, E), res, cold) in zip(axes, entries):
        m = METHOD[solver.variant]
        im = ax.imshow(E, origin="lower", aspect="auto", cmap=BLUES.reversed(),
                       extent=[gs[0], gs[-1], bs[0], bs[-1]])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("expected cost (darker = better)", color=INK_2)
        path = np.array([t[0] for t in cold.trace])
        ax.plot(path[:, 0], path[:, 1], color=SURFACE, lw=3.2, zorder=2)
        ax.plot(path[:, 0], path[:, 1], color=m["color"], lw=1.6, marker=".", ms=5, zorder=3,
                label="optimizer path from a cold start")
        ax.plot(*path[0], marker="o", ms=9, color=SURFACE, markeredgecolor=INK, zorder=4,
                label="cold start (γ, β ≈ 0)")
        ax.plot(cold.gammas[0], cold.betas[0], marker="X", ms=11, color=m["color"],
                markeredgecolor=INK, zorder=4, label="where it stopped")
        ax.plot(res.gammas[0], res.betas[0], marker="*", ms=17, color=SURFACE,
                markeredgecolor=INK, zorder=5, label="best found (grid scan + refine)")
        ax.set_xlabel("γ  (cost-layer angle)")
        ax.set_ylabel("β  (mixer angle)")
        ax.set_title(m["name"], loc="left", fontsize=11)
        ax.grid(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5)
        same = abs(cold.energy - res.energy) < 1e-4 * max(1.0, abs(res.energy))
        parts.append(f"{m['name']}: the cold start took {cold.evaluations} evaluations and "
                     + ("reached the best point" if same else
                        f"stopped in a local minimum (cost {cold.energy:.4f} vs best {res.energy:.4f})")
                     + f"; best γ={res.gammas[0]:.3f}, β={res.betas[0]:.3f}")
    fig.tight_layout()
    return fig, ("Heatmaps of expected cost over the two p=1 angles; darker is lower cost. "
                 "The line traces the classical optimizer's proposals. " + "; ".join(parts) + ".")


# --- Figure 4: probability flowing toward good portfolios ---------------------------
def fig_probability_flow(problem, res, variant, max_rows=80):
    tickers = problem.tickers
    feas = np.flatnonzero(problem.feasible)
    order = feas[np.argsort(problem.costs[feas])]
    steps = np.unique(np.linspace(0, len(res.trace) - 1, min(max_rows, len(res.trace))).astype(int))
    # valid portfolios best -> worst, then one column holding all invalid probability
    M = np.array([np.append(res.trace[s][2][order], 1 - res.trace[s][2][feas].sum()) for s in steps])
    ncol = len(order) + 1
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 7.4), height_ratios=[2.2, 1], sharex=False)
    _suptitle(fig, f"4 · Probability flowing toward good portfolios while the optimizer runs "
                   f"({METHOD[variant]['name']}, p={res.p}, cold start)")
    im = a.imshow(M, aspect="auto", cmap=BLUES, origin="lower", interpolation="nearest",
                  extent=[-0.5, ncol - 0.5, steps[0] + 1, steps[-1] + 2])
    fig.colorbar(im, ax=a, fraction=0.03, pad=0.01).set_label("probability", color=INK_2)
    a.axvline(len(order) - 0.5, color=SURFACE, lw=4)
    a.axvline(len(order) - 0.5, color=INK, lw=1.2, ls="--")
    a.set_title("valid portfolios, best → worst  |  all invalid picks", loc="left", fontsize=10)
    a.set_xticks([0, 1, 2, ncol - 1],
                 [f"★ {short(order[0], tickers)}", short(order[1], tickers),
                  short(order[2], tickers), "invalid"], rotation=20, ha="right", fontsize=8.5)
    a.set_ylabel("optimizer evaluation")
    a.grid(False)

    evals = np.arange(1, len(res.trace) + 1)
    p_opt = [t[2][problem.optimum] for t in res.trace]
    p_valid = [t[2][problem.feasible].sum() for t in res.trace]
    m = METHOD[variant]
    b.plot(evals, p_opt, color=m["color"], ls=m["ls"], label="P(best portfolio)")
    b.plot(evals, p_valid, color=INK_2, ls=":", label="P(valid portfolio)")
    rnd = 1 / len(feas)
    b.axhline(rnd, color=MUTED, ls="--", lw=1.2)
    b.text(evals[-1], rnd, f" random valid guess: {rnd:.1%}", va="bottom", ha="right",
           fontsize=9, color=INK_2)
    b.set_xlabel("optimizer evaluation (each = one run of the quantum circuit)")
    b.set_ylabel("probability")
    b.set_ylim(0, 1.05)
    b.legend(loc="center right")
    fig.tight_layout()
    desc = (f"Top: heatmap; rows are optimizer steps from a cold start (bottom = first), columns "
            f"are the {len(order)} valid portfolios sorted best to worst plus one column for all "
            f"invalid picks. Darker = more likely. Bottom: P(best portfolio) goes from {p_opt[0]:.1%} "
            f"to {res.p_opt:.1%} over {len(res.trace)} evaluations; P(valid) goes from "
            f"{p_valid[0]:.1%} to {res.p_feasible:.1%}; a random valid guess hits the best one "
            f"{rnd:.1%} of the time.")
    return fig, desc


# --- Figure 5: final distributions side by side --------------------------------------
def fig_distribution(problem, finals, top=10):
    """finals: {variant: DepthResult}."""
    tickers = problem.tickers
    feas = np.flatnonzero(problem.feasible)
    order = feas[np.argsort(problem.costs[feas])][:top]
    labels = [("★ " if k == problem.optimum else "") + short(k, tickers) for k in order]
    labels.append("any invalid pick")
    series = [(v, np.append(r.probs[order], 1 - r.p_feasible)) for v, r in finals.items()]
    rnd = np.append(np.full(len(order), 1 / len(feas)), 0.0)
    series.append(("random", rnd))
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    _suptitle(fig, f"5 · Where each method puts its probability: the {top} best valid portfolios "
                   f"(best on the left)")
    width = 0.8 / len(series)
    x = np.arange(len(labels))
    for i, (key, vals) in enumerate(series):
        m = METHOD[key]
        name = m["name"] + (f" (p={finals[key].p})" if key in finals else "")
        bars = ax.bar(x + (i - (len(series) - 1) / 2) * width, vals, width * 0.92, color=m["color"],
                      hatch=m["hatch"], edgecolor=SURFACE, linewidth=0.5, label=name)
        if key in finals:
            ax.bar_label(bars, [f"{v:.0%}" if v >= 0.05 else "" for v in vals], fontsize=8,
                         color=INK, padding=2)
    ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("probability of measuring it")
    ax.set_ylim(0, max(v.max() for _, v in series) * 1.15)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(series), fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    parts = [f"{METHOD[v]['name']}: {r.p_opt:.1%} on the best portfolio, {1 - r.p_feasible:.1%} on "
             f"invalid picks" for v, r in finals.items()]
    desc = ("Grouped bars of measurement probability for the best valid portfolios and for all "
            "invalid picks combined. " + "; ".join(parts) +
            f"; random valid guessing gives each {1 / len(feas):.1%}.")
    return fig, desc


# --- Figure 6: depth sweep ---------------------------------------------------------------
def fig_depth(problem, results_by_variant, random_ar):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    _suptitle(fig, "6 · Does a deeper circuit (more layers p) help? Noiseless simulation")
    rnd_opt = 1 / int(problem.feasible.sum())
    for v, rs in results_by_variant.items():
        m = METHOD[v]
        ps = [r.p for r in rs]
        a.plot(ps, [r.approx_ratio for r in rs], color=m["color"], ls=m["ls"], marker=m["marker"],
               label=m["name"])
        b.plot(ps, [r.p_opt for r in rs], color=m["color"], ls=m["ls"], marker=m["marker"],
               label=m["name"])
    for ax, ref, txt in ((a, random_ar, "random valid guess"), (b, rnd_opt, "random valid guess")):
        ax.axhline(ref, color=MUTED, ls="--", lw=1.2)
        ax.text(ax.get_xlim()[0], ref, f" {txt}", va="bottom", fontsize=9, color=INK_2)
        ax.set_xlabel("QAOA depth p (number of cost + mixer layers)")
        ax.set_xticks(sorted({r.p for rs in results_by_variant.values() for r in rs}))
    a.set_ylabel("approximation ratio (1 = always optimal)")
    a.set_ylim(0, 1.02)
    a.set_title("Solution quality", loc="left")
    b.set_ylabel("P(best portfolio)")
    b.set_ylim(0, None)
    b.set_title("Chance of the single best answer", loc="left")
    a.legend(loc="lower right")
    fig.tight_layout()
    parts = []
    for v, rs in results_by_variant.items():
        parts.append(METHOD[v]["name"] + ": " + ", ".join(f"p={r.p} ratio {r.approx_ratio:.2f}"
                                                         for r in rs))
    desc = ("Line charts of approximation ratio (left) and probability of the best portfolio "
            "(right) against depth. " + "; ".join(parts) + f"; random valid guess ratio {random_ar:.2f}.")
    return fig, desc


# --- Figure 7: noise ----------------------------------------------------------------------
def fig_noise(rows, random_ar):
    depths = sorted({r["p"] for r in rows})
    fig, axes = plt.subplots(1, len(depths), figsize=(4.2 * len(depths), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    _suptitle(fig, "7 · What survives hardware noise? Two-qubit depolarizing error after every CNOT")
    for ax, p in zip(axes, depths):
        for v in ("penalty", "xy"):
            rs = [r for r in rows if r["variant"] == v and r["p"] == p]
            if not rs:
                continue
            m = METHOD[v]
            e = [r["e2"] * 100 for r in rs]
            ax.plot(e, [r["approx_ratio"] for r in rs], color=m["color"], ls=m["ls"],
                    marker=m["marker"], label=f"{m['name']} ({rs[0]['two_qubit']} CNOTs)")
            if v == "xy":
                ax.plot(e, [r["approx_ratio_post"] for r in rs], color=m["color"], ls=":",
                        marker="P", label="  … + discard invalid samples")
        ax.axhline(random_ar, color=MUTED, ls="--", lw=1.2)
        ax.text(0, random_ar, " random valid guess", va="bottom", fontsize=8.5, color=INK_2)
        ax.set_title(f"depth p={p}", loc="left")
        ax.set_xlabel("two-qubit error rate, %")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower left", fontsize=8)
    axes[0].set_ylabel("approximation ratio")
    fig.tight_layout()
    worst = max(depths)
    pick = {(r["variant"], r["p"], r["e2"]): r for r in rows}
    e_mid = min({r["e2"] for r in rows if r["e2"] > 0}, key=lambda e: abs(e - 0.005))
    desc = (f"Small multiples, one per depth; approximation ratio against two-qubit error rate. "
            f"At {e_mid:.1%} error: XY p=1 ratio {pick[('xy', 1, e_mid)]['approx_ratio']:.2f}, "
            f"XY p={worst} {pick[('xy', worst, e_mid)]['approx_ratio']:.2f}, penalty p=1 "
            f"{pick[('penalty', 1, e_mid)]['approx_ratio']:.2f}. Discarding invalid samples "
            f"recovers XY p=1 to {pick[('xy', 1, e_mid)]['approx_ratio_post']:.2f}.")
    return fig, desc


# --- Figure 8: out-of-sample backtest -----------------------------------------------------
def fig_backtest(tickers, test_dates, portfolios):
    """portfolios: list of (key, label, weights, backtest dict)."""
    days = [dt.date.fromisoformat(d) for d in test_dates]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    _suptitle(fig, f"8 · Out of sample: hold each portfolio from {test_dates[0]} to {test_dates[-1]} "
                   f"(data the optimizers never saw)")
    for key, lab, w, bt in portfolios:
        m = METHOD[key]
        a.plot(days, bt["growth"], color=m["color"], ls=m["ls"], lw=2, label=lab)
        a.annotate(f"${bt['growth'][-1]:.2f}", (days[-1], bt["growth"][-1]), xytext=(4, 0),
                   textcoords="offset points", va="center", fontsize=8.5, color=INK)
    a.set_ylabel("value of $1 invested")
    a.set_title("Growth of $1", loc="left")
    a.legend(loc="upper left", fontsize=8.5)
    a.tick_params(axis="x", rotation=30)
    width = 0.8 / len(portfolios)
    x = np.arange(len(tickers))
    for i, (key, lab, w, _) in enumerate(portfolios):
        m = METHOD[key]
        b.bar(x + (i - (len(portfolios) - 1) / 2) * width, w * 100, width * 0.92, color=m["color"],
              hatch=m["hatch"], edgecolor=SURFACE, linewidth=0.5, label=lab)
    b.set_xticks(x, tickers)
    b.set_ylabel("weight, %")
    b.set_title("Allocations", loc="left")
    b.grid(axis="x", visible=False)
    fig.tight_layout()
    desc = ("Left: growth of $1 over the test window for each portfolio. Right: grouped bars of "
            "each portfolio's weights. Final values: " +
            "; ".join(f"{lab} ${bt['growth'][-1]:.2f}" for _, lab, _, bt in portfolios) + ".")
    return fig, desc


# --- Figure 9: backend run vs simulation ------------------------------------------------------
def fig_hardware(problem, ideal, measured, backend_name, variant, p, fitted=None, e_fit=0.0, top=10):
    """Measured shots vs the noiseless prediction and, if given, the best-fit noise model."""
    tickers = problem.tickers
    feas = np.flatnonzero(problem.feasible)
    order = feas[np.argsort(problem.costs[feas])][:top]
    labels = [("★ " if k == problem.optimum else "") + short(k, tickers) for k in order]
    labels.append("any invalid pick")

    def view(pr):
        return np.append(pr[order], 1 - pr[problem.feasible].sum())

    series = [("noiseless prediction", view(ideal), {"color": AXIS, "edgecolor": INK_2, "linewidth": 0.8})]
    if fitted is not None and e_fit > 0:
        series.append((f"our noise model, best fit: {e_fit:.1%} two-qubit error", view(fitted),
                       {"color": SURFACE, "edgecolor": INK_2, "hatch": "///", "linewidth": 0.8}))
    series.append((f"measured on {backend_name}", view(measured),
                   {"color": METHOD[variant]["color"], "edgecolor": SURFACE}))
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    _suptitle(fig, f"9 · One job on {backend_name}: measured shots vs prediction "
                   f"({METHOD[variant]['name']}, p={p})")
    x = np.arange(len(labels))
    width = 0.8 / len(series)
    for i, (name, vals, style) in enumerate(series):
        bars = ax.bar(x + (i - (len(series) - 1) / 2) * width, vals, width * 0.92, label=name, **style)
        if name.startswith("measured"):
            ax.bar_label(bars, [f"{v:.0%}" if v >= 0.05 else "" for v in vals], fontsize=8, padding=2)
    ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("fraction of shots")
    ax.set_ylim(0, max(v.max() for _, v, _ in series) * 1.15)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(series), fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    tvd = 0.5 * np.abs(ideal - measured).sum()
    desc = ("Grouped bars for the best valid portfolios plus all invalid picks: noiseless prediction"
            + (", best-fit noise model" if len(series) == 3 else "") + " and measured shots. "
            f"Best portfolio: predicted {ideal[problem.optimum]:.1%}, measured "
            f"{measured[problem.optimum]:.1%}; invalid picks: measured "
            f"{1 - measured[problem.feasible].sum():.1%}. Total variation distance to the noiseless "
            f"prediction: {tvd:.3f}"
            + (f"; to the noise model at {e_fit:.1%}: {0.5 * np.abs(fitted - measured).sum():.3f}."
               if len(series) == 3 else "."))
    return fig, desc

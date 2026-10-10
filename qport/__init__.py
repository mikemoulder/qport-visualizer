"""QPort Visualizer: hybrid quantum-classical portfolio selection, made visible.

Modules
-------
data       load bundled historical prices, estimate returns and covariance
problem    portfolio selection as a constrained binary problem, QUBO and Ising forms
classical  exact brute force, continuous Markowitz, random-guess baseline, backtest
qaoa       QAOA circuits (X mixer + penalty, XY mixer + Dicke state), hybrid loop
noise      depolarizing density-matrix study: how depth trades off against noise
viz        accessible matplotlib figures for every step
report     console tables and plain-language explanations
"""

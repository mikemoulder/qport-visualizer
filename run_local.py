"""Run QPort Visualizer outside Qollab.

Qollab injects `backend` from its Select QPU dialog and renders plt.show()
inline. Here we stand in Qiskit's BasicSimulator and save each figure as a
PNG under out/ instead.

    python run_local.py                 # defaults
    python run_local.py budget=2 max_depth=2 noise_study=False
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from qiskit.providers.basic_provider import BasicSimulator  # noqa: E402

import main as app  # noqa: E402

OUT = "out"
_count = [0]


def _save_instead_of_show(*_, **__):
    _count[0] += 1
    path = os.path.join(OUT, f"figure_{_count[0]}.png")
    plt.gcf().savefig(path, dpi=110, bbox_inches="tight")
    print(f"  (saved {path})")


def _parse(arg):
    key, value = arg.split("=", 1)
    default = app.main.__defaults__[app.main.__code__.co_varnames.index(key)]
    if isinstance(default, bool):
        return key, value.lower() in ("1", "true", "yes")
    return key, type(default)(value)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    plt.show = _save_instead_of_show
    app.backend = BasicSimulator()
    app.main(**dict(_parse(a) for a in sys.argv[1:]))

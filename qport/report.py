"""Console output: step headers, plain-language explanations, and text tables.

Every figure is paired with a printed description and a numeric table, so the
results are fully available to screen-reader users and in plain-text logs.
"""
import math
import textwrap

import matplotlib.pyplot as plt

WIDTH = 92
_figure_count = [0]


def header(step, title):
    print("\n" + "=" * WIDTH)
    print(f"STEP {step} · {title}")
    print("=" * WIDTH)


def explain(text):
    for para in textwrap.dedent(text).strip().split("\n\n"):
        print(textwrap.fill(" ".join(para.split()), WIDTH, initial_indent="  ",
                            subsequent_indent="  "))
        print()


def table(rows, columns, title=None):
    """rows: list of dicts; columns: list of (key, heading, format)."""
    if title:
        print(f"  {title}")
    cells = [[fmt.format(r[k]) if r.get(k) is not None else "-" for k, _, fmt in columns]
             for r in rows]
    heads = [h for _, h, _ in columns]
    widths = [max(len(h), *(len(c[i]) for c in cells)) for i, h in enumerate(heads)]
    line = "  " + "  ".join(h.ljust(w) for h, w in zip(heads, widths))
    print(line)
    print("  " + "  ".join("-" * w for w in widths))
    for c in cells:
        print("  " + "  ".join(v.ljust(w) for v, w in zip(c, widths)))
    print()


def show(fig, desc, enabled=True):
    _figure_count[0] += 1
    if enabled:
        print(textwrap.fill(f"[Figure, text description] {desc}", WIDTH, initial_indent="  ",
                            subsequent_indent="  "))
        print()
        plt.show()
    plt.close(fig)


def shots_to_solution(p, confidence=0.99):
    """Shots needed to see the best portfolio at least once with the given confidence."""
    if p <= 0:
        return math.inf
    if p >= 1:
        return 1
    return math.ceil(math.log(1 - confidence) / math.log(1 - p))

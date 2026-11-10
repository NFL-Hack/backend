"""Visual: Why the open man was ignored -- funnel from clear-open downfield plays to the
three bucket3 groups, from outputs/buckets.csv (src/combine.py). Output: why_ignored.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import OUT

REPO = os.path.dirname(os.path.dirname(OUT))
BUCKETS = os.path.join(REPO, "outputs", "buckets.csv")

BG, PANEL, FG, MUTED = "#0b1622", "#13263a", "#ffffff", "#9fb3c8"
GROUPS = [  # bucket3, label, colour, sub-buckets
    ("TOO_LATE", "TOO LATE", "#f2a541", ["pressured", "late"]),
    ("NOT_REAL_WINDOW", "NOT A REAL WINDOW", "#8a9bb0", ["blind", "closed"]),
    ("OPEN_WINDOW", "OPEN WINDOW: QB PASSED", "#e63946", ["missed"]),
]
SUB_LABEL = {"pressured": "pressured", "late": "opened too late", "blind": "out of sight",
             "closed": "not open at throw", "missed": "in time, in sight, open"}


def stats():
    b = pd.read_csv(BUCKETS)
    co = b[b.analyzed]
    ign = co[~co.is_target]
    return len(co), len(ign), ign.bucket3.value_counts(), ign.bucket.value_counts()


def main():
    n_co, n_ign, b3, sub = stats()
    print(f"clear-open {n_co}, ignored {n_ign}; " +
          ", ".join(f"{g} {b3.get(g, 0)} ({b3.get(g, 0) / n_ign:.1%})" for g, *_ in GROUPS))
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(12, 7.2), facecolor=BG)
    ax = fig.add_axes([0.03, 0.12, 0.94, 0.64], facecolor=BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.25, 3.2)
    ax.axis("off")

    # Row 1: all clear-open plays. Row 2: ignored. Row 3: split of ignored.
    ign_w = n_ign / n_co
    ax.barh(2.7, 1.0, height=0.5, color=PANEL, edgecolor="#2d4a66")
    ax.text(0.015, 2.7, f"{n_co:,}", color=FG, fontsize=26, fontweight="bold", va="center")
    ax.text(0.16, 2.7, "plays with a clear-open downfield receiver", color=MUTED, fontsize=14,
            va="center")
    x0 = (1 - ign_w) / 2
    ax.barh(1.85, ign_w, left=x0, height=0.5, color="#1f3b57", edgecolor="#2d4a66")
    ax.text(x0 + 0.015, 1.85, f"{n_ign:,}", color=FG, fontsize=26, fontweight="bold", va="center")
    ax.text(x0 + 0.15, 1.85, f"ignored ({n_ign / n_co:.1%}): ball went elsewhere", color=MUTED,
            fontsize=14, va="center")
    for y0, y1, w0, w1 in [(2.45, 2.1, 1.0, ign_w)]:
        l0, l1 = (1 - w0) / 2, (1 - w1) / 2
        ax.fill([l0, l0 + w0, l1 + w1, l1], [y0, y0, y1, y1], color="#1a3047", lw=0)

    left = x0
    for key, label, col, subs in GROUPS:
        n = int(b3.get(key, 0))
        w = ign_w * n / n_ign
        ax.barh(0.9, w, left=left, height=0.6, color=col, edgecolor=BG, lw=2)
        ax.text(left + w / 2, 0.98, f"{n / n_ign:.1%}", color=BG if key != "NOT_REAL_WINDOW" else FG,
                fontsize=24, fontweight="bold", ha="center", va="center")
        ax.text(left + w / 2, 0.72, f"{n:,} plays", color=BG if key != "NOT_REAL_WINDOW" else FG,
                fontsize=11, ha="center", va="center")
        ax.text(left + w / 2, 0.42, label, color=col, fontsize=12.5, fontweight="bold", ha="center")
        parts = "\n".join(f"{SUB_LABEL[s]} {sub.get(s, 0) / n_ign:.1%}" for s in subs)
        ax.text(left + w / 2, 0.33, parts, color=MUTED, fontsize=10.5, ha="center", va="top",
                linespacing=1.5)
        left += w
    ax.fill([x0, x0 + ign_w, x0 + ign_w, x0], [1.6, 1.6, 1.2, 1.2], color="#1a3047", lw=0)

    n_open = int(b3.get("OPEN_WINDOW", 0))
    fig.text(0.03, 0.9, f"{n_open / n_ign:.0%} of ignored open men were in the QB's sight, "
             f"before pressure,\nwith time to throw, and still open. The QB passed anyway.",
             color=FG, fontsize=21, fontweight="bold", va="center")
    fig.text(0.03, 0.795, "Why QBs ignored a clear-open downfield receiver, 2021 weeks 1-8",
             color=MUTED, fontsize=13)
    fig.text(0.03, 0.03,
             "Clear open downfield: 5+ yds past the LOS and most open by 2+ yds (projected 0.7 s). "
             "7 plays dropped: no snap event.\nToo late: window opened at/after the first pressure "
             "(rusher within 2 yds) or < 0.4 s before the throw. Not a real window: outside the QB's "
             "90° sight cone,\nor under 3 yds of separation at the throw. Descriptive, not causal: "
             "the QB's read and progression also drive where the ball goes.",
             color=MUTED, fontsize=9.2)
    fig.savefig(os.path.join(OUT, "why_ignored.png"), dpi=150, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    main()
